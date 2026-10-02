# 阶段05：完整 Ring AllReduce 与完成关系

> 保留的历史学习资料，不定义当前任务或验收。执行范围与进度以[核心复现计划](../plans/Mycroft_26日开发路线图.md)为准；源码解释按本文标注的版本阅读，候选观测点仍需真实运行验证。

本阶段把[阶段02](阶段02_从Host任务到GPU_Ring_AllReduce执行.md)的执行安排、[阶段03](阶段03_GPU_Proxy_NET执行与Mycroft进度观测.md)的跨机器传输、[阶段04](阶段04_接收端Proxy到GPU规约与转发.md)的接收与转发接成一次完整 AllReduce。

源码基线为 `third_party/nccl` 中的 NCCL `v2.21.5-1`。本文使用 Ring、SIMPLE、普通 buffer 路径。向量和分块参数用于手工推演，并不表示小消息在实际调优中一定会得到这些参数。

## 1. 先看这一次通信到底要做什么

应用的四个 rank 各有一张 GPU，每张 GPU 上都有完整的八元素输入向量：

| rank | chunk 0 | chunk 1 | chunk 2 | chunk 3 |
|---|---|---|---|---|
| 0 | `[1,2]` | `[3,4]` | `[5,6]` | `[7,8]` |
| 1 | `[10,20]` | `[30,40]` | `[50,60]` | `[70,80]` |
| 2 | `[100,200]` | `[300,400]` | `[500,600]` | `[700,800]` |
| 3 | `[1000,2000]` | `[3000,4000]` | `[5000,6000]` | `[7000,8000]` |

操作为 `ncclSum`，每个 rank 最终都应得到：

```text
[1111,2222,3333,4444,5555,6666,7777,8888]
```

为了逐行对应源码，先设一个 Channel 的 Ring 顺序为：

```text
rank 0 -> rank 1 -> rank 2 -> rank 3 -> rank 0
```

每个 chunk 有两个元素，一次 Ring 外层循环处理四个 chunk。对这条 Ring，`ring->index` 恰好分别是 0、1、2、3。

应用提交任务，CPU 安排执行，GPU 决定当前处理哪个 chunk；GPU、Proxy 和网络共同完成每一跳。得到完整向量后，应用通过 stream 的完成语义使用结果。

## 2. 阶段02、03、04在同一次操作中的位置

```text
四个rank的应用各自调用ncclAllReduce
                 |
                 v
Host记录任务、选择Ring/SIMPLE、划分Channel数据、组织kernel plan
                 |                                      阶段02
                 v
GPU kernel进入runRing，决定先发哪个chunk、之后收哪个chunk
                 |
                 | 每一次发送/接收展开为：
                 v
GPU写发送buffer -> 发布tail -> 发送Proxy -> NET网络请求
                 |                                      阶段03
                 v
接收Proxy确认完成 -> 发布接收tail -> GPU读取/规约/转发
                 |                                      阶段04
                 v
各chunk规约完成，再传播到每个rank
                 |
                 v
各Channel处理完各自范围，GPU执行结束，应用观察完成
                                                        阶段05
```

图中的 Proxy 工作与 GPU 工作并发推进。GPU 与 CPU Proxy 通过 buffer 和共享进度交接；函数调用关系主要位于各自的执行侧。

## 3. 固定版本中有哪些对象

| 对象 | 类型与作用 |
|---|---|
| `struct ncclKernelPlan` | Host 结构体，组织一次 kernel launch 的 work、Channel 和 Proxy 工作 |
| `struct ncclWork` | GPU work 容器，带 work header 和 work element |
| `struct ncclWorkElem` | collective 的 GPU 参数结构体，包含 buffer 指针、数据范围和 chunk 参数 |
| `struct ncclRing` | 当前 Channel 的 Ring 信息结构体，保存 `prev`、`next`、`index` |
| `prims` | GPU 中的 `Primitives<...>` 局部对象，提供发送、接收、规约成员函数 |
| `struct ncclProxyOp` | Host 提交的 Proxy 工作描述 |
| `struct ncclProxyArgs` | Proxy 运行时工作结构体，组织多个连接 sub |
| `struct ncclProxySubArgs` | 某条连接子工作的进度和异步请求结构体 |

阶段02部分内容采用了另一版本中的 `ncclDevWorkColl` 等名称。本文直接使用 2.21.5 的 `ncclWork` / `ncclWorkElem`。理解对象用途后，定位时仍以实际 tag 的类型和字段为准。

一个 kernel plan 可以组织多次 collective 的 work；一次 collective 又可以使用多个 Channel。阅读时要保留这两个层次。

## 4. `runRing()`如何找到要处理的数据

入口在 [`src/device/all_reduce.h`](../../third_party/nccl/src/device/all_reduce.h)，搜索 `void runRing`：

```cpp
__device__ __forceinline__ void runRing(ncclWorkElem *args)
```

`args` 是指向 GPU work element 的指针。本例中的主要字段和局部变量为：

```text
args->sendbuff：当前rank完整输入向量的指针
args->recvbuff：当前rank完整输出向量的指针
gridOffset    = args->workOffset = 0
channelCount  = args->workCount  = 8
chunkCount    = args->chunkCount = 2
nranks        = 4
loopCount     = nranks * chunkCount = 8
```

源码创建 primitive 对象：

```cpp
Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
  (tid, nthreads, &ring->prev, &ring->next,
   args->sendbuff, args->recvbuff, args->redOpArg);
```

这个对象取得前驱、后继和输入输出指针。后续成员函数将根据当前操作组织等待、复制或规约、进度发布。

取某个 chunk 时，核心计算为：

```cpp
chunkOffset = chunk * chunkCount;
offset = gridOffset + elemOffset + chunkOffset;
nelem = (int)min(chunkCount, remCount - chunkOffset);
```

本例只有一轮，`elemOffset=0`，所以：

| chunk | offset | nelem | 对应元素 |
|---|---:|---:|---|
| c0 | 0 | 2 | 第0、1个元素 |
| c1 | 2 | 2 | 第2、3个元素 |
| c2 | 4 | 2 | 第4、5个元素 |
| c3 | 6 | 2 | 第6、7个元素 |

offset 的单位是元素。对于 `float* input`，`input + 2` 指向第2个元素，实际跨过 `2*sizeof(float)` 字节。

## 5. 第一段：`send()`启动规约链

源码：

```cpp
chunk = modRanks(ringIx + nranks - 1);
// 计算offset、nelem
prims.send(offset, nelem);
```

对四个 rank，第一次向后继发送：

| 发送者 | chunk | 发送数据 | 接收者 |
|---|---|---|---|
| rank 0 | c3 | `[7,8]` | rank 1 |
| rank 1 | c0 | `[10,20]` | rank 2 |
| rank 2 | c1 | `[300,400]` | rank 3 |
| rank 3 | c2 | `[5000,6000]` | rank 0 |

每个 rank 都只发送自己完整输入中的一段。只追踪 c0：

```text
rank 1输入中的[10,20]
  -> rank 1 GPU写发送buffer并发布tail
  -> rank 1发送Proxy提交isend
  -> rank 2接收Proxy完成接收并发布tail
  -> rank 2 GPU可以读取[10,20]
```

发送 primitive 将输入复制到通信路径，数据的网络推进由 Proxy 独立完成。

## 6. 第二段：`recvReduceSend()`加入中间rank的贡献

源码：

```cpp
for (int j = 2; j < nranks; ++j) {
    chunk = modRanks(ringIx + nranks - j);
    // 计算offset、nelem
    prims.recvReduceSend(offset, nelem);
}
```

四 rank 时，这个循环执行两次：`j=2`、`j=3`。

| 本rank | `j=2`处理 | `j=3`处理 |
|---|---|---|
| rank 0 | c2 | c1 |
| rank 1 | c3 | c2 |
| rank 2 | c0 | c3 |
| rank 3 | c1 | c0 |

因此 c0 的数据继续变化：

```text
rank 2收到[10,20]
  + rank 2本地c0 [100,200]
  = [110,220]
  -> 发送给rank 3

rank 3收到[110,220]
  + rank 3本地c0 [1000,2000]
  = [1110,2220]
  -> 发送给rank 0
```

GPU 侧的工作顺序是等待接收数据和发送空间、逐元素规约、写发送目标、发布接收 head 与发送 tail。每个 rank 的前驱数据就绪时间可以不同，表中的逻辑步骤不要求全局同时开始。

## 7. 第三段：最后一次规约与第一次结果传播融合

源码：

```cpp
chunk = ringIx;
// 计算offset、nelem
prims.directRecvReduceCopySend(offset, offset, nelem, true);
```

这次操作对当前负责的 chunk 完成三件事：

1. 接收部分和，加入本 rank 的最后一份贡献。
2. 将最终结果写进本 rank 的 `recvbuff` 对应位置。
3. 将最终结果发送给后继，开始传播完整结果。

对 rank 0 的 c0：

```text
收到部分和：[1110,2220]
加入本地值：[   1,   2]
最终结果：  [1111,2222]

写recvbuff[0:2]
同时向rank 1发送[1111,2222]
```

四个最终 chunk 的首次生成位置：

| 生成位置 | chunk | 最终值 |
|---|---|---|
| rank 0 | c0 | `[1111,2222]` |
| rank 1 | c1 | `[3333,4444]` |
| rank 2 | c2 | `[5555,6666]` |
| rank 3 | c3 | `[7777,8888]` |

这就是 ReduceScatter 的结果分散状态。不过源码生成完整 chunk 后立刻发送，没有停下来等待所有 rank 到达一个全局阶段边界。

SIMPLE 的此成员函数内部有注释 `Direct is only for the send part`，实际调用 `genericOp<0,1,1,1,Input,Output>`。是否使用 direct buffer 访问还取决于连接能力和标志，不能只看函数名判断一定绕过普通 buffer。

## 8. 第四段：`directRecvCopySend()`保存并传播完整结果

源码：

```cpp
for (int j = 1; j < nranks - 1; ++j) {
    chunk = modRanks(ringIx + nranks - j);
    // 计算offset、nelem
    prims.directRecvCopySend(offset, nelem);
}
```

四 rank 时也是两次。此时接收的是已经规约完成的结果，操作为复制到输出并继续转发。

对 c0：

```text
rank 1从rank 0收到[1111,2222]
  -> 保存到自己的recvbuff[0:2]
  -> 原值转发给rank 2

rank 2从rank 1收到[1111,2222]
  -> 保存到自己的recvbuff[0:2]
  -> 原值转发给rank 3
```

传播过程中没有再次加入本地输入，否则会重复计入贡献。

## 9. 第五段：`directRecv()`补齐最后一个chunk

源码：

```cpp
chunk = modRanks(ringIx + 1);
// 计算offset、nelem
prims.directRecv(offset, nelem);
```

每个 rank 接收自己最后缺少的完整 chunk，写入输出，本轮不再向后继转发该 chunk。

对 c0：

```text
rank 3从rank 2收到[1111,2222]
  -> 保存到recvbuff[0:2]
```

c0 已覆盖全部 rank，c1、c2、c3 也按对应路径完成传播。四张 GPU 的输出都成为八元素完整结果。

## 10. 将全部primitive排在一张表里

| 代码动作 | rank 0处理 | rank 1处理 | rank 2处理 | rank 3处理 | 对该chunk做什么 |
|---|---|---|---|---|---|
| `send` | c3 | c0 | c1 | c2 | 发送初始贡献 |
| `recvReduceSend, j=2` | c2 | c3 | c0 | c1 | 收、规约、发 |
| `recvReduceSend, j=3` | c1 | c2 | c3 | c0 | 收、规约、发 |
| `directRecvReduceCopySend` | c0 | c1 | c2 | c3 | 最终规约、存结果、发结果 |
| `directRecvCopySend, j=1` | c3 | c0 | c1 | c2 | 收完整结果、存、发 |
| `directRecvCopySend, j=2` | c2 | c3 | c0 | c1 | 收完整结果、存、发 |
| `directRecv` | c1 | c2 | c3 | c0 | 收最后结果、存 |

本例每 rank 有六次 chunk 发送和六次 chunk 接收。表格有七行，是因为起始只有发送、结束只有接收，中间的成员函数同时组织多个动作。

算法通信轮次为 `2*(nranks-1)=6`。这些算法轮次与 Proxy 的协议 step 计数之间还隔着 `chunkSteps`、`sliceSteps` 等单位转换。

## 11. 一个Channel的数据超过四个chunk怎么办

外层循环：

```cpp
for (ssize_t elemOffset = 0; elemOffset < channelCount;
     elemOffset += loopCount) {
    // 上述五段primitive
}
```

若 Channel 负责16个元素，`chunkCount=2`，`nranks=4`：

```text
第1轮：elemOffset=0，处理元素[0,8)
第2轮：elemOffset=8，处理元素[8,16)
```

每轮临时用 c0—c3 表示该轮中的四段，实际全局元素位置还包含 `gridOffset + elemOffset`。同一个“c0”局部编号可以对应另一轮的数据。

当尾部不足 `loopCount` 时，源码使用 `lastChunkCount` 调整最后一轮，并用 `nelem` 限制实际元素数。因此 chunk 数量可以超过 rank 数；rank 数控制一轮 Ring 的组织方式。

## 12. chunk、slice、step和slot的依赖关系

```text
Channel负责一段完整输入范围
  -> runRing按轮次组织chunk
  -> primitive按slice处理chunk中的数据
  -> 每个slice推进若干协议step
  -> 协议step通过循环slot/FIFO选择buffer位置
```

| 名字 | 用途 | 关键单位 |
|---|---|---|
| Channel work范围 | 分配这个Channel处理的输入区间 | 元素 |
| Ring chunk | 决定本轮规约或传播哪一段 | `chunkCount`个元素 |
| slice | primitive内部的数据处理片段 | 实际slice元素数 |
| step | GPU/Proxy连接进度 | 协议step |
| slot | 循环buffer或FIFO中的位置 | step取模得到的索引 |

`computeCollSteps()` 对 Ring AllReduce 计算：

```cpp
steps = DIVUP(workCount, nRanks * chunkCount)
      * (nRanks - 1) * 2 * chunkSteps;
```

读作：Ring 外层循环轮数 × 每轮通信轮次 × 每次 chunk 对应的协议 step 数。

例如一轮四 rank，`chunkSteps=4`、`sliceSteps=2` 时，单方向 Proxy 的 `nsteps=24`，按 slice 单位计为12次推进。它们对应六次 chunk 发送，单位分别是24个 step、12个 slice、6个 Ring chunk。

参数用于说明单位关系；真实 slice 中还可能有尾部或空数据处理。

## 13. 多个Channel怎样共同覆盖输出

先把输入扩展为16个元素，示意分配为：

```text
Channel 0：workOffset=0，workCount=8
Channel 1：workOffset=8，workCount=8
```

每个 Channel 独立执行对应范围的 Ring：

```text
当前rank的recvbuff
  [0,8)  <- Channel 0的完整结果
  [8,16) <- Channel 1的完整结果
```

Host 的 `addCBDCollToPlan()` / `addTunedCollToPlan()` 负责范围分配，`setCollWorkElem()` 写入 `workCount`、`workOffset`、`lastChunkCount`。

launch时：

```cpp
dim3 grid = {(unsigned)plan->channelCount, 1, 1};
```

`ncclKernelMain()` 根据 `channelMask` 将 block 对应到实际 Channel，并加载该 Channel 的 work。

```text
一个kernel launch
  |-- block 0 -> 一个有效Channel -> 负责的范围
  \-- block 1 -> 另一个有效Channel -> 负责的范围
```

Channel ID 可以不连续，block 编号按 mask 中有效位排列。各 Channel 的 Ring 顺序也可能不同；`ring->index` 是本 rank 在该 Ring 中的位置，只有本文固定顺序中才恰好等于 rank。

多个 Channel 可并行推进，但具体 GPU 调度和共享网络资源影响实际重叠程度。

## 14. “完成”到底在哪一层

| 观察位置 | 可以确认什么 |
|---|---|
| 应用 `ncclAllReduce()` 成功返回 | 普通调用已完成相应提交；group中还需按group语义确认提交 |
| NET发送 `test(done)` | 某个本地网络发送request完成 |
| 发送 `sub->done == sub->nsteps` | 这条发送子工作的协议step处理完 |
| 接收 `sub->done == sub->nsteps` | 普通buffer路径中这条接收子工作的消费确认完 |
| `args->done == args->nsubs` | 这一组Proxy工作的sub全部完成 |
| 某个Channel执行完当前work | 该Channel当前范围的GPU处理完 |
| kernel结束 | 该launch内全部block执行结束，可能包含多次collective work |
| 对应stream同步成功，或其后的CUDA event完成 | 应用可确认本rank已提交的相应GPU操作完成并使用输出 |

源码中的 `args->state = ncclProxyOpNone` 对应 Proxy 工作结束。它没有自动给出一个覆盖整次 collective 的GPU完成时间。

`ncclKernelMain()` 的 work 循环先执行 work，再在 `header.isLast` 成立时退出。`isLast` 表示这个 Channel 的 work 链结束；一个 plan 可以有多个 collective，所以 kernel结束时间也可能覆盖多次操作。

另外，`workFifoDone` 在执行相关work之前便可更新，它确认 work描述已读取、FIFO描述空间可回收，不能当成通信数据完成。

## 15. 应用如何确认结果可用

常规调用示意，输入输出均为GPU指针，实际程序应检查返回值：

```cpp
ncclAllReduce(sendbuff, recvbuff, count,
              ncclFloat, ncclSum, comm, stream);

cudaStreamSynchronize(stream);
// 同步成功后，本rank可以读取或使用recvbuff结果。
```

也可以在 NCCL 工作之后向相同 stream 记录 CUDA event，由应用查询或等待这个 event，确认它之前的工作已完成。

这里的完成证据属于本 rank。其他 rank 也会完成同一次 collective，但本 rank 的同步返回不会替代其他进程的完成记录。

group 中的调用要先完成 group 的提交，再按相应 stream 观察完成。对于非阻塞 communicator，还应先确认后台提交状态成功。官方说明见 [CUDA Stream Semantics](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/usage/streams.html) 和 [Group Calls](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/usage/groups.html)。

## 16. kernel退出和Proxy收尾为什么要分别理解

SIMPLE primitive 的析构函数 `~Primitives()` 会同步线程、保存 connection step，供后续操作继续使用。

普通 buffer 路径中，用户输入已复制到 NCCL 管理的通信 buffer，一些发送 request 的 Proxy 收尾可以具有自己的生命周期；析构函数没有统一等待每条普通发送 request 全部完成的循环。

注册用户 buffer 的 `UserBufferMode` 路径则包含专门等待：GPU 等 Proxy 重置 FIFO size，保证后续 kernel 不会过早覆盖仍被网络直接访问的用户 buffer。

因此需要分别观察GPU结果完成、网络request完成和Proxy资源收尾。采集器若只统计Proxy的所有sub结束，首先应将它解释为所覆盖Proxy工作的完成；与整次collective完成的对应关系需要单独建立。

## 17. 两个Channel中的阻塞怎样影响整次操作

假设 Channel 0 完成，而 Channel 1 的 GPU 正在等待前驱数据：

```text
Channel 0处理完输出[0,8)
Channel 1尚未处理完输出[8,16)
                  |
                  v
当前kernel仍有block未结束
                  |
                  v
该stream后续依赖这次通信的工作继续等待
```

即使最后完成的是 Channel 1，根因也可能在它的前驱、前驱的发送Proxy或更远的依赖处。阶段06需要记录操作、Channel、连接与进度，才能沿这条链追踪。

## 18. 源码阅读入口与回顾

| 阅读问题 | 文件 | 搜索词 |
|---|---|---|
| GPU参数结构是什么 | [`src/include/device.h`](../../third_party/nccl/src/include/device.h) | `struct ncclWorkElem`、`struct ncclWork` |
| 数据范围如何分给Channel | [`src/enqueue.cc`](../../third_party/nccl/src/enqueue.cc) | `addCBDCollToPlan`、`addTunedCollToPlan`、`setCollWorkElem` |
| 如何计算Proxy step数 | [`src/enqueue.cc`](../../third_party/nccl/src/enqueue.cc) | `computeCollSteps` |
| kernel如何启动 | [`src/enqueue.cc`](../../third_party/nccl/src/enqueue.cc) | `ncclLaunchKernel` |
| block如何找到Channel及退出 | [`src/device/common.h`](../../third_party/nccl/src/device/common.h) | `ncclKernelMain`、`workFifoDone`、`header.isLast` |
| 五段Ring处理顺序 | [`src/device/all_reduce.h`](../../third_party/nccl/src/device/all_reduce.h) | `runRing` |
| primitive如何等待、处理和结束 | [`src/device/prims_simple.h`](../../third_party/nccl/src/device/prims_simple.h) | `genericOp`、`waitPeer`、`postPeer`、`~Primitives` |
| Proxy如何完成其工作 | [`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc) | `sendProxyProgress`、`recvProxyProgress`、`args->done` |

复习时先沿 c0 走一遍：rank 1初始贡献、rank 2和3中间规约、rank 0最终规约，再传播到rank 1、2、3；然后展开其中一跳的GPU/Proxy协作；最后看多个Channel如何共同覆盖输出。

AllReduce的接口语义可查 [NCCL 2.21.5 Collective Communication Functions](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/api/colls.html#ncclallreduce)。源码执行与本文手工推演不代表已经完成动态插桩验收。

下一篇：[阶段06：Mycroft 的操作身份、进度观测与阻塞依赖](阶段06_Mycroft操作身份_进度观测与阻塞依赖.md)。

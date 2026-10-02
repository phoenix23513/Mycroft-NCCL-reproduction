# 阶段03：GPU—Proxy—NET 执行与 Mycroft 进度观测

> 保留的历史学习资料，不定义当前任务或验收。执行范围与进度以[核心复现计划](../plans/Mycroft_26日开发路线图.md)为准；源码解释按本文标注的版本阅读，候选观测点仍需真实运行验证。

本文承接《阶段02：从 Host 任务到 GPU Ring AllReduce 执行》。阶段02结束于 GPU kernel 进入 `runRing()` 并调用通信 primitive；本文继续解释一个 SIMPLE 协议的数据片段如何经过 GPU、连接缓冲区、Proxy 和 NET 插件，并说明 Mycroft 的 `GPU_ready`、`RDMA_transmitted`、`RDMA_done` 可以从哪些源码状态中推导。

本文是源码学习笔记，不代表真实插桩已经完成。源码基线固定为仓库 submodule 中的 NCCL `v2.21.5-1`。所有需要跨 rank 或实机确认的结论均明确标为“候选”或“待动态验证”。

## 1. 本阶段在完整流程中的位置

阶段02已经梳理：

```text
ncclAllReduce()
  -> Host task / devWork / kernel plan
  -> CUDA kernel
  -> 每个有效 Channel 的 CUDA block
  -> runRing()
```

本文继续：

```text
runRing()
  -> Primitives
  -> waitPeer()：等待连接条件并选择缓冲区位置
  -> GPU 读取、复制或规约数据
  -> postPeer()：发布 GPU 进度
  -> sendProxyProgress() / recvProxyProgress()
  -> ncclNet->isend() / irecv()
  -> ncclNet->test()
  -> 更新 head / tail
  -> 下一个 GPU 或 Proxy 阶段继续
```

这里同时存在两条流：

```text
数据流：用户 GPU buffer -> 连接 buffer -> NIC -> 网络 -> NIC -> 接收连接 buffer -> GPU

控制流：step / slot / FIFO size / tail / head / request / transmitted / done
```

数据流搬运字节，控制流回答“数据是否准备好、网络请求是否提交、传输是否完成、缓冲区是否可以复用”。

## 2. 固定场景：只追踪一个数据片段

本文使用以下场景：

```text
两台机器，每台一张 GPU
rank 0 -> rank 1
Ring AllReduce
NET Transport
SIMPLE 协议
普通 NCCL 连接缓冲区
```

当前需要发送的数据为两个 `float`：

```text
rank 0 用户输入：[1.0, 2.0]
元素数：2
字节数：2 * sizeof(float) = 8
```

为便于说明，假设这个数据片段占用一个传输 step。真实 NCCL 中一个 chunk 可以继续拆成多个 slice，一个 slice 也由 `sliceSteps` 等参数决定占用多少协议 step，不能永久把 chunk、slice 和 step 画等号。

## 3. Channel、connection 和 slot 的结构关系

### 3.1 Channel

Channel 是 NCCL 的软件逻辑通信通道。每个 Channel 至少关联：

- 当前 Channel 负责的数据范围；
- Ring 或 Tree 拓扑；
- 当前 rank 的前驱和后继；
- 到 peer 的发送、接收 connector；
- GPU work 和可能需要的 Proxy work。

Channel 驱动 GPU、Proxy、PCIe、NIC 和网络等硬件资源，但它自身不是物理网线、网卡或 GPU。

### 3.2 connection 与 connector

对某个 Channel、peer 和方向，NCCL 保存 connector：

```text
comm
  \-- channels[channelId]
        \-- peers[peer]
              |-- send[connIndex]
              \-- recv[connIndex]
```

发送 connector 和接收 connector 分别保存连接状态。一个 connector 中与本阶段最相关的两部分是：

```text
connector
  |-- ncclConnInfo
  |     |-- buffs[]
  |     |-- head / tail
  |     |-- connFifo
  |     \-- step
  |
  \-- ncclProxyConnector
        |-- connection
        \-- proxyProgress
```

GPU primitive 使用 `ncclConnInfo` 中的 buffer 和进度字段；Proxy 使用 `ncclProxyConnector` 找到 transport 连接和进度函数。二者通过同一条连接的共享控制状态协作。

### 3.3 slot

SIMPLE 协议连接 buffer 被循环划分为多个可复用位置：

```text
conn->buffs[NCCL_PROTO_SIMPLE]

+--------+--------+--------+-----+
| slot 0 | slot 1 | slot 2 | ... |
+--------+--------+--------+-----+
```

`slot` 是学习中对这些物理缓冲区位置的称呼，不是一个单独的 NCCL C 结构体类型。当前 step 使用的位置通常由下式确定：

```c
slot = step % NCCL_STEPS;
```

在 `prims_simple.h` 中，对应的指针计算是：

```c
connEltsFifo + (step % NCCL_STEPS) * connStepSize
```

逻辑 step 持续增加，有限数量的 slot 循环复用。

## 4. `ncclConnInfo` 保存什么

NCCL 2.21.5 中的关键结构：

```c
struct ncclConnInfo {
  char *buffs[NCCL_NUM_PROTOCOLS];
  void *mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t *tail;
  uint64_t *head;
  int flags;
  int shared;
  int stepSize;
  void **ptrExchange;
  uint64_t *redOpArgExchange;
  struct ncclConnFifo *connFifo;
  uint64_t step;
  ...
};
```

源码：[`third_party/nccl/src/include/device.h`](../../third_party/nccl/src/include/device.h)

本阶段关注：

| 字段 | 含义 |
|---|---|
| `buffs[protocol]` | 当前协议使用的数据缓冲区指针 |
| `head` | 消费/释放进度，生产者据此判断旧 slot 能否复用 |
| `tail` | 生产/发布进度，消费者据此判断新数据是否可用 |
| `stepSize` | SIMPLE buffer 中一个协议 step 对应的空间大小 |
| `connFifo` | GPU 与 Proxy 交换每个 step 的 size、offset、mode 等信息 |
| `step` | 连接保存的当前协议进度 |

`head` 和 `tail` 是指向 64 位进度计数器的指针，不是用户数组首尾地址。

### 4.1 为什么必须有两个进度

连接 buffer 是生产者—消费者循环队列：

```text
tail：生产者已经准备到哪里
head：消费者已经处理并释放到哪里
```

因此：

```text
等待消费的数据量 = tail - head
可继续生产的空间 = buffer容量 - (tail - head)
```

发送方向：

```text
GPU 是生产者：写数据后推进 tail
Proxy 是消费者：网络完成后推进 head
```

接收方向：

```text
Proxy 是生产者：网络数据可供 GPU 使用后推进 tail
GPU 是消费者：读取或规约完成后推进 head
```

这是一条通用规则；具体指针位于本地还是远端映射内存，需要结合发送/接收 connector 和 transport map 阅读。

## 5. NCCL 2.21.5 的 Ring primitive 顺序

NCCL 2.21.5 的 `runRing()` 为当前 Channel 创建 `Primitives`，然后按 Ring AllReduce 顺序调用：

```text
send()
  -> 第一步发送本地 chunk

recvReduceSend()
  -> 接收前驱数据，与本地数据规约，再发送给后继

directRecvReduceCopySend()
  -> 完成当前 rank 负责 chunk 的最终规约，写入 recvbuff，并发送结果

directRecvCopySend()
  -> 接收最终结果，写入 recvbuff，再继续转发

directRecv()
  -> 接收最后缺少的最终 chunk
```

源码：[`third_party/nccl/src/device/all_reduce.h`](../../third_party/nccl/src/device/all_reduce.h)

本地较新 master 曾出现 `directSend()`、`directRecvReduceDirectSend()` 等命名。阶段03以目标版本 2.21.5 为准，不能把 master 中的函数名和行号直接当作 2.21.5 结论。

## 6. `Primitives` 在这一层解决什么问题

Ring算法决定：

```text
当前处理哪个 chunk
从哪个 peer 接收
向哪个 peer 发送
执行复制还是规约
```

SIMPLE `Primitives` 负责：

```text
取得 send/recv connection
等待 head 或 tail 满足条件
选择当前 slot
设置输入、输出指针
组织 GPU 线程复制或规约
发布新的连接进度
```

关键执行骨架：

```text
primitive方法
  -> genericOp()
       -> waitPeer()
       -> GPU reduceCopy/复制
       -> postPeer()
```

源码：[`third_party/nccl/src/device/prims_simple.h`](../../third_party/nccl/src/device/prims_simple.h)

### 6.1 `waitPeer()`

`waitPeer()`完成两件事：

1. 等待连接进度允许当前 step 前进；
2. 根据 `step % NCCL_STEPS` 选择本次使用的 buffer 位置。

普通 SIMPLE 路径还会写入：

```c
connFifo[step % NCCL_STEPS].size = nelts * sizeof(T);
```

这让 Proxy 知道当前 slot 中有多少有效字节。

### 6.2 GPU复制或规约

GPU线程随后执行实际数据计算：

```c
// 只发送时的概念模型
send_slot[i] = local_input[i];

// 接收并规约时的概念模型
send_slot[i] = received_slot[i] + local_input[i];
```

实际实现使用模板化 `reduceCopy()` 等设备函数，并由多个GPU线程协作完成。

### 6.3 `postPeer()`

数据写好后，`postPeer()`先保证内存可见性，再更新连接进度：

```c
step += StepPerSlice;
st_relaxed_sys_global(connStepPtr, step);
```

在发送侧，相应角色的 `connStepPtr` 指向发送 `tail`；在接收消费侧，相应角色会更新接收 `head`。

## 7. 发送端完整过程

继续追踪 rank 0 的 `[1.0, 2.0]`。

### 7.1 初始状态

```text
发送 step = 0
发送 head = 0
发送 tail = 0
slot 0 = 空
connFifo[0].size = -1
```

### 7.2 Proxy提供可用credit

`sendProxyProgress()`首先推进 `sub->posted`。这里表示Proxy已经向GPU开放一定数量的buffer/credit，使GPU能够生产数据。

```text
posted：Proxy允许GPU使用到哪里
```

`posted` 不表示GPU已经写好数据，因此不能把它直接记录为 Mycroft 的 `GPU_ready`。

### 7.3 GPU选择slot并写数据

GPU检查 `head`，确认slot没有被尚未完成的旧请求占用。当前：

```text
slot = 0 % NCCL_STEPS = 0
```

然后写入：

```text
slot 0 = [1.0, 2.0]
connFifo[0].size = 8
```

### 7.4 GPU推进tail

数据和size对系统可见后：

```text
发送 tail：0 -> 1
```

此时可表达为：

```text
GPU 已经准备好 step 0 的数据
Proxy 尚未提交这个 step 的网络请求
```

### 7.5 Proxy判断数据ready

NCCL 2.21.5发送Proxy检查：

```c
connFifo[buffSlot].size != -1
&& *recvTail > expected_tail
```

SIMPLE普通buffer路径中，这两个条件共同说明：该slot具有有效大小，并且GPU已经发布相应进度。

### 7.6 Proxy调用`isend()`

数据ready后：

```c
ncclNet->isend(
    resources->netSendComm,
    buff,
    size,
    resources->tpRank,
    sub->mhandle,
    sub->requests + buffSlot
);
```

如果返回非空request：

```c
sub->transmitted += args->sliceSteps;
```

这里确认的是“网络发送请求已经成功提交并进入异步推进”，不是“远端GPU已经收到并使用数据”。

### 7.7 Proxy调用`test()`

对于已经提交的request：

```c
ncclNet->test(request, &done, &size);
```

当 `done` 为真：

```text
重置 connFifo[slot].size = -1
sub->done 前进
发送 head 前进
```

`head` 前进后，GPU以后可以循环复用该slot。

### 7.8 发送侧状态表

| 时刻 | slot数据 | tail | transmitted | done | head |
|---|---|---:|---:|---:|---:|
| 初始 | 空 | 0 | 0 | 0 | 0 |
| GPU写完并发布 | `[1,2]` | 1 | 0 | 0 | 0 |
| `isend()`返回request | `[1,2]` | 1 | 1 | 0 | 0 |
| `test()`确认完成 | 可释放 | 1 | 1 | 1 | 1 |

以上数字仅用于一个step的说明。真实字段通常按 `sliceSteps` 增长，需要统一单位后再比较。

## 8. 字节实际怎样移动

`tail`、`transmitted` 和 `done` 都是控制状态，数据本身沿另一条路径移动。

### 8.1 GDR路径

在GPUDirect RDMA可用时，可以概括为：

```text
rank 0 用户 GPU buffer
  -> GPU kernel复制/规约
rank 0 NCCL发送buffer
  -> 发送NIC通过PCIe DMA读取
网络
  -> 接收NIC通过PCIe DMA写入
rank 1 NCCL接收buffer
  -> rank 1 GPU读取、复制或规约
rank 1 recvbuff或下一跳发送buffer
```

CPU Proxy提交描述符并轮询完成，不逐元素复制payload。

### 8.2 非GDR路径

如果网络插件不能直接访问GPU内存，连接buffer可能经过Host可访问的映射或暂存区。准确位置由transport map、`useGdr`、注册方式和协议共同决定。因此“NIC一定直接读写用户GPU buffer”不是普通SIMPLE路径的通用结论。

## 9. 接收端完整过程

接收端与发送端使用另一组连接状态。

### 9.1 Proxy预先提交接收

`recvProxyProgress()`为当前step选择接收位置并调用：

```c
ncclNet->irecv(..., ptrs, sizes, ..., requestPtr);
```

request有效后，接收 `posted` 前进。

### 9.2 网络完成与必要的flush

Proxy使用：

```c
ncclNet->test(request, &done, sizes);
```

网络接收完成后，`received` 前进。GDR场景下还可能调用 `iflush()` 或执行等价可见性操作，保证NIC写入的数据能被GPU安全读取。

### 9.3 Proxy向GPU发布数据

接收数据及必要flush完成后：

```text
接收 transmitted 前进
接收 tail 前进
```

这里接收侧的 `transmitted` 表示接收路径已经推进到可交付GPU的阶段，它与发送侧 `transmitted` 的 `isend()` 提交语义不同。相同字段名不能脱离所在函数解释。

### 9.4 GPU消费接收slot

GPU primitive等待接收 `tail`，然后读取slot：

```text
收到的数据：[1.0, 2.0]
本地的数据：[10.0, 20.0]
SUM结果：   [11.0, 22.0]
```

消费完成后GPU推进接收 `head`，允许Proxy以后复用该接收slot。

### 9.5 Proxy确认GPU已经消费

接收Proxy观察GPU返回的 `head`，推进接收侧 `done`。因此接收侧完整阶段是：

```text
irecv已提交
  -> 网络接收完成
  -> flush完成
  -> tail通知GPU
  -> GPU读取/规约并推进head
  -> recv Proxy done
```

## 10. 两rank Ring AllReduce中的连接和slot

假设两个rank各有两个chunk：

```text
rank 0：[A0 | B0]
rank 1：[A1 | B1]
```

Ring顺序为：

```text
rank 0 -> rank 1 -> rank 0
```

第一轮发送原始chunk：

```text
rank 0发送B0，使用发送slot 0
rank 1发送A1，使用发送slot 0
```

第二轮接收并规约：

```text
rank 0：A0 + A1 = A，保存A并通过发送slot 1发给rank 1
rank 1：B1 + B0 = B，保存B并通过发送slot 1发给rank 0
```

最后接收：

```text
rank 0收到最终B
rank 1收到最终A
```

结果：

```text
rank 0 recvbuff = [A | B]
rank 1 recvbuff = [A | B]
```

这里：

```text
chunk：Ring算法当前处理的数据段
slice：为了流水传输从chunk继续切出的片段
step：连接协议的逻辑进度编号
slot：当前step使用的循环buffer位置
```

## 11. 多Channel意味着什么

一个AllReduce使用多个Channel时：

```text
完整tensor
  |-- Channel 0负责一部分数据
  |-- Channel 1负责另一部分数据
  \-- ...
```

每个Channel拥有自己的逻辑拓扑、peer connector和进度。GPU kernel中通常由不同CUDA block推进不同有效Channel。多个Channel可以共享同一GPU、Proxy线程、PCIe和NIC，因此逻辑进度独立不等于物理资源完全隔离。

一个Channel停滞时，其余Channel可能继续前进；整个collective通常要等待所有相关Channel完成。因此最后完成的Channel或rank不一定是最早出现异常的位置。

## 12. Proxy work如何绑定到Channel和connection

Host调度阶段为需要Proxy的Channel生成 `struct ncclProxyOp`。与本阶段最相关的字段包括：

```text
channelId
peer
nsteps
sliceSteps
chunkSteps
nbytes
protocol
pattern
opCount
```

`ncclProxySaveOp()`根据：

```text
channelId + Ring prev/next + send/recv方向 + connIndex
```

找到具体connector，并把工作绑定到该connector的Proxy connection。运行时，`ncclProxyOpToArgs()`把描述转换成 `ncclProxyArgs/ncclProxySubArgs`，Proxy progress线程反复调用transport提供的进度函数。

```text
ncclKernelPlan
  -> 每个Channel的Proxy op
  -> ncclProxySaveOp()
  -> recv connector / send connector
  -> recvProxyProgress() / sendProxyProgress()
```

一份Proxy op描述一段需要推进的工作，而不是为每个字节创建一个对象。`sub->posted/transmitted/done`等字段记录这段工作的累计进度。

## 13. Mycroft三个进度量的2.21.5源码候选

Mycroft论文使用 `GPU_ready`、`RDMA_transmitted` 和 `RDMA_done` 表达发送关键路径。NCCL 2.21.5没有一个现成的、名字就叫 `GPU_ready` 的累计成员，必须根据真实状态转换派生。

| Mycroft/E02量 | NCCL 2.21.5发送侧来源 | 当前结论 |
|---|---|---|
| `total_chunks` | `sub->nsteps / args->sliceSteps` | 候选；源码单位是协议step，不保证等同论文chunk |
| `GPU_ready` | SIMPLE路径中FIFO size有效且GPU `tail` 已超过当前期望step，readiness条件首次成立 | 候选；需要新增轻量累计计数 |
| `RDMA_transmitted` | `isend()`返回非空request后推进的 `sub->transmitted` | 字段推进语义已由源码确认；对外单位仍需归一化 |
| `RDMA_done` | `ncclNet->test()`返回完成后推进的 `sub->done` | 字段推进语义已由源码确认；不表示远端GPU已消费 |

发送侧抽象应满足：

```text
rdma_done <= rdma_transmitted <= gpu_ready <= total
```

比较前必须把四个量转换为同一单位。例如如果计数按step累加，就全部保留step；如果对外报告slice数量，就统一除以 `sliceSteps`。

### 13.1 为什么`posted`不是`GPU_ready`

发送Proxy先提供credit并推进 `posted`，随后GPU才可能写数据和推进tail：

```text
Proxy posted/credit
  -> GPU写数据
  -> GPU发布tail
  -> readiness成立
```

因此 `posted` 是GPU生产数据的前置许可，不是GPU已经完成生产的证据。

### 13.2 发送与接收的同名字段不能混用

发送侧：

```text
transmitted：isend request已提交
done：发送request完成
```

接收侧：

```text
posted：irecv已提交
received：网络接收完成
transmitted：完成flush并向GPU发布
done：GPU已经消费、接收工作可以结束
```

所以E02的三个累计量只对应发送侧简化状态机，不能直接套用到 `recvProxyProgress()`。

## 14. 正常和异常状态怎样解释

设统一单位下：

```text
total = 100
gpu_ready = 80
rdma_transmitted = 75
rdma_done = 72
```

表示：

```text
GPU准备了80个单位
其中75个已提交网络发送
其中72个已由发送侧网络request确认完成
```

典型停滞：

| 状态 | 当前卡点 | 应优先检查 |
|---|---|---|
| `gpu_ready`长期不增加 | 数据还未到发送ready阶段 | GPU kernel、上游Ring依赖、计算/通信资源竞争 |
| `gpu_ready > rdma_transmitted`且差距持续扩大 | ready数据没有及时变成发送request | Proxy调度、`isend()`、接收端是否准备、NET插件 |
| `rdma_transmitted > rdma_done`且差距持续扩大 | request已提交但迟迟未完成 | NIC、RDMA/Socket transport、网络或远端状态 |
| 三者都完成但collective未完成 | 问题可能在其他Channel、接收消费或后续同步 | 其他flow、recv路径、kernel/stream完成条件 |

这些状态只能缩小调查范围，不能只凭一个不等式直接断言某块硬件损坏。Ring依赖会把一个rank的慢速传播给后续rank，必须结合peer和Channel关系判断主动异常与受影响者。

## 15. operation和Channel身份

Mycroft需要把不同rank上的同一次collective关联起来。当前2.21.5源码映射为：

```text
communicator候选：comm->commHash
operation序号强候选：globalized proxyOp->opCount >> 1
rank：comm->rank
channel：proxyOp/sub->channelId
peer：sub->peer或连接元数据
```

`uploadProxyOps()`会把plan-local `opCount`平移到communicator历史序号，然后调用 `ncclProxySaveOp()`。插桩必须在全局化之后、恢复旧值之前取得候选序号。

建议组合：

```text
(capture_id, commHash, opCount >> 1, rank, channelId)
```

边界：

- `commHash`是64位hash，理论上可能碰撞；
- `ncclComm_t`指针不能作为跨进程communicator身份；
- `opCount >> 1`仍需两个rank连续多次AllReduce和CUDA Graph replay动态验证；
- Channel编号只在对应communicator内有意义；
- 一个NET connection可能拥有多个QP，单独一个QPN不是永久全局ID。

详细证据见：[`docs/architecture/event-field-sources.md`](../architecture/event-field-sources.md)。

## 16. Mycroft候选插桩区域

当前固定版本中的候选区域：

| 位置 | 目的 |
|---|---|
| communicator初始化完成路径 | 记录 `commHash/rank/GPU` 等低频元数据 |
| `uploadProxyOps()` | 取得全局化operation候选序号和Channel工作描述 |
| `sendProxyProgress()` readiness分支 | 派生发送侧 `GPU_ready` |
| `isend()`返回有效request之后 | 观察 `RDMA_transmitted` |
| 发送 `test()`返回完成之后 | 观察 `RDMA_done` |
| NET/IB connect/accept | 一次性记录NIC、连接和QP元数据 |

热路径插桩要求：

```text
固定大小记录
预分配内存
非阻塞写入
不在热循环做printf、文件I/O或动态格式化
周期性累计快照，而不是每个chunk都输出长日志
```

候选点详细说明见：[`instrumentation/nccl-2.21.5/tracepoints.md`](../../instrumentation/nccl-2.21.5/tracepoints.md)。

## 17. 当前已经确认与仍待验证的边界

### 17.1 源码可以直接确认

- Channel通过connector关联peer和方向；
- SIMPLE buffer按协议step循环复用；
- GPU primitive通过FIFO、head和tail与Proxy协作；
- 发送Proxy在readiness成立后调用 `isend()`；
- `isend()`返回request后 `sub->transmitted` 前进；
- `test()`确认发送完成后 `sub->done` 前进；
- 接收Proxy依次经历 `irecv`、receive completion、flush、tail发布和GPU消费确认；
- 发送与接收侧同名计数器具有不同阶段语义。

### 17.2 E06仍需动态验证

- `(commHash, opCount >> 1)`是否在目标工作负载中跨rank稳定对齐；
- 多Channel是否共享同一operation候选序号；
- CUDA Graph replay如何推进operation序号；
- 派生 `GPU_ready` 的计数方式和去重逻辑；
- `nsteps/sliceSteps`与项目对外 `total_chunks` 的单位对应；
- operation completion应在哪一层聚合所有Channel/sub；
- Socket与IB/RoCE路径的transport-specific元数据；
- 插桩对正确性、带宽、CPU和丢事件的实际影响。

## 18. 结构关系汇总

```text
struct ncclComm
  |
  \-- channels[channelId]
        |
        |-- Ring prev / next
        |
        \-- peers[peer]
              |
              |-- send connector
              |     |-- ncclConnInfo
              |     |     |-- buffs[protocol] -> slot[]
              |     |     |-- head / tail
              |     |     \-- connFifo
              |     \-- ncclProxyConnector -> sendProxyProgress()
              |
              \-- recv connector
                    |-- ncclConnInfo
                    \-- ncclProxyConnector -> recvProxyProgress()

GPU kernel / Channel block
  -> runRing()
  -> Primitives
  -> waitPeer()
  -> GPU copy/reduce
  -> postPeer()
  -> tail/head

Proxy progress thread
  -> sendProxyProgress()
       |-- readiness -> GPU_ready候选
       |-- isend() -> transmitted
       \-- test(done) -> done/head
  -> recvProxyProgress()
       |-- irecv()
       |-- test(received)
       |-- flush/tail
       \-- 等待GPU head后完成
```

## 19. 源码索引

| 主题 | NCCL 2.21.5源码 |
|---|---|
| `ncclConnInfo`、connector、device Channel | [`src/include/device.h`](../../third_party/nccl/src/include/device.h) |
| SIMPLE primitive、`waitPeer()`、`postPeer()` | [`src/device/prims_simple.h`](../../third_party/nccl/src/device/prims_simple.h) |
| Ring AllReduce primitive顺序 | [`src/device/all_reduce.h`](../../third_party/nccl/src/device/all_reduce.h) |
| Proxy类型和累计进度字段 | [`src/include/proxy.h`](../../third_party/nccl/src/include/proxy.h) |
| Proxy工作绑定与progress循环 | [`src/proxy.cc`](../../third_party/nccl/src/proxy.cc) |
| NET send/recv progress与`isend/irecv/test` | [`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc) |
| operation序号全局化、Proxy op上传 | [`src/enqueue.cc`](../../third_party/nccl/src/enqueue.cc) |
| IB/RoCE连接和QP元数据 | [`src/transport/net_ib.cc`](../../third_party/nccl/src/transport/net_ib.cc) |

## 20. 本阶段结论

本阶段建立的核心执行模型是：

```text
Ring算法决定处理哪个chunk
  -> GPU primitive把数据生产到连接buffer
  -> tail发布GPU ready状态
  -> Proxy调用isend提交网络请求
  -> test确认发送完成并推进head
  -> 远端Proxy通过接收tail把数据交给GPU
  -> 远端GPU复制、规约或继续转发
```

对Mycroft复现最重要的结论是：三个进度量应从同一条发送flow的真实状态转换中采集，而不是根据相似字段名猜测。

```text
GPU_ready：readiness条件首次成立时派生
RDMA_transmitted：isend返回有效request后累计
RDMA_done：发送test返回done后累计
```

这些源码位置已经明确，但身份跨rank一致性、计数单位、completion聚合和运行开销仍必须通过NCCL 2.21.5动态实验确认。

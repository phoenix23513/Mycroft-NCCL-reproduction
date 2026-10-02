# 阶段06：Mycroft 的操作身份、进度观测与阻塞依赖

> 保留的历史学习资料，不定义当前任务或验收。执行范围与进度以[核心复现计划](../plans/Mycroft_26日开发路线图.md)为准；源码解释按本文标注的版本阅读，候选观测点仍需真实运行验证。

本阶段承接[阶段05：完整 Ring AllReduce 与完成关系](阶段05_完整Ring_AllReduce与完成关系.md)。前面已经解释一次 collective 怎样执行；本文解释怎样把它记录成可分析的事件：辨认同一次操作、找出停止推进的阶段，并沿通信依赖寻找可能根因。

源码基线为 `third_party/nccl` 中的 NCCL `v2.21.5-1`。主要场景为 Ring、SIMPLE、普通 NET buffer 的发送侧。本笔记包含源码事实、独立采集设计和解释用的假设轨迹；采集候选仍需要真实运行验证。

## 1. 从一个实际问题开始

训练程序等待 AllReduce 完成，输出只告诉我们某个 rank 长时间没有结束。

沿阶段05的执行过程，等待可能发生在：

```text
GPU尚未执行到通信
  -> GPU等前驱数据或下游发送空间
  -> GPU已产生数据，但Proxy未提交请求
  -> 请求已提交，但网络完成未返回
  -> 对端收到数据，但GPU尚未消费
```

Mycroft 的目标是采集通信状态，并结合内部控制和数据依赖解释异常。论文入口：[Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training](https://arxiv.org/abs/2509.03018)。

完成这个任务至少要回答三个问题：

1. 当前事件属于哪次 AllReduce？
2. 哪条连接上的哪个阶段停止了推进？
3. 这个阶段在等待谁，是否有对端或上游证据？

因此本阶段先讲身份，再讲进度，最后讲依赖；字段随着它解决的问题引入。

## 2. 为什么首先需要操作身份

假设同一个 communicator 连续运行三次 AllReduce，每次都使用多个 Channel。

```text
communicator A
  |-- AllReduce 第1次
  |     |-- rank 0，Channel 0/1
  |     \-- rank 1，Channel 0/1
  |-- AllReduce 第2次
  |     |-- rank 0，Channel 0/1
  |     \-- rank 1，Channel 0/1
  \-- AllReduce 第3次
        |-- rank 0，Channel 0/1
        \-- rank 1，Channel 0/1
```

日志如果只记 `rank=0, channel=1, done=20`，就无法判断它属于哪次调用，也不能与另一 rank 的相应操作配对。

应逐级确定：

```text
采集任务
  -> communicator
  -> collective operation
  -> rank
  -> Channel
  -> 连接方向和peer
  -> 当前进度与采样时间
```

## 3. communicator身份：从初始化追踪`commHash`

每个参与 rank 的进程中有自己的 `struct ncclComm`，应用持有本地 `ncclComm_t` 句柄。同一组 rank 通过共同的初始化输入建立通信。

[`src/init.cc`](../../third_party/nccl/src/init.cc) 中的初始化路径执行：

```cpp
comm->commHash = getHash(job->commId.internal, NCCL_UNIQUE_ID_BYTES);
```

`commHash` 是 `ncclComm` 的64位整数成员，由 communicator 初始化标识计算。通常由同一 UniqueId 初始化的参与 rank 可得到相同 hash。

```text
共同初始化标识
  |-- rank 0 -> 本地ncclComm -> commHash H
  |-- rank 1 -> 本地ncclComm -> commHash H
  |-- rank 2 -> 本地ncclComm -> commHash H
  \-- rank 3 -> 本地ncclComm -> commHash H
```

采集侧可将本次采集标识与 hash 组合为 communicator 身份候选：

```text
communicator_id = capture_id + commHash
```

`capture_id` 是采集器分配的标识；NCCL 没有这个同名成员。`commHash` 的碰撞、communicator split/share 的生命周期和跨 rank 对应仍需在目标运行中验证。

本地 `ncclComm_t` 指针值可帮助进程内查表；跨进程关联应使用可以在各进程中一致识别的逻辑身份。

## 4. operation身份：序号在哪里生成

在 [`src/enqueue.cc`](../../third_party/nccl/src/enqueue.cc) 的 `addCBDCollToPlan()` / `addTunedCollToPlan()` 中：

```cpp
uint64_t opCount = uint64_t(plan->collOpCount++) << 1 | 0;
```

`plan` 是指向 `ncclKernelPlan` 的指针。`plan->collOpCount` 记录该 plan 内的 collective 数量，后缀 `++` 先取旧值再增加。

假设一个 plan 包含两次 AllReduce：

```text
第1次：旧collOpCount=0 -> opCount=0
第2次：旧collOpCount=1 -> opCount=2
```

左移一位为最低位留出类别标记：collective 使用0，P2P工作使用1。

这个 `opCount` 在遍历该 collective 的 Channel 之前生成，所以同一次 collective 分配到各 Channel 的 Proxy op 使用同一个 plan 内编号。

但下一个 plan 的本地计数也可能从0开始，需要进一步转换为历史序号。

## 5. `uploadProxyOps()`把plan内编号平移到历史中

同一文件的关键代码为：

```cpp
uint64_t collOpCount = comm->sharedRes->collOpCount;
comm->sharedRes->collOpCount += plan->collOpCount;
```

这里先记下历史起点，再为该 plan 中的 collective 保留序号范围。

对 collective 分支：

```cpp
q->opCount = (collOpCount << 1) + oldId;
NCCLCHECK(ncclProxySaveOp(comm, q, nullptr));
q->opCount = oldId;
```

例如历史起点为20，plan内两次操作的 `oldId` 分别为0、2：

| 操作 | plan内编号 | 上传时opCount | `opCount >> 1` |
|---|---:|---:|---:|
| 本plan第1次 | 0 | 40 | 20 |
| 本plan第2次 | 2 | 42 | 21 |

所以 collective 的 operation 序号候选为：

```text
op_seq_candidate = 已全局化的opCount >> 1
operation_key    = (communicator_id, op_seq_candidate)
```

这里“全局化”是相对于本地共享资源历史平移，不意味着代码已经完成跨主机一致性验证。

记录点应位于平移之后、恢复 `oldId` 之前，或记录已经传入 Proxy 运行时的相应值。尤其是 graph replay，plan可以重复上传，源码保留plan内编号以便下次再平移。

`comm->opCount` 是另一个字段，在 `ncclProxyStart()` 中按Proxy启动批次增加。一批可能包含多次collective，用它标记每次AllReduce会混淆操作。

## 6. 序号怎样进入Proxy进度函数

[`src/proxy.cc`](../../third_party/nccl/src/proxy.cc) 中 `ncclProxyOpToArgs()` 执行：

```cpp
sub->connection = op->connection;
sub->channelId = op->channelId;
sub->nsteps = op->nsteps;
sub->nbytes = op->nbytes;

args->opCount = op->opCount;
args->sliceSteps = op->sliceSteps;
args->protocol = op->protocol;
args->coll = op->coll;
```

身份沿着已有工作传递：

```text
ncclComm：commHash、rank
        |
        v
plan内生成collective opCount
        |
        v
同一次操作的多个Channel Proxy op
        |
        v
uploadProxyOps平移编号
        |
        v
ncclProxySaveOp / SaveProxy绑定连接
        |
        v
ncclProxyOpToArgs复制到args/sub
        |
        v
sendProxyProgress或recvProxyProgress读取运行时状态
```

进度函数签名只直接接收 `proxyState` 和 `args`。采集器需要在前面的调度/连接绑定处建立元数据关联，才能从连接工作恢复原 communicator 的身份和 rank。

## 7. 一条发送flow还需要哪些身份

以 communicator A 的第20次 AllReduce、rank 1、Channel 0向rank 2发送为例：

```text
communicator=A
operation=20
rank=1
channel=0
direction=send
peer=2
```

同一operation还有接收工作，需要区分发送和接收的不同计数含义。必要时还要保留 connector index 或采集器 connection ID。

操作和flow两个层次：

```text
operation_key = (communicator_id, op_seq)

flow_key = (operation_key, rank, channel,
            direction, peer, connection_id)
```

flow表示一个有明确方向和对端的观测对象；该tuple是采集设计，不是NCCL现成字段。

Ring的peer来源在 `ncclProxySaveOp()` 中：

```cpp
SaveProxy(..., proxyRecv, ring->prev, ...);
SaveProxy(..., proxySend, ring->next, ...);
```

`SaveProxy()` 根据方向和peer选择 connector。应该在这里或相应连接元数据中保存真正的前驱/后继。

`ncclProxyOpToArgs()` 的 `sub->peer = op->root` 并不保证是 Ring 的相邻 rank。根节点参数、transport rank和communicator rank应结合各自映射解释。

## 8. 在同一条发送flow上观察三个边界

回到阶段05：rank 1向rank 2发送c0中的数据。

```text
rank 1 GPU准备发送slice
       |
       v
发送buffer数据和FIFO size/tail就绪      <- GPU_ready候选
       |
       v
发送Proxy调用isend，得到有效request     <- transmitted
       |
       v
NET test返回完成                       <- done
```

[`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc) 的 `sendProxyProgress()` 是观察这些边界的主要位置。

三个进度应在同一flow、同一operation生命周期中累计。不同Channel、不同方向或不同操作的值无法直接代入同一个不变量。

## 9. `GPU_ready`从条件派生

普通SIMPLE发送路径的readiness条件涉及：

```cpp
connFifo[buffSlot].size != -1
*recvTail > sub->base + sub->transmitted
```

FIFO size说明本次slice的有效字节数已经发布，tail说明GPU已推进到该发送位置。

源码随后进入 `if (ready)` 并尝试 `isend()`。当前NCCL没有名为 `gpu_ready` 的现成累计成员，采集器要根据“哪个发送step第一次通过readiness检查”派生计数。

如果 `isend()` 暂时未返回有效request，下一轮可能再次检查同一个step。按检查次数累加会重复计数，需要按operation、connection和绝对step去重。

一种去重语义是：

```text
当前候选step = base + transmitted
readiness首次成立 -> 对应ready累计量前进一次
同一候选step再次成立 -> 累计量保持
transmitted前进后 -> 开始观察下一候选step
```

这只直接观察Proxy检查到的候选slice。GPU可能提前准备后面的slice；若要准确采集全部ready前沿，需要根据SIMPLE tail/FIFO规则进一步设计并验证。不能把“Proxy已观察到ready”的计数未经验证解释为任意时刻的全部GPU生产量。

`sub->posted` 表示Proxy已提供的buffer/credit进度，发生在数据实际ready之前，不能用它直接替代GPU_ready。

## 10. `transmitted`对应请求真正提交

源码先调用：

```cpp
NCCLCHECK(proxyState->ncclNet->isend(
    resources->netSendComm, buff, size,
    resources->tpRank, sub->mhandle,
    sub->requests + buffSlot));
```

只有请求指针非空才推进：

```cpp
if (sub->requests[buffSlot] != NULL) {
    sub->transmitted += args->sliceSteps;
}
```

因此它表示成功得到异步发送request的累计协议step。函数被调用过、函数成功返回、得到有效request，是需要分别看清的动作。

论文或项目模型名为 `RDMA_transmitted`；NET/SIMPLE的源码边界也可能运行在Socket上。采集时应记录实际transport，再解释网络类型。

## 11. `done`对应本地网络发送完成

源码检查已有request：

```cpp
NCCLCHECK(proxyState->ncclNet->test(
    sub->requests[buffSlot], &done, &size));
```

完成后，普通buffer路径先重置FIFO size，再推进消费与空间归还相关状态，其中：

```cpp
sub->done += args->sliceSteps;
```

该边界确认本地发送request完成。接收端GPU读取、规约和写入最终输出发生在另外一条执行链上，要使用接收侧状态或GPU完成证据判断。

## 12. 三个量先统一单位，再比较

原始计数器单位是协议step：

```text
ready_steps：派生得到的同单位进度
sent_steps ：sub->transmitted
done_steps ：sub->done
total_steps：sub->nsteps
```

在已验证适用的固定SIMPLE路径中，可按 `sliceSteps` 归一化为发送slice数量：

```text
ready_slices = ready_steps / sliceSteps
sent_slices  = transmitted / sliceSteps
done_slices  = done / sliceSteps
total_slices = nsteps / sliceSteps
```

这要求相应计数可按该单位整除，且生命周期、注册buffer分支和尾部处理已验证。

对应的不变量是：

```text
0 <= done_slices <= sent_slices <= ready_slices <= total_slices
```

阶段05举过一轮4-rank、`chunkSteps=4`、`sliceSteps=2`的单位例子：

```text
Ring chunk发送次数：6
协议step总数：      24
发送slice总数：     12
```

现有项目模型字段名是 `total_chunks`。如果适配器将它定义为归一化的发送slice数，必须明确记录该单位，不能再把12解释成Ring算法的chunk数量。

## 13. 用连续快照判断是否停止推进

一条假设的同rank、同发送flow轨迹，所有值已经统一为slice数量：

| 本地采样时间 | ready | transmitted | done | total |
|---|---:|---:|---:|---:|
| t0 | 40 | 35 | 30 | 100 |
| t1 | 60 | 55 | 50 | 100 |
| t2 | 80 | 75 | 70 | 100 |

三个边界持续推进，说明流水线在工作；不同值反映同时处于各阶段的数据数量。

另一条假设轨迹：

| 本地采样时间 | ready | transmitted | done | total |
|---|---:|---:|---:|---:|
| t0 | 60 | 50 | 45 | 100 |
| t1 | 60 | 50 | 50 | 100 |
| t2 | 60 | 50 | 50 | 100 |
| t3 | 60 | 50 | 50 | 100 |

可以确认：有已观察到ready的数据；已提交部分已经完成；后续请求在该时间窗口没有成功提交。

这将调查范围引向Proxy执行、NET提交条件和对端接收准备。要确认根因，还要检查对应远端、采样器是否持续运行、是否丢记录以及该等待是否超过正常窗口。

## 14. 三种进度差距怎样联系执行依赖

| 持续状态 | 已观察到的阶段边界 | 可能需要查看的依赖 |
|---|---|---|
| `ready > transmitted` | ready数据尚未成为有效发送request | Proxy调度、isend、接收端准备、NET插件 |
| `transmitted > done` | 已提交request尚未完成 | transport完成机制、网络、远端接收状态 |
| `ready == transmitted == done < total` | 当前观察前沿没有新增ready数据 | GPU是否运行、前驱数据、下游空间、之前stream工作 |

第三种状态既可能是正常等待，也可能是异常停滞；Proxy readiness观察方式还可能限制ready前沿的精度。时间窗口与其他状态证据决定解释。

所有计数为0的快照只说明尚无这些边界的推进。某些路径没有Proxy工作，或采集器没有覆盖相应路径，也会缺记录。

## 15. 将发送flow与对端接收flow配起来

阶段05的固定Ring中：

```text
rank 1，Channel 0，send到rank 2
                  |
                  v
rank 2，Channel 0，recv自rank 1
```

配对前需确认：相同communicator和operation、对应Channel与连接、互补方向和一致peer关系，协议与传输单位也匹配。

接收端的计数链为：

```text
posted -> received -> transmitted -> done
接收提交   网络完成    数据发布给GPU   普通buffer被GPU消费
```

例如本地发送request迟迟不完成，可以进一步查对端是否已经posted接收；对端收到但尚未发布，可查flush；发布后GPU消费不推进，可查GPU的前驱数据和下游空间等待。

NET插件决定send/recv完成的具体时间语义，两侧不同名义边界不能只按整数大小强行等同。

## 16. 一个阻塞如何沿Ring传播

继续使用 `0 -> 1 -> 2 -> 3 -> 0`，假设我们控制并记录了rank 2的发送Proxy暂停。这是解释用场景，不代表项目已经完成故障注入。

```text
rank 2 GPU已准备某个待发slice
  -> rank 2发送Proxy暂停，未继续提交
  -> rank 3接收不到该slice
  -> rank 3 recvReduceSend等待接收tail
  -> rank 3无法产生对应下一跳结果
  -> rank 0等待来自rank 3的结果
  -> 更下游的相关数据依赖继续受影响
```

该链涉及特定chunk/slice；其他已准备数据可能仍短暂推进。buffer窗口和流水线深度影响停滞传播速度。

观察上可能看到rank 3、rank 0等多个rank很慢。但最早的已知干预发生在rank 2发送Proxy，其余是依赖传播。

若没有已知干预证据，分析只能把rank 2的提交边界列为候选问题位置，再检查该Proxy是否被调度、NET提交是否返回有效request、对端接收是否已准备。

## 17. 下游空间如何反过来阻塞本地GPU

阶段04讲过 `recvReduceSend()` 同时等待接收tail和发送head。

```text
当前rank已收到输入slice
  -> 下一跳发送buffer没有可用空间
  -> WaitSend角色继续等待
  -> GPU暂时不能完成本轮规约转发
  -> 当前接收head不能继续前进
  -> 接收Proxy的done不前进
  -> 当前接收窗口逐渐填满
```

因此 `接收transmitted > 接收done` 说明GPU消费确认落后，但“GPU计算本身有问题”只是可能原因之一；下游连接的空间回收同样是依赖。

分析关系可画为：

```text
本rank的接收slot释放
       依赖
GPU完成接收/规约/转发
       依赖
前驱数据ready + 后继发送空间ready
                    |
                    v
             后继相关网络工作推进
```

## 18. 时间戳与持续时间怎么使用

真实采样可使用采集进程的单调时钟记录纳秒时间。它适合衡量同进程内的时间间隔：

```text
duration = 本地结束时间 - 本地开始时间
```

不同主机的单调时钟有不同起点，不能直接比较原始值并判断哪个rank先停。

跨rank分析先按身份配对，并在各自时间域计算持续时间、增量和操作顺序；严格跨主机先后关系还需要经过验证的时钟对齐或消息依赖。

`started_at_ns` 也必须绑定观测动作。如果在Host提交处取开始时间、在stream完成处取结束时间，得到的是包含排队等待的提交到完成时延；它和GPU kernel运行时长具有不同含义。

## 19. operation大小与传输大小各回答什么

阶段05中每rank输入8个float：

```text
operation逻辑大小 = 8 * 4 = 32字节
```

若按手工模型每次发送2个float、每rank发送6次chunk：

```text
每次发送有效载荷 = 8字节
每rank算法发送有效载荷总量 = 6 * 8 = 48字节
```

32字节描述这次AllReduce处理的向量大小；48字节描述该简化Ring的数据发送量。真实协议还涉及padding、空slice、协议标志等开销。

源码中：

- `ncclInfo.nBytes` 是operation逻辑规模来源，AllReduce从 `count * typeSize` 得到。
- `ncclProxyOp.nbytes` 是Proxy传输粒度参数，普通路径常来自 `stepSize * sliceSteps`。
- `isend()` 的 `size` 描述本次提交的字节数，应按实际路径解释。

计算吞吐时需说明使用逻辑字节数还是传输字节数，不能将它们混作同一个大小。

## 20. 与本项目Event结构怎样对应

[`src/mycroft/schema/event.py`](../../src/mycroft/schema/event.py) 已定义以下结构：

```text
Event
  |-- context：communicator_id / op_seq / collective / rank / channel
  |-- time：value / domain
  |-- event_kind
  |-- dependencies
  \-- payload
        |-- ProgressSnapshotPayload
        |     total_chunks / gpu_ready / rdma_transmitted / rdma_done
        \-- OperationCompletionPayload
              started_at_ns / message_bytes
```

真实采集的对应候选：

| Event字段 | 数据来源或解释 |
|---|---|
| `communicator_id` | 采集标识与commHash关联 |
| `op_seq` | collective上传后的 `opCount >> 1`，待跨rank验证 |
| `collective` | Proxy工作中的collective类型映射 |
| `rank` | 原communicator的rank，应通过元数据传递 |
| `channel` | `sub->channelId` |
| `gpu_ready` | 去重和单位归一化后的readiness派生量 |
| `rdma_transmitted` | 发送 `sub->transmitted` 的归一化值 |
| `rdma_done` | 发送 `sub->done` 的归一化值 |
| `total_chunks` | 已明确单位的总数候选，常从 `nsteps/sliceSteps` 得到 |

当前 `EventContext` 没有direction、peer、connection ID字段。对固定单连接Ring发送侧可以先限定适配范围；更完整的跨端依赖恢复需要元数据表或明确的契约扩展。本文flow_key不能直接冒充已有Event接口。

`dependencies` 是结构中的引用字段，但实际通信依赖仍需要采集与恢复。E01的确定性依赖模型和真实NCCL轨迹具有不同证据来源。

详细字段映射见[Event字段来源](../architecture/event-field-sources.md)。

## 21. completion log怎样获得真实含义

`OperationCompletionPayload` 已定义完成记录的分析接口，真实完成采集仍要回答阶段05中的层级问题：

```text
一个request完成
  -> 一个sub完成
  -> 一组Proxy工作完成

各Channel的GPU work完成
  -> 对应GPU操作完成
  -> 本rank应用可以使用结果
```

单个request、单个sub或单个Channel结束，覆盖不了整次operation。所有已知Proxy子工作结束也需与GPU work关系验证，特别是没有Proxy的路径、注册buffer和一kernel含多个collective的情况。

在原生workload中，为特定操作记录其后的CUDA event，可以提供本rank stream完成证据。若多个操作group在同一launch中，不能未经细分就把同一个结束event解释为每个内部操作的精确结束瞬间。

Event v2规定operation completion使用 `channel=null`，`time.value` 为观测完成时间，payload提供同一时钟域的开始时间和operation逻辑字节数。生产真实记录前必须确认观测覆盖的operation边界。

## 22. 插桩怎样进入现有执行链

插桩是在真实状态转换附近增加采集动作。例如在有效发送request产生后，读取该flow身份与进度，写入预分配事件区域。

```text
初始化/调度：建立身份和连接元数据
         |
         v
NET progress：观察状态转换、更新轻量计数
         |
         v
预分配有界内存记录
         |
         v
独立reader导出事件
         |
         v
分析器按operation和flow恢复状态、检查窗口、追踪依赖
```

Proxy热路径避免阻塞I/O和动态格式化日志，以免采集本身改变提交节奏。reader落后时记录丢事件或覆盖情况，缺失轨迹应反映在分析证据中。

源码位置候选见[候选插桩点](../../instrumentation/nccl-2.21.5/tracepoints.md)。本文描述采集设计，未新增插桩实现。

## 23. 哪些结论来自源码，哪些需要实验

| 内容 | 当前证据 |
|---|---|
| commHash生成位置 | 固定版本源码可确认 |
| plan内opCount及历史平移 | 固定版本源码可确认 |
| 同一次调度中的多Channel共用opCount | 相关Channel循环与赋值可确认 |
| 发送transmitted/done的转换边界 | NET发送源码可确认 |
| GPU_ready需要派生、posted表示credit | readiness与posted分支可确认 |
| 跨rank身份是否持续对齐 | 需要连续operation真实轨迹 |
| communicator split/share、graph replay身份 | 需要对应运行验证 |
| ready前沿与计数单位是否正确 | 需要SIMPLE路径实测及边界检查 |
| Proxy聚合点是否代表整次collective完成 | 需要与GPU/stream完成证据关联 |
| 对端flow是否准确匹配、根因是否可区分 | 需要多rank正常与受控异常证据 |
| 采集对时延、CPU和丢事件的影响 | 需要开启/关闭采集对比 |

论文提供方法，NCCL源码提供可观测状态，真实实验验证采集映射和分析效果。三者在笔记中应各自保留证据边界。

## 24. 源码定位与复习顺序

| 要追踪的关系 | 文件 | 搜索词 |
|---|---|---|
| communicator身份生成 | [`src/init.cc`](../../third_party/nccl/src/init.cc) | `comm->commHash =` |
| communicator和shared历史字段 | [`src/include/comm.h`](../../third_party/nccl/src/include/comm.h) | `commHash`、`ncclSharedResources`、`collOpCount` |
| plan内operation序号 | [`src/enqueue.cc`](../../third_party/nccl/src/enqueue.cc) | `plan->collOpCount++` |
| 序号平移与恢复 | [`src/enqueue.cc`](../../third_party/nccl/src/enqueue.cc) | `uploadProxyOps`、`q->opCount =` |
| operation字节数 | [`src/include/info.h`](../../third_party/nccl/src/include/info.h) | `ncclInfoSetDerived` |
| 连接方向与peer选择 | [`src/proxy.cc`](../../third_party/nccl/src/proxy.cc) | `ncclProxySaveOp`、`SaveProxy` |
| Proxy字段传递 | [`src/proxy.cc`](../../third_party/nccl/src/proxy.cc) | `ncclProxyOpToArgs` |
| Proxy启动批次计数 | [`src/proxy.cc`](../../third_party/nccl/src/proxy.cc) | `ncclProxyStart`、`comm->opCount++` |
| 运行时计数结构 | [`src/include/proxy.h`](../../third_party/nccl/src/include/proxy.h) | `ncclProxyArgs`、`ncclProxySubArgs` |
| 发送ready、提交、完成 | [`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc) | `sendProxyProgress`、`if (ready)`、`isend`、`test` |
| 接收状态和head确认 | [`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc) | `recvProxyProgress`、`recvTail`、`sendHead` |
| GPU连接等待依赖 | [`src/device/prims_simple.h`](../../third_party/nccl/src/device/prims_simple.h) | `waitPeer`、`loadRecvConn`、`loadSendConn` |
| 分析器Event接口 | [`src/mycroft/schema/event.py`](../../src/mycroft/schema/event.py) | `EventContext`、`ProgressSnapshotPayload` |
| 契约有效性条件 | [`src/mycroft/schema/validate.py`](../../src/mycroft/schema/validate.py) | `_validate_progress_snapshot`、`_validate_operation_completion` |

复习时从“某次AllReduce的一条发送连接没有推进”出发：先给它operation和flow身份，再找ready、提交、完成三个源码边界，最后沿前驱数据、后继空间和对端接收关系追查。能够解释这条链后，后续重点转入真实采集、正常运行校验与受控异常实验。

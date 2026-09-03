# NCCL 专有名词、英文简称与常见函数速查

本文用于解决 NCCL 学习中“名称很多，但不知道它是什么、位于哪里、和前后对象有什么关系”的问题。内容按照一次通信的执行顺序组织，可作为源码阅读时的长期速查表。

## 0. 使用范围与版本说明

- NCCL 目标版本：2.21.5，服务于 Mycroft 复现。
- 当前本地源码：NCCL master，commit `5067397c2676d5aed50042fc39e5c8ee96eb0027`。
- 公共 API 和集合通信语义优先参考 NCCL 2.21.5 官方文档。
- 内部结构体和内部函数可能随版本变化。本文标注的本地源码位置不能直接视为 NCCL 2.21.5 的插桩结论。
- `GPU_ready`、`RDMA_transmitted`、`RDMA_done` 等 Mycroft 指标的准确源码来源仍需在 NCCL 2.21.5 中逐项确认。

## 1. 先记住整条主线

```text
初始化

外部框架确定 rank / nranks / CUDA device
  -> ncclGetUniqueId()
  -> 外部分发 ncclUniqueId
  -> ncclCommInitRank()
  -> bootstrap
  -> peerInfo
  -> topology
  -> Channel / Ring / transport connection
  -> communicator ready


一次 AllReduce

ncclAllReduce()
  -> ncclInfo
  -> ncclTaskColl
  -> planner
  -> ncclDevWorkColl
  -> ncclKernelPlan
  -> ncclDevKernelArgs
  -> ncclLaunchKernel()
  -> ncclKernelMain()
  -> RunWorkBatch.run()
  -> RunWorkColl.run()
  -> runRing()
  -> GPU Primitives


跨机网络推进

GPU 准备数据
  -> connection 控制状态
  -> ncclProxyOp
  -> Proxy CPU 线程
  -> NET transport / plugin
  -> NIC / RDMA
  -> 网络完成
  -> Proxy 更新进度
  -> GPU 继续执行
```

## 2. 本文使用的“类型”标签

看到新名字时，首先判断它属于哪一类。

| 标签 | 含义 | 例子 |
|---|---|---|
| 公共 API 函数 | 应用或训练框架可以调用的 NCCL 函数 | `ncclAllReduce()` |
| 内部 Host 函数 | 在 CPU 上执行的 NCCL 内部函数 | `ncclLaunchPrepare()` |
| Device 函数 | 在 GPU 上执行的函数 | `runRing()` |
| 结构体类型 | 描述某类对象的内存布局 | `struct ncclTaskColl` |
| 结构体指针 | 保存某个结构体对象的位置 | `struct ncclTaskColl *task` |
| 成员 | 某个结构体内部的字段 | `comm->rank` |
| 句柄 | 对外隐藏内部实现的本地引用值 | `ncclComm_t` |
| 队列/容器 | 保存若干对象指针并规定组织顺序 | `collTaskQueue` |
| CPU 线程 | Host CPU 上的一条执行流 | Proxy thread |
| CUDA stream | GPU 工作的异步顺序队列 | `cudaStream_t stream` |
| 算法 | rank 之间按什么拓扑交换数据 | Ring、Tree |
| 协议 | 每条连接怎样分片、同步和搬运数据 | Simple、LL、LL128 |
| transport | 两个 peer 通过什么机制建立数据路径 | P2P、SHM、NET |

## 3. 高频英文简称

### 3.1 系统、软件和执行设备

| 简称 | 英文 | 中文与作用 |
|---|---|---|
| NCCL | NVIDIA Collective Communications Library | NVIDIA 集合通信库，面向多 GPU collective 和 P2P 通信 |
| CUDA | 常解释为 Compute Unified Device Architecture | NVIDIA 的 GPU 编程与运行时平台 |
| DDP | Distributed Data Parallel | PyTorch 分布式数据并行；通常在反向传播期间触发梯度 collective |
| MPI | Message Passing Interface | 进程间消息传递标准；可用于启动进程和分发 NCCL UniqueId，但不是 NCCL 的内部组成部分 |
| CPU | Central Processing Unit | Host 处理器，执行训练框架 Host 代码、NCCL Host 调度和 Proxy 线程 |
| GPU | Graphics Processing Unit | 执行训练计算和 NCCL device kernel |
| NIC | Network Interface Card | 网卡，负责节点之间的实际网络传输 |
| NVML | NVIDIA Management Library | 查询和管理 GPU 设备信息的 NVIDIA 库 |
| PCIe | Peripheral Component Interconnect Express | CPU、GPU、NIC 等设备之间的总线互连 |
| NVLink | NVIDIA 高速互连技术名称 | GPU 之间或 GPU 与交换设备之间的高速互连 |
| SM | Streaming Multiprocessor | GPU 内执行 CUDA thread block 的硬件计算单元 |
| ABI | Application Binary Interface | 二进制调用与符号规则；`extern "C"` 与稳定 ABI 有关 |
| API | Application Programming Interface | 应用可调用的接口，例如 `ncclAllReduce()` |

### 3.2 进程、内存和通信

| 简称 | 英文 | 中文与作用 |
|---|---|---|
| PID | Process Identifier | 操作系统进程编号 |
| UUID | Universally Unique Identifier | 通用唯一标识；GPU UUID 用于识别物理或逻辑 GPU |
| IPC | Inter-Process Communication | 进程间通信；CUDA IPC 可让不同进程访问共享的 GPU 资源 |
| SHM | Shared Memory | 共享内存 transport，常用于同一主机不同进程之间的通信 |
| P2P | Peer-to-Peer | 点对点；在不同上下文中可能指 NCCL send/recv，也可能指 GPU P2P transport |
| DMA | Direct Memory Access | 设备直接读写内存，避免 CPU 逐字节搬运 |
| RDMA | Remote Direct Memory Access | 远程直接内存访问，允许网络设备直接访问已注册内存 |
| GDR | GPUDirect RDMA | NIC 直接访问 GPU memory 的数据路径能力 |
| IB | InfiniBand | 常用于高性能计算集群的网络技术 |
| RoCE | RDMA over Converged Ethernet | 在以太网上实现 RDMA 的技术 |
| RMA | Remote Memory Access | 远程内存访问操作类型；是比具体 Ring collective 更广的访问语义 |
| FIFO | First In, First Out | 先进先出队列或环形缓冲区；NCCL 中也用于 work 和连接 step 的组织 |

### 3.3 RDMA 和网卡队列

| 简称 | 英文 | 中文与作用 |
|---|---|---|
| QP | Queue Pair | RDMA 队列对，通常由发送队列和接收队列组成，是通信连接的重要标识 |
| SQ | Send Queue | RDMA 发送工作队列 |
| RQ | Receive Queue | RDMA 接收工作队列 |
| CQ | Completion Queue | 网络操作完成队列 |
| CQE | Completion Queue Entry | CQ 中的一条完成记录 |
| WQE | Work Queue Element | 提交到发送或接收队列的一项网络工作 |
| MR | Memory Region | 注册给 RDMA/NIC 使用的一段内存区域 |

这些词描述的是网卡和 RDMA 层，不应与 NCCL 的 `task`、`work`、`plan` 混为一谈。

### 3.4 NCCL 算法、协议和调试

| 简称 | 英文 | 中文与作用 |
|---|---|---|
| LL | Low Latency | NCCL 低延迟协议，适用于特定消息规模和平台条件 |
| LL128 | Low Latency 128 | NCCL 的另一种低延迟协议，不等同于 Simple |
| NET | Network transport | NCCL 跨节点网络 transport 的常用名称 |
| CollNet | Collective Network | 利用网络侧 collective 能力的 NCCL 算法/transport 体系 |
| NVLS | NVLink SHARP | 利用 NVLink 网络内规约能力的 NCCL 路径 |
| CE | Copy Engine | GPU 复制引擎；部分新路径会出现 CE collective 名称 |
| RAS | Reliability, Availability, Serviceability | 可靠性、可用性和可维护性功能 |
| NVTX | NVIDIA Tools Extension | 用于在性能分析工具中标记范围和事件 |

### 3.5 Mycroft 分析术语

| 简称 | 英文 | 中文与作用 |
|---|---|---|
| RCA | Root Cause Analysis | 根因分析 |
| MVP | Minimum Viable Product | 最小可运行版本 |
| trace | Execution Trace | 运行轨迹，由多条事件组成 |
| tracepoint | Trace Point | 插入源码的观测点 |
| timestamp | Time Stamp | 事件时间戳 |
| straggler | Straggler | 明显落后但不一定完全失败的参与者 |
| failure | Failure | 故障或停止推进状态 |
| trigger | Trigger | 触发分析的规则，例如一段时间没有 collective 完成 |

## 4. 应用和执行环境中的常见名词

### Host

- 英文：Host。
- 类型：执行环境概念。
- 含义：运行 CPU 程序的一侧。
- 在 NCCL 中执行：API 参数检查、task/plan 创建、Proxy 和网络插件调用。
- `Host` 不等同于“用户主线程”；Proxy 线程同样运行在 Host 上。

### Device

- 英文：Device。
- 类型：执行环境概念。
- 在当前上下文通常指 CUDA GPU。
- `device code` 指 GPU 执行的代码；`device memory` 指 GPU 可访问的内存。

### Process

- 英文：Process，进程。
- 类型：操作系统执行单位。
- 有独立虚拟地址空间。
- 多进程训练中通常每个进程管理一张 GPU，但这不是 NCCL API 强制规定的唯一模式。

### Thread

- 英文：Thread，线程。
- 类型：进程内部执行流。
- 同一进程中的线程共享地址空间。
- 用户 Host 线程和 NCCL Proxy 线程都是 CPU 线程，但职责不同。

### Kernel

- 英文：Kernel。
- 类型：GPU 函数及其一次 GPU 执行实例。
- NCCL Host 代码通过 CUDA stream 提交 kernel，GPU 随后异步执行。

### CUDA stream

- 英文：CUDA Stream。
- 类型：CUDA 运行时对象/句柄。
- 作用：规定同一 stream 上 GPU 工作的执行顺序。
- `ncclAllReduce()` Host 返回不等于 stream 上的 NCCL kernel 已完成。

### CUDA event

- 英文：CUDA Event。
- 类型：CUDA 同步和时间观测对象。
- 可用于建立 stream 依赖或判断 GPU 工作是否到达某个位置。

## 5. Collective 操作名

### Collective

- 英文：Collective Communication，集合通信。
- 类型：通信操作类别。
- 含义：一个 communicator 内的多个 rank 共同参与的通信操作。
- 参与 rank 必须按匹配顺序调用匹配的 collective，否则可能等待、报错或产生未定义结果。

### AllReduce

- 英文：All-Reduce。
- 公共 API：`ncclAllReduce()`。
- 作用：对各 rank 的同一位置执行 sum/max 等规约，并把完整结果写到每个 rank。

```text
rank 0: [1, 2]
rank 1: [10, 20]

sum AllReduce 后：
rank 0: [11, 22]
rank 1: [11, 22]
```

### ReduceScatter

- 英文：Reduce-Scatter。
- 公共 API：`ncclReduceScatter()`。
- 作用：先规约，再把完整规约结果分块，每个 rank 得到其中一块。
- Ring AllReduce 的前半阶段具有 ReduceScatter 的逻辑效果。

### AllGather

- 英文：All-Gather。
- 公共 API：`ncclAllGather()`。
- 作用：每个 rank 提供一块数据，最终每个 rank 收集到所有块。
- Ring AllReduce 的后半阶段具有 AllGather 的逻辑效果。

```text
AllReduce 在逻辑上等价于：
ReduceScatter + AllGather
```

这不表示 `runRing()` 内部真的再次调用两个公共 API；GPU kernel 会把两段步骤连续甚至融合执行。

### Reduce

- 英文：Reduce。
- 公共 API：`ncclReduce()`。
- 作用：规约结果只写到指定 root rank。

### Broadcast / Bcast

- 英文：Broadcast；`Bcast` 是常见缩写。
- 公共 API：`ncclBroadcast()`、`ncclBcast()`。
- 作用：把 root rank 的数据复制到所有 rank。

### Send / Recv

- 英文：Send / Receive。
- 公共 API：`ncclSend()`、`ncclRecv()`。
- 类型：点对点操作，不是所有 rank 必须共同参与的 collective。

### P2P 的两种常见含义

```text
P2P operation
    ncclSend/ncclRecv 点对点操作

P2P transport
    同机 GPU 之间的直接访问或 CUDA IPC 数据路径
```

看见 `P2P` 时必须根据文件和上下文判断含义。

### Group call

- 公共 API：`ncclGroupStart()` / `ncclGroupEnd()`。
- 作用：把多个 NCCL 调用组合起来，支持多 GPU 提交、操作聚合等场景。
- group 内部某个 API 返回时，工作可能尚未真正提交到 CUDA stream；最外层 `ncclGroupEnd()` 才结束整组提交。

## 6. Communicator 初始化术语

### Communicator / comm

- 英文：Communicator，通信器。
- `comm`：源码中的常用变量缩写。
- 内部类型：`struct ncclComm`。
- API 类型：`ncclComm_t`，本质上是 `struct ncclComm *` 的别名。
- 作用：保存一个本地 rank 的长期通信状态，包括 rank 身份、Channel、连接、拓扑和 planner。

```c
typedef struct ncclComm *ncclComm_t;
```

### Handle / 句柄

- 类型：API 设计概念。
- 含义：调用者持有、但不需要了解内部结构的本地引用值。
- `ncclComm_t` 是 communicator 句柄。
- 不同进程中的句柄指针值不同，不能用作跨 rank communicator ID。

### Rank

- 类型：communicator 内的整数编号。
- 范围：`[0, nranks-1]`。
- 唯一性范围：只在所属 communicator 内不能重复。
- 每个 communicator 都可以拥有自己的 rank 0、rank 1 等。
- rank 不是机器编号，也不是固定的物理 GPU 编号。

### nranks / nRanks

- 英文：Number of Ranks。
- 类型：整数。
- `nranks` 常见于 API 参数；`nRanks` 常见于内部成员。
- 含义：当前 communicator 的 rank 总数。

### localRank

- 英文：Local Rank。
- 类型：整数编号。
- 含义：当前 rank 在本节点内部的编号。
- 它与 communicator 全局 rank 不一定相同。

### Node / Host / Machine

- Node：集群节点，通常表示一台服务器。
- Host：可表示运行 CPU 程序的一侧，也可泛指主机；需看上下文。
- Machine：机器，非特定 API 类型。

### CUDA device / cudaDev

- `CUDA device`：CUDA 运行时看到的 GPU 设备。
- `cudaDev`：NCCL 内部常用的设备编号字段。
- rank 与 CUDA device 的映射由调用者初始化 communicator 时建立。

### `ncclUniqueId`

- 类型：初始化凭据数据结构。
- 生成函数：`ncclGetUniqueId()`。
- 作用：让所有参与者进入同一次初始化 rendezvous。
- 分发者：MPI、TCP store、训练框架等 NCCL 外部机制。
- 它不是本地 `ncclComm` 指针，也不应直接等同于一个永久、全局可用的 `comm_id`。

### Rendezvous

- 英文：Rendezvous，会合。
- 类型：分布式初始化过程概念。
- 含义：多个参与者使用共同凭据找到彼此并建立初始化联系。

### Bootstrap

- 英文：Bootstrap，引导启动。
- 类型：NCCL 初始化控制通信模块。
- 常见函数：`bootstrapInit()`、`bootstrapAllGather()`。
- 作用：正式 Channel/transport 尚未建立时，交换 peerInfo、连接信息等初始化数据。
- bootstrap 不是后续大规模梯度数据传输的主要路径。

### `peerInfo`

- 完整类型：`struct ncclPeerInfo`。
- 类型：结构体及其数组。
- 创建：每个 rank 先通过 `fillInfo()` 填写本地信息。
- 交换：通过 `bootstrapAllGather()` 收集所有 rank 的信息。
- 内容：rank、CUDA/NVML 设备、host/process 标识、PCI bus、GPU UUID、GDR 支持等。

### Peer

- 英文：Peer，对端或通信伙伴。
- 不是固定结构体名称。
- 在某条连接中，相对于当前 rank 的另一个 rank 即 peer。

### Topology / topo

- 英文：Topology，拓扑。
- `topo`：常见变量缩写。
- 作用：描述 GPU、CPU、PCIe、NVLink、NIC 及路径关系，为算法、Channel 和 transport 选择提供依据。

### Connection / Connector

- Connection：两端已经或正在建立的通信状态。
- Connector：NCCL 中封装某一方向连接状态的对象名称。
- 同一个 Channel 通常分别维护 send 和 recv 方向的连接。

### `ncclCommInitRank()`

- 类型：公共 API 函数，Host 执行。
- 输入：输出句柄位置、nranks、UniqueId、当前 rank。
- 隐含前提：调用者已选择当前 CUDA device。
- 输出：当前进程中的本地 communicator 指针。

## 7. Host 任务提交术语

### Enqueue

- 英文：Enqueue，入队。
- 含义：把任务或工作指针连接到某个队列，供后续阶段处理。
- 它不自动表示 GPU 已开始执行，也不自动表示网络已经发送。

### `ncclInfo`

- 完整类型：`struct ncclInfo`。
- 类型：Host 局部结构体。
- 作用：临时汇总一次 API 调用的参数，例如 buffer、count、datatype、op、comm、stream。
- 生命周期：只存在于当前 Host 调用链，不是长期任务身份。

### `ncclEnqueueCheck()`

- 类型：内部 Host 函数。
- 作用：检查 communicator 和参数，并把 API 请求送入 group/task 提交流程。

### `taskAppend()`

- 类型：内部 Host 函数。
- 作用：判断请求属于 collective、P2P、RMA 或特殊路径，并调用对应 task 创建逻辑。

### `collTaskAppend()`

- `coll`：collective 的常用缩写。
- 类型：内部 Host 函数。
- 作用：为普通多 rank collective 创建并连接 `ncclTaskColl`。

### Task

- 英文：Task，任务。
- 在当前主线主要指 `struct ncclTaskColl` 对象。
- 作用：保存一次用户 collective 请求及 Host 调度所需信息。
- task 不是线程，也不是 GPU kernel。

### `ncclTaskColl`

- 完整类型：`struct ncclTaskColl`。
- `Coll`：collective。
- 类型：Host 结构体。
- 内容：func、sendbuff/recvbuff、count、datatype、规约、algorithm、protocol、Channel 限制等。
- 创建后会进入当前 `comm->planner` 的容器。

### Planner

- 完整类型：`struct ncclKernelPlanner`。
- 类型：`struct ncclComm` 的直接成员状态。
- 它不是函数、线程或单个队列。
- 作用：收集 task、stream 和 work，并组装 kernel plan。

### Sorter

- 完整类型：`struct ncclTaskCollSorter`。
- 类型：Host 容器/排序状态。
- 作用：粗略按任务规模组织 collective task，便于后续调度。

### Queue

- 英文：Queue，队列。
- NCCL 中许多队列是侵入式队列：对象自身含 `next` 指针，队列保存和移动对象指针。
- 对象进入另一个队列通常不是复制整个对象，而是改变指针连接关系。

### `ncclPrepareTasks()`

- 类型：内部 Host 函数。
- 作用：整理已提交任务，为 collective 选择 algorithm、protocol、Channel 数和 device function 等执行参数。

## 8. 从 Host task 到 GPU work

### Work

- 英文：Work，工作描述。
- 与 task 的区别：task 偏向用户请求和 Host 生命周期；work 偏向 GPU 实际执行所需的紧凑信息。

### `ncclTasksRegAndEnqueue()`

- `Reg`：Register/Registration，注册。
- 类型：内部 Host 函数。
- 作用：处理 buffer 注册，并为 task 创建对应 `ncclDevWorkColl`，连接到 `collWorkQueue`。
- 当前本地 master 中，devWork 在 kernel plan 之前创建。

### `ncclDevWorkColl`

- 名称拆分：NCCL + Device + Work + Collective。
- 完整类型：`struct ncclDevWorkColl`。
- 类型：GPU 工作描述结构体。
- 创建位置：Host 先构造；随后放到 GPU kernel 可读取的 work 区域。
- 内容：buffer 指针、Channel 范围、数据分块、规约参数和 warp 数等。

### `ncclWorkList` / `workNode`

- `struct ncclWorkList`：Host 链表节点结构体。
- `workNode`：指向某个节点的局部指针变量。
- 当前实现中，实际 `ncclDevWorkColl` 数据可紧跟在节点之后。

```text
workNode
   |
   v
+---------------------+------------------------+
| struct ncclWorkList | struct ncclDevWorkColl |
+---------------------+------------------------+
```

### Register / Registration

- 英文：Register / Registration，注册。
- 在通信系统中通常表示把用户 buffer 注册为某种 transport、IPC、RDMA 或图执行可使用的资源。
- 注册不等同于复制完整 buffer。

### `ncclKernelPlan`

- 完整类型：`struct ncclKernelPlan`。
- 类型：Host 结构体。
- 作用：描述一次 kernel launch 工作包。
- 包含：task 队列、work 队列、Proxy op 队列、Channel mask、kernel function、kernel args、block 线程数等。
- 一个 plan 可以组织多个 task；一个 task 可以使用多个 Channel。

### Plan

- 英文：Plan，执行计划。
- 它不是用户的一次 collective 本身，而是 Host 为一次 kernel launch 组装的执行单元。

### `ncclLaunchPrepare()`

- 类型：内部 Host 函数。
- 作用：分配 `ncclKernelPlan`，把 planner 中待处理任务调度进一个或多个 plan，并准备 launch 顺序。

### `scheduleCollTasksToPlan()`

- `schedule`：调度。
- 类型：内部 Host 函数。
- 作用：把 collective task 与对应 devWork 编入 plan，填写 Channel 范围、chunk 参数、work batch、kernel function 和 Proxy op。

### Schedule

- 英文：Schedule，调度。
- 含义：决定任务在哪些 Channel、哪些 batch 和哪次 kernel launch 中执行。
- 不应简单翻译成“立刻执行”。

### `ncclDevWorkBatch`

- 名称拆分：Device Work Batch。
- 类型：GPU 可读取的 batch 描述结构体。
- 作用：告诉某个 Channel 从 work 存储区域的哪个位置读取哪些 work。

### Batch

- 英文：Batch，批次。
- 在此指把多个 work 组织成 GPU 可依次处理的一组，不是深度学习中的训练 batch。

### `finishPlan()`

- 类型：内部 Host 函数。
- 作用：完成 plan，创建 kernel 参数，排列各 Channel 的 work batch，并整理 Proxy op。

### `ncclDevKernelArgs`

- 名称拆分：Device Kernel Arguments。
- 类型：GPU kernel 参数结构体。
- 作用：向 GPU 提供 device communicator、Channel mask、work 存储位置等入口信息。

### `uploadWork()`

- 类型：内部 Host 函数。
- 作用：根据 Args/FIFO/Persistent 存储类型，把 work 放入 GPU kernel 可读取的区域。
- `upload` 不一定意味着每次都执行普通 Host-to-Device memcpy。

### `ncclLaunchKernel()`

- 类型：内部 Host 函数。
- 作用：根据 plan 设置 grid、block、共享内存和 kernel 参数，并向 CUDA stream 提交 GPU kernel。
- 返回时只能证明 Host launch 调用完成，不能证明 collective 已完成。

## 9. CUDA 与 GPU 执行术语

### Grid

- 英文：Grid。
- 类型：一次 CUDA kernel launch 的全部 thread block 集合。

### Block / Thread block

- 英文：Thread Block。
- 类型：一组可以共享 shared memory 并进行 block 内同步的 CUDA thread。
- 当前普通 NCCL collective launch 中，一个有效 Channel 对应 grid 中一个 block。
- Channel 是 NCCL 逻辑对象，block 是 CUDA 执行单位，二者不能永久画等号。

### Thread

- CUDA thread：GPU 上最小的编程执行线程。
- CPU thread：Host 进程中的执行流。
- 二者只共享“线程”这个词，执行环境完全不同。

### Warp

- 英文：Warp。
- 类型：NVIDIA GPU 的一组线程调度单位，通常包含 32 个 CUDA thread。
- `nWarps` 表示某个 work 使用多少个 warp。

### Shared memory / shmem

- 英文：Shared Memory；`shmem` 是常见缩写。
- CUDA block 内线程共享的低延迟存储。
- 注意：CUDA shared memory 与 Host 进程间 `SHM transport` 不是同一种东西。

### `channelMask`

- 类型：位集合成员。
- 每一位表示某个 Channel 是否参与当前 plan。
- GPU kernel 根据 `blockIdx.x` 和 `channelMask` 得到实际 `channelId`。

### `ncclKernelMain()`

- 类型：Device 模板函数，GPU 执行。
- 作用：加载 communicator、Channel、work batch，并调用具体 collective 实现。

### `RunWorkBatch`

- 类型：C++ 模板结构体，不是单独的普通函数。
- 调用形式：构造临时对象后调用 `.run()`。
- 作用：遍历一个 batch 中的 work。

### `RunWorkColl`

- 类型：C++ 模板结构体。
- 模板参数通常编码 collective、datatype、reduction、algorithm 和 protocol。
- 针对 `AllReduce + Ring + Simple` 的模板特化最终调用相应 `runRing()`。

### `runRing()`

- 类型：Device 函数，GPU 执行。
- 作用：按照 Ring 算法为当前 Channel 的数据执行 ReduceScatter 和 AllGather 步骤。
- 同名函数可出现在 AllReduce、Broadcast、ReduceScatter 等不同 device 文件中，必须结合文件判断。

## 10. Algorithm、Protocol、Transport 三者的区别

| 名称 | 回答的问题 | 例子 |
|---|---|---|
| Algorithm | 多个 rank 按什么逻辑拓扑和顺序交换数据？ | Ring、Tree |
| Protocol | 一条连接上的数据怎样分片、同步和搬运？ | Simple、LL、LL128 |
| Transport | 两个 peer 通过什么底层机制连接？ | P2P、SHM、NET |

同一次 AllReduce 可能是：

```text
Algorithm = Ring
Protocol  = Simple
Transport = rank 对之间分别选择 P2P、SHM 或 NET
```

不能说“Ring 和 RDMA 是两个同级算法”。Ring 是 collective 算法，RDMA 是网络数据传输机制。

## 11. Ring、Channel 和数据分块

### Ring

- 英文：Ring，环。
- 类型：collective 算法拓扑。
- 每个 rank 通常有逻辑 `prev` 和 `next`。
- chunk 按 Ring 顺序流动并形成跨 rank 依赖。

### Tree

- 英文：Tree，树。
- 类型：collective 算法拓扑。
- 节点按父子关系执行上行规约和下行分发等步骤。
- 当前 Mycroft 学习主线先以 Ring 为主。

### Channel

- 类型：NCCL 逻辑执行通道。
- 内部类型：Host 侧可见 `struct ncclChannel`，GPU 侧有对应 device Channel 状态。
- 作用：让同一次 collective 的不同数据范围并行推进，并保存 Ring/Tree 拓扑与连接。
- Channel 不是 rank、GPU、物理网线或 Proxy 线程。

### `channelId`

- 类型：整数编号。
- 作用：在当前 communicator/plan 上下文中区分 Channel。
- 单独一个 `channelId` 不能跨 communicator 唯一标识一条通信流。

### Chunk

- 英文：Chunk，数据块。
- 类型：逻辑数据单位。
- 是某个 Channel 在某轮 Ring 中推进的一段连续数据，不固定为一个标量。

### Slice

- 英文：Slice，切片。
- 类型：比 chunk 更细的流水处理单位。
- primitive 可按 slice 推进 ready、发送、接收和完成，以形成流水线。

### Step

- 英文：Step，步骤。
- 类型：连接 FIFO/协议推进的逻辑序号。
- 在 Ring 讲解中也常指 chunk 前进一跳；具体代码中必须确认它是算法 step 还是连接缓冲 step。

### Loop

- 英文：Loop，循环。
- 当一个 Channel 的数据多于一轮 `nranks * chunkCount` 时，`runRing()` 外层循环继续处理后续数据。

### `ncclCollCbdPart()`

- `Coll`：Collective。
- `Cbd`：Continuous Byte Distribution，连续字节分配。
- 类型：Host/Device 都可使用的内联模板函数。
- 作用：根据 Channel 和 devWork 分块字段，计算当前 Channel 的 offset、count 和 chunkCount。

## 12. GPU primitive 操作名

### Primitive

- 英文：Primitive，原语。
- 类型：GPU 通信操作封装。
- 组合接收、规约、复制、发送和连接同步等底层动作。

### `directSend()`

- 类型：`Primitives` 对象的 Device 成员函数。
- 作用：把本地数据发送给 Ring next，作为 ReduceScatter 起始动作之一。

### `directRecvReduceDirectSend()`

- 名称拆分：Receive + Reduce + Send。
- 作用：从 prev 接收，与本地对应数据规约，再向 next 发送。

### `directRecvReduceCopyDirectSend()`

- 名称拆分：Receive + Reduce + Copy + Send。
- 作用：完成某个 chunk 的最终规约、写入输出，并立即向 next 传播完整 chunk。
- 在 Ring AllReduce 中连接 ReduceScatter 结束与 AllGather 开始。

### `directRecvCopyDirectSend()`

- 名称拆分：Receive + Copy + Send。
- 作用：接收已经完成规约的 chunk，保存到输出并继续转发；不再进行 reduce。

### `directRecv()`

- 作用：接收并保存最后一个需要的完整 chunk，结束当前轮 AllGather。

名称中的 `direct` 是 NCCL primitive 路径术语，不应脱离具体协议简单解释成“必然绕过所有中间 buffer”。

## 13. Proxy 与 Host 网络推进

### Proxy

- 英文：Proxy，代理。
- 在 NCCL 中指 Host 侧替 GPU 推进某些通信操作的机制。

### Proxy thread

- 类型：NCCL 创建的 Host CPU 后台线程。
- 作用：读取 Proxy 工作、调用 transport/network plugin、检查完成并更新连接进度。
- 它与用户调用 `ncclAllReduce()` 的 Host 线程不是同一条执行流。
- 一个 Channel 不等于一个 Proxy 线程；一个线程可推进多个操作。

### `ncclProxyOp`

- 完整类型：`struct ncclProxyOp`。
- 类型：Host 结构体，Proxy 工作描述。
- 内容：connection、字节数、Channel、算法、协议、step/chunk 参数、rank/peer 等。
- 它是任务单，不是线程。

### `ncclProxyArgs` / `ncclProxySubArgs`

- 类型：Proxy progress 期间使用的 Host 状态结构体。
- `sub`：sub-operation，子操作。
- `ncclProxySubArgs` 中可保存 posted、received、transmitted、done、网络 request 等进度。
- 这些内部字段随版本变化，不能未经 2.21.5 核对就直接当作 Mycroft 指标来源。

### `ncclAddProxyOpIfNeeded()`

- 类型：内部 Host 函数。
- 作用：如果当前工作需要 Proxy，则复制并连接相应 `ncclProxyOp` 到 plan。
- 名称中的 `IfNeeded` 表明不是所有路径都必须创建同样的 Proxy 工作。

### `uploadProxyOps()`

- 类型：内部 Host 函数。
- 作用：把 plan 中已经组织好的 Proxy 工作提交给 Proxy 子系统。
- 它与 `uploadWork()` 的目标不同：前者面向 Proxy，后者面向 GPU kernel work。

### Progress

- 英文：Progress，推进。
- 含义：非阻塞操作可能不会一次函数调用就完成，需要反复检查条件、提交下一段、测试完成并更新状态。

### `proxyProgress`

- 类型：`ncclTransportComm` 函数表中的函数指针成员。
- 作用：不同 transport 提供自己的 Proxy 推进实现。
- 调用 `tcomm->proxyProgress(...)` 属于函数指针间接调用。

## 14. Transport 和网络插件

### Transport

- 英文：Transport，传输机制。
- 内部类型：`struct ncclTransport`。
- 由 `canConnect`、send 函数表和 recv 函数表组成。
- 主要实例：`p2pTransport`、`shmTransport`、`netTransport`、`collNetTransport`。

### `ncclTransportComm`

- 类型：函数表结构体。
- 分 send 和 recv 两组。
- 成员包括 setup、connect、free、proxySetup、proxyConnect、proxyProgress 等函数指针。

### P2P transport

- 类型：同机 peer 数据路径。
- 常涉及 CUDA P2P、CUDA IPC 或直接 GPU 访问能力。
- 不等同于 `ncclSend/ncclRecv` 操作本身。

### SHM transport

- 类型：同机进程间共享内存数据路径。
- 当 GPU 直接 P2P 不可用或不适合时，可能使用 Host shared memory 中转。

### NET transport

- 类型：跨机或需要网络插件的数据路径。
- 通过 NCCL NET API 对接 IB/RoCE/socket 等网络实现。

### Network plugin

- 英文：Network Plugin，网络插件。
- 类型：实现 NCCL NET 函数表的动态组件或内建实现。
- 负责连接、内存注册、发送、接收、测试完成等网络操作。

### `isend()` / `irecv()`

- 前缀 `i` 通常表示 asynchronous/immediate 风格的非阻塞接口。
- `isend`：提交发送请求。
- `irecv`：提交接收请求。
- 返回成功不一定表示真实数据传输已经完成。

### `test()`

- 类型：网络插件完成检查接口。
- 作用：查询某个异步 request 是否完成，并可能返回实际完成字节数等信息。

### Request

- 英文：Request，请求对象。
- 类型：网络插件返回并维护的异步操作状态。
- Proxy 后续使用 `test()` 等接口推进或确认完成。

### Head / Tail

- 英文：Head / Tail，头部/尾部进度。
- 类型：环形 FIFO 或连接控制状态中的计数器概念。
- 常用于协调生产者和消费者已经生产、消费或完成到哪个 step。
- 具体谁写 head、谁写 tail 会因 send/recv、protocol 和 transport 而变化，不能脱离具体代码背固定规则。

### Posted / Received / Transmitted / Done

- `posted`：已经准备或发布到某个处理阶段的数量。
- `received`：已经接收的数量。
- `transmitted`：已经向发送/传输阶段推进的数量。
- `done`：已经完成的数量。
- 它们通常是累计进度，不一定是字节数；具体单位必须查看字段所在结构和更新代码。

## 15. 数据 buffer 与控制状态

### `sendbuff`

- 类型：指向本地输入 buffer 的指针。
- 公共 API 参数类型通常为 `const void *`。
- 指针是当前进程的本地值，跨 rank 不要求相同。

### `recvbuff`

- 类型：指向本地输出 buffer 的指针。
- AllReduce 完成后保存当前 rank 的完整规约结果。

### Buffer

- 英文：Buffer，缓冲区。
- 保存真实数据或中间数据的连续内存区域。
- 数据 buffer 与 head/tail/size 等控制状态必须区分。

### Control state

- 英文：Control State，控制状态。
- 保存数据是否 ready、哪个 step 已提交、哪个 step 已完成等少量元数据。
- GPU、Proxy 和 NIC 通过数据 buffer 加控制状态协同工作。

### Ready

- 英文：Ready，已准备。
- 表示某个阶段的前置条件满足。
- 不能只看到字段名 `ready` 就断言它等同于 Mycroft 的 `GPU_ready`；需要确认谁更新、单位和生命周期。

### Completion

- 英文：Completion，完成。
- 可以指一次网络 request、一个 slice、一个 chunk、一个 Channel、一个 kernel 或整个 collective 的完成。
- 阅读时必须说明完成的是哪个层级。

## 16. Mycroft 观测字段

以下名称来自 Mycroft 论文的 trace schema。它们是分析字段，不等于已经确认的 NCCL 内部字段。

| 名称 | 英文/含义 | 用途 | 当前确认状态 |
|---|---|---|---|
| IP | Internet Protocol address | 标识主机网络地址 | 语义明确，采集位置待设计 |
| `comm_id` | Communicator identifier | 跨进程关联同一逻辑 communicator | NCCL 2.21.5 可靠来源待确认 |
| `Gid` | 论文中的全局/组标识字段 | 区分更高层执行实体 | 精确定义和来源待结合实现确认 |
| `GPU_id` | GPU identifier | 区分本机 GPU | 需要确定使用 cudaDev、busId、UUID 或其他稳定标识 |
| `Channel_id` | Channel identifier | 区分 collective 内并行 Channel | 需要与 comm_id 等组合才有足够作用域 |
| `QP_id` | Queue Pair identifier | 区分 RDMA 连接 | 需从 NET/IB 或插件实现确认 |
| `op_name` | Operation name | AllReduce、AllGather 等操作名 | 可从 collective 类型映射 |
| `op_seq` | Operation sequence | communicator 内 collective 顺序 | 可靠、跨 rank 一致的来源待确认 |
| `msg_size` | Message size | 当前 collective 或流的数据规模 | 必须明确是 API bytes、Channel bytes 还是 chunk bytes |
| `GPU_ready` | GPU-ready progress | GPU 已准备的数据进度 | 2.21.5 插桩位置待确认 |
| `RDMA_transmitted` | RDMA transmitted progress | 已提交/传输的数据进度 | 2.21.5 插桩位置和单位待确认 |
| `RDMA_done` | RDMA completion progress | 已完成网络数据进度 | 2.21.5 插桩位置和单位待确认 |
| `stuck_time` | Stuck duration | 当前状态未推进的时间 | 由采集时间线计算或记录 |
| `total_chunks` | Total number of chunks | 操作需要完成的总 chunk 数 | 需与 NCCL 分块语义对齐 |

### `opCount`、`collOpCount` 与 `op_seq`

这三个名字不能直接画等号：

- `opCount`：可用于 Proxy 工作排序或连接操作编号，作用域与编码方式依实现而定。
- `collOpCount`：当前 plan 内 collective 计数成员。
- `op_seq`：Mycroft 需要的 communicator-local 跨 rank 操作序号。

在确认作用域、一致性和生命周期前，不能因为名称相似就把前两者直接当作 `op_seq`。

### `MinOp`

- 英文：Minimum Completed Operation，最小已完成操作进度。
- 用途：比较多 rank 的 collective 完成位置，找到最落后的操作边界。

### `MinData`

- 英文：Minimum Data Progress，最小数据进度。
- 用途：在相关操作中比较 rank/flow 的传输进度。

### Root cause / affected rank

- Root cause：根因，最早主动停止或异常推进的组件/rank。
- Affected rank：受影响对象，因为等待上游而变慢。
- Slowest rank：观察到最慢的 rank，不一定是根因。

## 17. 常见函数名中的英语动词

| 动词 | 常见含义 | 示例 |
|---|---|---|
| Get | 取得或生成供调用者使用的值 | `ncclGetUniqueId()` |
| Init | 建立对象的初始状态 | `ncclCommInitRank()` |
| Fill | 填写已有结构体 | `fillInfo()` |
| Gather | 从多个参与者收集数据 | `bootstrapAllGather()` |
| Check | 检查条件或能力 | `ncclEnqueueCheck()` |
| Append | 追加到当前任务集合或链表 | `taskAppend()` |
| Enqueue | 把对象指针放入队列 | `ncclTasksRegAndEnqueue()` |
| Prepare | 为后续阶段准备和整理 | `ncclPrepareTasks()` |
| Schedule | 分配执行资源、Channel 或 plan | `scheduleCollTasksToPlan()` |
| Finish | 完成对象的剩余组装 | `finishPlan()` |
| Upload | 放入另一个执行者可读取的区域 | `uploadWork()` |
| Launch | 提交异步执行 | `ncclLaunchKernel()` |
| Setup | 分配或准备连接一侧资源 | transport `setup` |
| Connect | 让两端资源形成可用连接 | transport `connect` |
| Register | 注册内存或资源供 transport 使用 | `proxyRegister` |
| Progress | 推进非阻塞状态机 | `proxyProgress` |
| Poll | 轮询一个或多个状态 | event/proxy polling |
| Test | 查询异步 request 是否完成 | NET `test()` |
| Reclaim | 回收 plan/task 等对象和资源 | `reclaimPlan` |
| Free | 释放或回收到资源管理器 | transport `free` |

同一个动词不保证函数已经完成全部工作。例如 `Launch` 通常只表示提交，不表示 GPU 执行完成。

## 18. C/C++ 源码阅读术语

### `struct`

- 结构体类型，定义一组字段的内存布局。
- `struct ncclTaskColl` 是类型；不是某个具体任务对象。

### Pointer / `*`

- 指针，保存某个对象的内存位置。
- `struct ncclTaskColl *task` 表示局部变量 `task` 是指针，不是结构体对象本身。

### Address-of / `&`

- 在表达式中，`&object` 取得对象的指针。
- 在 C++ 类型中，`T&` 表示引用；两种用法必须根据位置区分。

### Reference / 引用

- C++ 中已有对象的别名。
- `T&` 可修改引用；`const T&` 是只读引用。
- NCCL 核心代码仍大量使用 C 风格结构体指针。

### `typedef` / `using`

- 类型别名，不创建新对象。
- `ncclComm_t` 是 `struct ncclComm *` 的类型别名。

### Enum

- 英文：Enumeration，枚举。
- 用命名常量表达有限状态，例如 algorithm、protocol、Proxy op state。

### Macro

- 宏，预处理阶段的文本展开。
- `NCCLCHECK(...)` 等宏通常包含函数调用、错误检查和控制流。
- 阅读时不能把宏名当作普通空壳，应查看展开逻辑。

### Function pointer

- 函数指针，保存可调用函数的位置。
- transport 和 network plugin 通过函数表选择具体实现。

```c
tcomm->proxyProgress(...);
```

这是通过函数指针成员间接调用，不是编译时固定函数名调用。

### Callback

- 回调函数，被另一个模块或运行时在适当时机调用。
- CUDA Host callback、资源回收 callback 都属于这种模式。

### Template

- C++ 模板，让同一套代码按 datatype、reduction、algorithm、protocol 生成不同实现。
- `RunWorkColl<...>` 是模板类型实例，不是运行时凭空产生的新 task。

### `inline` / `__forceinline__`

- 允许或要求编译器尽量把函数体展开到调用位置，也与头文件定义规则有关。
- 不改变函数的逻辑职责。

### `constexpr`

- 表示值或函数在满足条件时可以在编译期求值。

### `extern "C"`

- 让 C++ 编译器使用 C linkage 导出公共符号。
- 不表示函数内部只能使用 C 语法。

## 19. 最容易混淆的概念对照

### communicator、UniqueId、rank、句柄

| 名称 | 解决的问题 |
|---|---|
| UniqueId | 让参与者进入同一次初始化 rendezvous |
| 逻辑 communicator | 定义一组共同通信的 rank |
| 本地 `ncclComm` 对象 | 保存当前进程当前 rank 的通信状态 |
| `ncclComm_t` 句柄 | 调用者引用本地 `ncclComm` 对象的值 |
| rank | 在所属 communicator 内区分参与者 |

### task、work、plan、kernel

| 名称 | 核心问题 |
|---|---|
| task | 用户请求做什么 |
| devWork | GPU 具体需要哪些数据和参数 |
| plan | 本次 kernel launch 怎样组织 task/work/Channel |
| kernel | GPU 上真正执行的程序 |

### Channel、chunk、step、slice

| 名称 | 核心问题 |
|---|---|
| Channel | 哪条逻辑并行通道负责这一段数据 |
| chunk | 当前 Ring 轮次推进哪一段数据 |
| step | chunk/连接进度前进到哪一步 |
| slice | chunk 内更细的流水单位 |

### Proxy 线程、Proxy op、NET、NIC

| 名称 | 类型与作用 |
|---|---|
| Proxy thread | Host CPU 执行者 |
| `ncclProxyOp` | 交给 Proxy 的工作描述结构体 |
| NET transport | NCCL 统一网络传输层 |
| Network plugin | NET 接口的具体实现 |
| NIC | 实际传输数据的硬件 |

### Host 返回、launch、网络提交、完成

```text
ncclAllReduce() 返回
    不等于 GPU kernel 完成

ncclLaunchKernel() 返回
    不等于 kernel 已执行完成

isend/irecv 返回
    不等于网络 request 完成

一次 request 完成
    不等于 Channel 完成

一个 Channel 完成
    不等于整个 collective 完成
```

### 数据流与控制流

```text
真实数据流：
GPU buffer -> NIC -> 网络 -> 远端 buffer

控制/任务流：
API -> task -> work/plan -> GPU 与 Proxy

进度流：
ready -> transmitted -> done
```

## 20. 源码索引

| 主题 | 本地源码位置 |
|---|---|
| 公共 NCCL 类型和 API | [`src/nccl.h.in`](../nccl/src/nccl.h.in) |
| `ncclAllReduce()` 等 collective API | [`src/collectives.cc`](../nccl/src/collectives.cc) |
| communicator 初始化、peerInfo 交换 | [`src/init.cc`](../nccl/src/init.cc) |
| bootstrap | [`src/bootstrap.cc`](../nccl/src/bootstrap.cc) |
| `ncclPeerInfo`、transport 函数表 | [`src/include/transport.h`](../nccl/src/include/transport.h) |
| `ncclComm`、task、planner、plan、Channel | [`src/include/comm.h`](../nccl/src/include/comm.h) |
| task、devWork、plan、launch | [`src/enqueue.cc`](../nccl/src/enqueue.cc) |
| group 提交顺序 | [`src/group.cc`](../nccl/src/group.cc) |
| `ncclDevWorkColl`、分块、kernel args | [`src/include/device.h`](../nccl/src/include/device.h) |
| GPU kernel 主入口和模板 dispatch | [`src/device/common.h`](../nccl/src/device/common.h) |
| Ring AllReduce | [`src/device/all_reduce.h`](../nccl/src/device/all_reduce.h) |
| Simple primitive | [`src/device/prims_simple.h`](../nccl/src/device/prims_simple.h) |
| Proxy 类型与进度状态 | [`src/include/proxy.h`](../nccl/src/include/proxy.h) |
| Proxy 主体实现 | [`src/proxy.cc`](../nccl/src/proxy.cc) |
| NET transport | [`src/transport/net.cc`](../nccl/src/transport/net.cc) |
| NET/IB 实现 | [`src/transport/net_ib/`](../nccl/src/transport/net_ib/) |

## 21. 官方参考

- [NCCL 2.21.5 文档首页](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/index.html)
- [NCCL 2.21.5 Creating a Communicator](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/usage/communicators.html)
- [NCCL 2.21.5 Collective Operations](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/usage/collectives.html)
- [NCCL 2.21.5 Group Calls](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/usage/groups.html)
- [NCCL 2.21.5 CUDA Stream Semantics](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/usage/streams.html)

## 22. 阅读一个新名字时的固定检查法

遇到陌生名称，依次问：

```text
1. 它是结构体、指针、函数、成员、队列、线程还是硬件？

2. 它属于 Host、GPU、Proxy、transport 还是 NIC？

3. 它描述的是任务、真实数据，还是进度状态？

4. 谁创建或更新它？谁读取它？

5. 它的作用域是什么：进程、communicator、rank、Channel、connection 还是单次操作？

6. 它的生命周期从哪里开始，到哪里结束？

7. 它确认了哪一层“完成”：API、kernel、网络 request、Channel 还是 collective？
```

只有回答完这些问题，才适合把该名称用于源码插桩或 Mycroft 事件设计。

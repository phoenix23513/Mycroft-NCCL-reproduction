# NCCL 源码学习进展

最后更新：2026-08-04

## 项目目标

围绕 Mycroft 复现需求学习 NCCL：理解集合通信的执行过程，定位 Host、GPU 与网络侧观测点，构造性能异常，并依据事件依赖和运行状态定位根因。

目标版本为 NCCL 2.21.5。当前先使用本地 master 熟悉主干结构，正式插桩前再迁移到 2.21.5。

## 当前进度

已完成 communicator 初始化、AllReduce Host 任务创建、任务准备、kernel plan 生成、work 上传和 GPU kernel 提交主线；当前到达 Ring AllReduce 的 `runRing()` 入口，尚未深入 device primitive、Proxy 和 RDMA。

```text
ncclCommInitRank()
  -> bootstrap / peerInfo / topology / transport

ncclAllReduce()
  -> ncclInfo
  -> ncclEnqueueCheck()
  -> taskAppend()
  -> collTaskAppend()
  -> ncclTaskColl
  -> comm->planner.collSorter
  -> ncclPrepareTasks()
  -> comm->planner.collTaskQueue
  -> ncclTasksRegAndEnqueue()
  -> ncclDevWorkColl / collWorkQueue
  -> ncclLaunchPrepare()
  -> scheduleCollTasksToPlan()
  -> ncclKernelPlan / planQueue
  -> finishPlan() / uploadWork()
  -> ncclLaunchKernel()
  -> ncclKernelMain()
  -> RunWorkColl
  -> runRing()   <- 当前进度
```

## 已完成内容

### Communicator 初始化

- 梳理了外部调用者、CUDA device、`nranks`、rank 和 `ncclUniqueId` 的确定顺序。
- 明确 `ncclUniqueId` 需要由 MPI、socket、store 等 NCCL 外部机制分发。
- 梳理了本地 `ncclComm` 对象创建、`bootstrapInit()`、`fillInfo()`、`bootstrapAllGather()`、拓扑和 transport 初始化的依赖关系。
- 明确每个进程持有独立的本地 communicator 指针；rank 只在所属 communicator 内唯一。
- 明确同一进程可持有多个 communicator 指针，每次 `ncclAllReduce()` 只使用传入的一个 communicator。

### AllReduce Host 任务创建

- 追踪了 `ncclAllReduce() → ncclEnqueueCheck() → taskAppend() → collTaskAppend()`。
- 区分了局部 `struct ncclInfo info` 和长期 `struct ncclTaskColl` 任务对象。
- 梳理了任务内存池、`comm->memPermanent` 和任务回收复用过程。
- 梳理了 `comm->planner`、`collSorter`、`collTaskQueue` 和 `planQueue` 的包含关系。
- 明确任务指针只适合本地单次生命周期调试，不能作为跨进程或永久操作 ID。

### 任务准备与 kernel plan

- 追踪了 `ncclPrepareTasks()` 对 collective task 的分类、算法与协议选择，以及 `algorithm`、`protocol`、`nMaxChannels`、`nWarps` 和 `devFuncId` 等字段的写入。
- 追踪了 `ncclTasksRegAndEnqueue()` 将 Host 端 `ncclTaskColl` 转换为 `ncclDevWorkColl`，并建立 `collTaskQueue` 与 `collWorkQueue` 的对应关系。
- 追踪了 `ncclLaunchPrepare() → scheduleCollTasksToPlan()` 创建 `ncclKernelPlan`、分配 Channel、生成 work batch，并将 task 和 work 指针转移到 plan 的过程。
- 区分了 task、device work、work batch 和 kernel plan 的职责与包含关系。

### GPU kernel 提交

- 追踪了 `finishPlan()` 创建 `ncclDevKernelArgs`、组织各 Channel 的 `ncclDevWorkBatch`，并选择 Args、FIFO 或 Persistent work 存储方式的过程。
- 追踪了 `uploadWork()` 将 `ncclDevWorkColl` 复制到 GPU 可读取位置的过程。
- 追踪了 `ncclLaunchKernel()` 根据 `channelMask` 设置 grid、根据 `threadPerBlock` 设置 block，并向 NCCL 使用的 CUDA stream 提交 kernel。
- 明确一个被启动的 CUDA block 对应一个有效 Channel；GPU 通过 `channelMask`、batch 和 work 偏移找到本 Channel 的 `ncclDevWorkColl`。
- 已追踪至 `ncclKernelMain() → RunWorkColl → runRing()`，尚未展开 Ring 的数据传递步骤。

### 官方文档对照

已将 NVIDIA User Guide 中 communicator、collective、group 和 CUDA stream 的公开语义与当前源码路径对照：

- communicator 中每个 CUDA device 对应唯一 rank；
- collective 需要所有参与 rank 使用匹配的操作、`count` 和 `datatype`；
- group 用于多 GPU 提交、collective 聚合和 P2P 合并；
- NCCL Host 调用返回不等于 GPU collective 已完成。

## 待完成内容

| 内容 | 状态 |
|---|---|
| `ncclPrepareTasks()` 任务整理与方案字段 | 主线已完成，选择模型待深入 |
| `ncclTaskColl → ncclDevWorkColl → ncclKernelPlan` | 已完成 |
| Channel 分配、work batch 与 kernel 参数 | 已完成主线 |
| Ring AllReduce 的 ReduceScatter 与 AllGather | 进行中 |
| GPU device primitive | 未开始 |
| Proxy、NET/IB 和 RDMA 进度 | 未开始 |
| Mycroft 三个进度指标的源码落点 | 未开始 |
| NCCL 2.21.5 tracepoint 实现 | 未开始 |
| 多 rank、跨节点故障验证 | 未开始 |

## 环境与阻塞

- 本地源码：`<NCCL_SOURCE_DIR>`
- 当前分支：`master`
- 当前 commit：`5067397c2676d5aed50042fc39e5c8ee96eb0027`
- 当前可见 GPU：1 张 RTX 5060 Laptop GPU
- WSL 中尚未发现 CUDA Toolkit 和 `nvcc`，当前无法编译修改后的 NCCL。
- 单 GPU `world_size=1` 不能验证真实跨 rank collective、Channel 或 RDMA。
- Mycroft 使用 NCCL 2.21.5，与当前 master 存在版本差异。

## 下一项工作

分析 `runRing()`、`ncclCollCbdPart()` 和 Ring 使用的 `Primitives`，确认：

- 单个 Channel 负责的数据范围和 chunk 划分方式；
- 四个 rank 完成 ReduceScatter 和 AllGather 的实际 send/recv/reduce 顺序；
- GPU primitive 与 Channel 的 send/recv 连接如何衔接；
- 多 Channel 如何并行处理同一个 AllReduce 的不同数据范围。

## 产出

- [阶段 01：Communicator 初始化与 AllReduce Host 任务提交](notes/阶段01_Communicator与AllReduce任务提交.md)

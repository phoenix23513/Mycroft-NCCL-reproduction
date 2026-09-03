# 第二份任务提交：从 Host 任务到 GPU Ring AllReduce 执行

本文承接第一份任务提交《NCCL Communicator 与 AllReduce Host 提交路径》。第一份文档结束于 `ncclTaskColl` 进入 planner；本文继续追踪任务如何变成 GPU 可以读取的工作描述、如何被 `ncclKernelPlan` 组织并提交到 CUDA stream，以及 Ring AllReduce 在 GPU 上怎样完成 ReduceScatter 和 AllGather。

本阶段同时补充了《面向 NCCL 源码阅读的 C++ 补充计划》的前两个阶段：C++ 基础语法和底层源码常用机制。本文不把这些语法单独罗列，而是放到对应的 NCCL 源码位置解释。

源码基线：本地 NCCL master，commit `5067397c2676d5aed50042fc39e5c8ee96eb0027`。Mycroft 使用 NCCL 2.21.5；正式确定插桩位置时必须重新对照 2.21.5，不能直接把当前 master 的字段和函数位置当作 2.21.5 结论。

## 1. 本阶段在完整流程中的位置

```text
第一份任务提交已经梳理

ncclAllReduce()
  -> struct ncclInfo
  -> struct ncclTaskColl
  -> ncclPrepareTasks()
  -> comm->planner.collTaskQueue

本文继续

comm->planner.collTaskQueue
  -> ncclTasksRegAndEnqueue()
  -> struct ncclDevWorkColl
  -> comm->planner.collWorkQueue
  -> ncclLaunchPrepare()
  -> 分配 struct ncclKernelPlan
  -> scheduleCollTasksToPlan()
  -> Channel 分配、chunk 参数和 work batch
  -> finishPlan()
  -> struct ncclDevKernelArgs
  -> uploadWork()
  -> ncclLaunchKernel()
  -> CUDA stream 中的 NCCL GPU kernel
  -> ncclKernelMain()
  -> RunWorkBatch
  -> RunWorkColl<AllReduce, Ring, Simple>
  -> runRing()
  -> ReduceScatter + AllGather
```

这条路径跨越两种执行环境：

```text
Host CPU 执行
  ncclTasksRegAndEnqueue()
  ncclLaunchPrepare()
  scheduleCollTasksToPlan()
  finishPlan()
  uploadWork()
  ncclLaunchKernel()
       |
       | 向 CUDA stream 提交 kernel
       v
CUDA device 执行
  ncclKernelMain()
  RunWorkBatch
  RunWorkColl
  runRing()
  Primitives
```

`ncclLaunchKernel()` 返回只表示 Host 已经完成 kernel 提交；集合通信随后由 CUDA device 异步执行。是否完成需要根据 CUDA stream 或 event 判断。

## 2. 四个核心对象及其先后顺序

### 2.1 类型和用途

| 名称 | 类型 | 所在位置 | 解决的问题 |
|---|---|---|---|
| `struct ncclTaskColl` | Host 结构体 | Host 内存 | 保存一次用户 collective 的语义和调度选择 |
| `struct ncclTaskColl *task` | 结构体指针变量 | Host 函数局部变量或队列 | 指向一个已有 task 对象 |
| `struct ncclDevWorkColl` | GPU 工作描述结构体 | 先在 Host 构造，随后放入 GPU 可读区域 | 保存 buffer、规约参数、Channel 范围和分块参数 |
| `struct ncclKernelPlan` | Host 结构体 | Host 内存 | 组织一次 kernel launch 包含的 task、work、batch、Channel 和 Proxy 工作 |
| `struct ncclKernelPlan *plan` | 结构体指针变量 | Host 调度函数 | 指向本次 launch plan |
| `struct ncclDevKernelArgs` | kernel 参数结构体 | Host 准备，GPU kernel 读取 | 保存 device communicator、Channel mask、work 存储位置等 kernel 入口信息 |

### 2.2 正确的创建顺序

```text
ncclTaskColl 已存在
       |
       v
ncclTasksRegAndEnqueue()
创建 ncclDevWorkColl
       |
       v
ncclLaunchPrepare()
分配 ncclKernelPlan
       |
       v
scheduleCollTasksToPlan()
由 plan 组织已经存在的 task 和 devWork
       |
       v
finishPlan()
创建 ncclDevKernelArgs 和 work batch
```

因此，`ncclDevWorkColl` 在 `ncclKernelPlan` 之前创建。plan 不是先生成一个空框架再产生 devWork；它是在后面把已经对应好的 task 和 devWork 编入一次 launch。

## 3. `ncclTaskColl` 如何变成 `ncclDevWorkColl`

入口函数：

```c
ncclResult_t ncclTasksRegAndEnqueue(struct ncclComm *comm);
```

- `ncclTasksRegAndEnqueue`：Host 函数。
- `comm`：指向当前本地 communicator 对象的结构体指针。
- `planner`：函数内的 `struct ncclKernelPlanner *` 指针，指向 `comm->planner`。
- `task`：`struct ncclTaskColl *` 指针，依次指向 `collTaskQueue` 中的 task。

函数首先取得 task：

```c
struct ncclTaskColl *task;
task = ncclIntruQueueHead(&planner->collTaskQueue);
```

然后在循环中创建局部结构体值：

```c
struct ncclDevWorkColl devWork = {};
```

这里的 `devWork` 不是指针，而是当前 Host 函数栈上的结构体变量。函数把 task 中 GPU 执行需要的字段复制到它：

```c
devWork.sendbuff = (void *)task->sendbuff;
devWork.recvbuff = (void *)task->recvbuff;
devWork.root = task->root;
devWork.nWarps = task->nWarps;
devWork.redOpArg = task->opDev.scalarArg;
```

随后分配一个 `struct ncclWorkList *workNode`，把 `devWork` 的内容复制到紧跟在节点后的内存：

```c
workNode = ncclMemoryStackAllocInlineArray<
    ncclWorkList,
    ncclDevWorkColl
>(&comm->memScoped, 1);

memcpy(
    (void *)(workNode + 1),
    (void *)&devWork,
    sizeof(struct ncclDevWorkColl)
);
```

最后，队列保存 `workNode` 指针：

```c
ncclIntruQueueEnqueue(&planner->collWorkQueue, workNode);
```

内存关系可以用 C 风格表示为：

```text
workNode 指针
      |
      v
+----------------------+-------------------------+
| struct ncclWorkList  | struct ncclDevWorkColl  |
+----------------------+-------------------------+
                        ^
                        |
                (workNode + 1)
```

局部变量 `devWork` 在函数迭代结束后可以消失，因为它的内容已经复制到了 `workNode` 后面的内存。

源码：[`src/enqueue.cc`](../nccl/src/enqueue.cc)、[`src/include/device.h`](../nccl/src/include/device.h)

## 4. task 与 devWork 为什么同时存在

二者描述的是同一次 collective，但服务于不同阶段：

| 对象 | 主要使用者 | 保留的内容 |
|---|---|---|
| `ncclTaskColl` | Host 调度、清理、Profiler 和 Proxy 准备逻辑 | API 语义、算法、协议、datatype、task 生命周期信息 |
| `ncclDevWorkColl` | GPU kernel | buffer 指针、规约参数、Channel 范围、每个 Channel 的数据范围和 chunk 大小 |

不能简单地认为 devWork 创建后 task 就无用了。Host 在 plan 生命周期中仍需要 task，例如：

- 将 task 连接到 `plan->collTaskQueue`，统一管理和回收；
- 生成与 Channel 对应的 Proxy 工作时读取 task 信息；
- 保存 Profiler 和清理相关状态。

而 GPU 不需要读取完整的 Host task，所以使用更紧凑的 `ncclDevWorkColl`。

## 5. `ncclKernelPlan` 在什么时候创建

`ncclLaunchPrepare()` 检查 planner 中是否存在待处理任务。如果有，就从 plan 内存池分配：

```c
struct ncclKernelPlan *plan =
    ncclMemoryPoolAlloc<struct ncclKernelPlan>(
        &comm->memPool_ncclKernelPlan,
        &comm->memPermanent
    );
```

- `plan`：局部结构体指针变量。
- `&comm->memPool_ncclKernelPlan`：指向 plan 空闲对象池。
- `&comm->memPermanent`：指向后备 Host 内存。

这两个参数是并列的“对象池 + 后备内存”，不是递进包含关系。

plan 初始建立后，`scheduleCollTasksToPlan()` 同时读取：

```c
task = ncclIntruQueueHead(&planner->collTaskQueue);
workNode = ncclIntruQueueHead(&planner->collWorkQueue);
```

两个队列保持相同顺序，因此当前 task 与当前 workNode 对应同一次 collective。

## 6. `scheduleCollTasksToPlan()` 做了什么

这一函数不是再次创建 collective，而是把已有 task 和 devWork 具体化为可执行方案。

### 6.1 填写 Channel 范围和分块参数

函数通过指针取得 workNode 后面的 devWork：

```c
struct ncclDevWorkColl *devWork =
    (struct ncclDevWorkColl *)(workNode + 1);
```

随后填写：

```text
devWork->channelLo / channelHi
  当前 work 使用哪些 Channel

devWork->cbd.countLo / countMid / countHi
  最低、中间、最高 Channel 分别负责多少元素

devWork->cbd.chunkGrainsLo / Mid / Hi
  各类 Channel 每次处理的 chunk 大小
```

这里说明：真实 NCCL 的 chunk 不是固定的一个标量。它是某个 Channel 所负责数据范围中的一段，大小由消息量、datatype、算法、协议和 Channel 数共同决定。

### 6.2 为每个有效 Channel 建立 work batch

```c
ncclAddWorkBatchToPlan(
    comm,
    plan,
    channelId,
    workNode->workType,
    task->devFuncId,
    plan->workBytes
);
```

`struct ncclDevWorkBatch` 不复制完整 collective 数据；它告诉 GPU 某个 Channel 应从 work 存储区的哪个偏移读取多少个 work，并使用哪个 device function。

### 6.3 设置 kernel launch 信息

```c
plan->channelMask |= ...;
plan->threadPerBlock = max(...);
plan->kernelFn = ncclDevKernelForFunc[task->devFuncId];
```

- `channelMask`：位集合，标记本 plan 使用哪些 Channel。
- `threadPerBlock`：每个 CUDA block 的线程数。
- `kernelFn`：指向要启动的 GPU kernel 的函数指针。

### 6.4 把对象从 planner 转移到 plan

```c
ncclIntruQueueDequeue(&planner->collTaskQueue);
ncclIntruQueueDequeue(&planner->collWorkQueue);

ncclIntruQueueEnqueue(&plan->collTaskQueue, task);
ncclIntruQueueEnqueue(&plan->workQueue, workNode);
```

所以 plan 同时组织 task 和 work：

```text
struct ncclKernelPlan
|
|-- collTaskQueue
|     \-- struct ncclTaskColl
|
|-- workQueue
|     \-- workNode + struct ncclDevWorkColl
|
|-- proxyOpQueue
|     \-- Host Proxy 后续处理的网络工作
|
|-- channelMask
|-- threadPerBlock
|-- kernelFn
\-- kernelArgs
```

源码：[`src/enqueue.cc`](../nccl/src/enqueue.cc)、[`src/include/comm.h`](../nccl/src/include/comm.h)

## 7. `finishPlan()` 与 kernel 参数

`finishPlan()` 为 plan 创建 `struct ncclDevKernelArgs`：

```c
plan->kernelArgs =
    (struct ncclDevKernelArgs *)ncclMemoryStackAlloc(...);

plan->kernelArgs->comm = comm->devComm;
plan->kernelArgs->channelMask = plan->channelMask;
plan->kernelArgs->workStorageType = plan->workStorageType;
```

随后把各 Channel 的 `ncclDevWorkBatch` 排列在 kernel 参数区中。每个有效 Channel 的第一个 batch 可以由对应 CUDA block 找到。

work 有三种存储方式：

| 类型 | 含义 |
|---|---|
| `Args` | work 足够小时，直接放在 kernel 参数区域 |
| `Fifo` | 普通非持久任务放在 communicator 的 work FIFO |
| `Persistent` | CUDA Graph 等持久场景使用独立的持久 work buffer |

`uploadWork()` 根据上述类型，把 `plan->workQueue` 中的 work 数据复制到 GPU kernel 能读取的位置。这里的“upload”不能简单理解为每次都执行一次普通 Host-to-Device memcpy：Args、映射 FIFO 和 Persistent 路径的处理方式不同。

源码：[`src/enqueue.cc`](../nccl/src/enqueue.cc)、[`src/include/device.h`](../nccl/src/include/device.h)

## 8. `ncclLaunchKernel()` 如何启动 GPU 工作

核心参数：

```c
int nChannels = countOneBits(plan->channelMask);

dim3 grid = {(unsigned)nChannels, 1, 1};
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};

cudaStream_t launchStream = planner->streams->stream;
```

当前普通 launch 中：

- grid 中的 block 数等于有效 Channel 数；
- 每个 block 使用 `threadPerBlock` 个线程；
- kernel 被提交到 planner 收集的 CUDA stream；
- `plan->kernelArgs` 作为 kernel 参数；
- `plan->kernelFn` 决定实际启动哪个 kernel。

因此可以先建立这个对应关系：

```text
一个有效 Channel
  <-> 当前 launch 中的一个 CUDA block
```

这是当前普通 collective launch 的结构关系，不表示一个 Channel 永远只能由一个物理 GPU SM 执行，也不表示 Channel 就是 CUDA block。

根据 NCCL 2.21.5 官方文档，collective API 与传入的 CUDA stream 关联；Host API 返回后，GPU 仍可继续异步执行。`ncclLaunchKernel()` 被调用也不能证明 Ring 或网络传输已经完成。

源码：[`src/group.cc`](../nccl/src/group.cc)、[`src/enqueue.cc`](../nccl/src/enqueue.cc)

## 9. GPU kernel 如何找到本 Channel 的工作

GPU 入口为：

```c
ncclKernelMain(struct ncclDevKernelArgs const *args)
```

它首先读取：

- `threadIdx.x`：当前 block 内线程编号；
- `blockIdx.x`：当前 block 编号；
- `args->channelMask`：本 plan 的有效 Channel 位集合。

kernel 通过 `channelMask` 把 `blockIdx.x` 映射为实际 `channelId`：

```text
channelMask = 00101010
               ^ ^ ^
             有效 Channel 1、3、5

block 0 -> Channel 1
block 1 -> Channel 3
block 2 -> Channel 5
```

之后，各线程协作加载：

```text
device communicator 状态
当前 ncclDevChannel
当前 Channel 的 work batch
batch 指向的 ncclDevWorkColl
```

`RunWorkBatch` 遍历本 batch 中的 work，然后调用：

```c
RunWorkColl<Fn, T, RedOp, Algo, Proto>().run(...);
```

对于 `AllReduce + Ring + Simple`，模板特化最终进入：

```c
runRing<T, RedOp, Proto>(tid, nthreads, work);
```

源码：[`src/device/common.h`](../nccl/src/device/common.h)、[`src/device/all_reduce.h`](../nccl/src/device/all_reduce.h)

## 10. Channel 是什么

Channel 是 NCCL 把同一次 collective 数据切分后并行推进的逻辑执行通道。当前路径中，每个 Channel 至少关联：

- 自己负责的数据范围；
- 一个 `channelId`；
- Ring 的 `prev`、`next` 和顺序信息；
- 对应的 send/recv 连接；
- 本 Channel 的 work batch；
- 后续可能需要的 Proxy 工作。

它不是：

- 一个 rank；
- 一张 GPU；
- 一个 communicator；
- 固定的一条物理网络线；
- 一个永久等同于 CUDA block 的对象。

一次 AllReduce 可以使用多个 Channel。各 Channel 处理输入向量的不同范围，并行执行类似的 Ring 步骤，最后共同覆盖整个 `recvbuff`。

`ncclCollCbdPart()` 使用 `channelId` 和 `ncclDevWorkColl` 中的 count/chunk 字段，计算当前 Channel 的：

```text
gridOffset   当前 Channel 数据范围的起点
channelCount 当前 Channel 负责的元素数
chunkCount   当前循环中一个 chunk 的元素数
```

## 11. Ring AllReduce 的数据模型

官方语义是：每个 rank 提供一个包含 `N` 个元素的输入向量，AllReduce 完成后每个 rank 得到相同的 `N` 元素结果。以 sum 为例：

```text
out[i] = in_rank0[i] + in_rank1[i] + ... + in_rank(k-1)[i]
```

为了手工追踪，使用以下明确的简化条件：

```text
4 个 rank
Ring 顺序为 0 -> 1 -> 2 -> 3 -> 0
每个 rank 保存一个四元素向量
只使用一个 Channel
把每个元素暂时看作一个 chunk
```

输入：

| rank | 完整输入向量 |
|---|---|
| rank 0 | `[1, 2, 3, 4]` |
| rank 1 | `[10, 20, 30, 40]` |
| rank 2 | `[100, 200, 300, 400]` |
| rank 3 | `[1000, 2000, 3000, 4000]` |

按位置求和后，完整结果应为：

```text
[1111, 2222, 3333, 4444]
```

这个“一元素一个 chunk”的设定只用于手工说明。真实 NCCL 的 chunk 通常包含一段连续数据，而且一个 Channel 会循环处理多个 chunk。

## 12. ReduceScatter：先完成规约并分散结果

在上述固定 Ring 中，令 chunk `c` 的规约结果最终停在 rank `c`。初始时 rank `r` 先发送 chunk `(r-1) mod 4`。

| 逻辑 step | rank 0 发送 | rank 1 发送 | rank 2 发送 | rank 3 发送 |
|---:|---|---|---|---|
| 0 | c3 -> r1 | c0 -> r2 | c1 -> r3 | c2 -> r0 |
| 1 | c2 -> r1 | c3 -> r2 | c0 -> r3 | c1 -> r0 |
| 2 | c1 -> r1 | c2 -> r2 | c3 -> r3 | c0 -> r0 |

只跟踪 chunk 0：

```text
rank 1 的 chunk 0 本地贡献：10
  -> rank 2 收到并加上 100，得到 110
  -> rank 3 收到并加上 1000，得到 1110
  -> rank 0 收到并加上 1，得到 1111
```

这里每个 rank 始终保存完整输入向量。`10、100、1000、1` 只是各向量对 chunk 0 的本地贡献，不是每个 rank 的全部数据。

ReduceScatter 完成后：

```text
rank 0 保存规约完成的 c0 = 1111
rank 1 保存规约完成的 c1 = 2222
rank 2 保存规约完成的 c2 = 3333
rank 3 保存规约完成的 c3 = 4444
```

此时规约已经完成，但每个 rank 只拥有完整结果的一部分。

## 13. AllGather：传播已经规约完成的 chunk

AllGather 不再做加法。每个 owner 把已经完成的 chunk 发给 next rank；接收者保存后继续转发。

| 逻辑 step | rank 0 发送 | rank 1 发送 | rank 2 发送 | rank 3 发送 |
|---:|---|---|---|---|
| 0 | c0 -> r1 | c1 -> r2 | c2 -> r3 | c3 -> r0 |
| 1 | c3 -> r1 | c0 -> r2 | c1 -> r3 | c2 -> r0 |
| 2 | c2 -> r1 | c3 -> r2 | c0 -> r3 | c1 -> r0 |

三个 step 后，四个 rank 都拥有：

```text
c0 = 1111
c1 = 2222
c2 = 3333
c3 = 4444
```

所以从逻辑上可以写成：

```text
Ring AllReduce = ReduceScatter + AllGather
```

但在 `runRing()` 中，这两个阶段不是分别调用 `ncclReduceScatter()` 和 `ncclAllGather()` API。它们是在同一个 GPU kernel 的循环中连续完成的。

## 14. `runRing()` 中的五类 primitive

`runRing()` 创建局部对象：

```c
Primitives<...> prims(...);
```

- `Primitives<...>`：模板类。
- `prims`：局部对象，不是指针。
- `prims.directSend()` 等：成员函数调用。

Ring Simple 路径的顺序为：

| 源码调用 | 执行含义 | 所属逻辑阶段 |
|---|---|---|
| `directSend` | 把本地初始 chunk 发给 next | ReduceScatter 起点 |
| `directRecvReduceDirectSend` | 从 prev 接收，与本地数据规约，再发给 next | ReduceScatter 中间步骤 |
| `directRecvReduceCopyDirectSend` | 完成当前 rank 所负责 chunk 的最终规约、写入结果，并立即发给 next | ReduceScatter 结束与 AllGather 开始的边界 |
| `directRecvCopyDirectSend` | 接收完整 chunk、保存并转发 | AllGather 中间步骤 |
| `directRecv` | 接收并保存最后一个缺少的完整 chunk | AllGather 结束 |

中间的 `directRecvReduceCopyDirectSend()` 同时完成 ReduceScatter 的最后一次接收规约和 AllGather 的第一次发送。因此源码中的阶段边界是融合的，不能机械地按两个完全分离的循环理解。

## 15. 本阶段 C++ 知识如何用于源码阅读

### 15.1 `struct`、结构体指针和局部结构体值

```c
struct ncclDevWorkColl devWork = {};
struct ncclDevWorkColl *devWorkPtr;
```

第一行创建结构体值；第二行只创建一个指针变量。源码阅读时必须先区分对象和指向对象的指针。

### 15.2 引用与 `const`

本阶段补充了：

```text
T&       可修改引用
const T& 只读引用
const T* 指向只读对象的指针
T* const 不能改变指向的指针
```

NCCL 主路径仍大量采用 C 风格结构体指针。理解引用后，可以判断 C++ 辅助函数是在复制对象，还是直接操作调用者的对象；讲解 NCCL 对象关系时仍优先写清楚“哪个指针指向哪个结构体”。

### 15.3 `using`、`auto`、lambda 与 `constexpr`

`runRing()` 中：

```c
auto modRanks = [&] __device__(int r) -> int { ... };
```

- `auto` 让编译器推导 lambda 对象类型；
- `[&]` 表示按引用捕获外部变量；
- `__device__` 表示该 lambda 在 GPU device 代码中使用；
- `modRanks` 用于把可能越界一次的 Ring 下标映射回 `[0, nranks-1]`。

`scheduleCollTasksToPlan()` 中：

```c
constexpr size_t MinTrafficPerChannel = 32 << 10;
```

`constexpr` 表示该常量可以在编译期确定。

### 15.4 模板和模板特化

```c
RunWorkColl<
    ncclFuncAllReduce,
    T,
    RedOp,
    NCCL_ALGO_RING,
    NCCL_PROTO_SIMPLE
>
```

尖括号中的内容是模板参数，分别确定 collective 类型、数据类型、规约、算法和协议。针对 `AllReduce + Ring + Simple` 的特化把通用 dispatch 连接到对应 `runRing()`。

模板使 NCCL 能为不同 datatype、reduction、algorithm 和 protocol 生成对应 device 实现，不代表运行时创建了一个新的 communicator 或 task。

### 15.5 函数指针和函数表

Host 侧：

```c
plan->kernelFn = ncclDevKernelForFunc[task->devFuncId];
```

GPU 侧：

```c
ncclDevFuncTable[ncclShmem.funcId]();
```

这两处都不是把固定函数名直接写死：索引先选择函数指针，再通过函数指针调用。`devFuncId` 把当前 collective、datatype、规约、算法和协议组合映射到对应实现。

### 15.6 宏与条件编译

NCCL 常用：

```c
NCCLCHECKGOTO(expression, result, failure);
```

宏展开后会执行表达式、检查返回值，并在失败时跳到清理路径。阅读时应把它还原为普通控制流，而不是忽略宏内部的函数调用。

`ncclLaunchKernel()` 中还存在：

```c
#if CUDART_VERSION >= ...
```

这表示不同 CUDA Toolkit 版本会编译不同 launch 路径。条件编译未满足的代码不会进入当前二进制文件。

### 15.7 声明、定义、头文件与链接

例如 `ncclLaunchKernel()` 的声明位于 `src/include/enqueue.h`，定义位于 `src/enqueue.cc`。阅读调用关系时：

```text
头文件声明
  -> 让其他源文件知道函数签名

.cc / .cu 中的定义
  -> 提供真正实现

编译和链接
  -> 把不同目标文件及 CUDA device 代码组合为 NCCL 库
```

只找到声明不等于已经找到实现。

### 15.8 `extern "C"`

NCCL 对外 API 使用 C linkage，使 C 和 C++ 调用者看到稳定的 C 符号。函数内部仍然可以使用模板、class、lambda 等 C++ 实现机制。

## 16. 对 Mycroft 复现的实际意义

本阶段建立了以下执行边界：

```text
Host task
  -> GPU work 描述
  -> kernel plan
  -> CUDA stream 中的 kernel
  -> 每个有效 Channel 的 CUDA block
  -> Ring primitive
```

这能够解释：

- Host API 返回不等于 GPU 通信完成；
- 同一次 task 可以被分配到多个 Channel；
- 一个 Channel 的 chunk 必须等待 Ring 前驱步骤；
- 某个 rank 延迟可能通过 chunk 依赖阻塞后继 rank；
- 最后完成的 rank 不一定是最早发生问题的 rank。

本阶段尚不能确定 Mycroft 的插桩位置。以下内容还没有系统完成：

- send/recv Proxy 操作的完整生命周期；
- GPU 与 Proxy 通过 head、tail、FIFO 协作的源码细节；
- NET/IB、QP 和 RDMA completion；
- `GPU_ready`、`RDMA_transmitted`、`RDMA_done` 在 NCCL 2.21.5 中的准确来源；
- 跨 rank 的 communicator 和 collective 操作身份。

这些是下一阶段需要继续追踪的内容，不能作为本阶段已经完成的成果。

## 17. 结构关系汇总

```text
struct ncclComm *comm
  |
  \-- comm->planner
        |
        |-- collTaskQueue
        |     \-- struct ncclTaskColl
        |
        |-- collWorkQueue
        |     \-- workNode + struct ncclDevWorkColl
        |
        \-- planQueue
              \-- struct ncclKernelPlan
                    |
                    |-- collTaskQueue -> ncclTaskColl
                    |-- workQueue -> ncclDevWorkColl
                    |-- proxyOpQueue
                    |-- ncclDevWorkBatch[]
                    |-- struct ncclDevKernelArgs *kernelArgs
                    |-- channelMask
                    |-- threadPerBlock
                    \-- kernelFn

GPU kernel
  |
  |-- blockIdx.x -> channelId
  |-- ncclDevChannel -> Ring prev / next
  |-- ncclDevWorkBatch -> ncclDevWorkColl
  \-- RunWorkColl<AllReduce, Ring, Simple>
        \-- runRing()
              |-- ReduceScatter
              \-- AllGather
```

## 18. 源码索引

| 主题 | 源码位置 |
|---|---|
| group 中任务准备和 launch 顺序 | [`src/group.cc`](../nccl/src/group.cc) |
| task 转 devWork、plan 调度、work 上传和 kernel launch | [`src/enqueue.cc`](../nccl/src/enqueue.cc) |
| `ncclKernelPlan`、planner 和 Host 队列 | [`src/include/comm.h`](../nccl/src/include/comm.h) |
| `ncclDevWorkColl`、`ncclDevKernelArgs` 和分块计算 | [`src/include/device.h`](../nccl/src/include/device.h) |
| GPU kernel 主入口和模板 dispatch | [`src/device/common.h`](../nccl/src/device/common.h) |
| Ring AllReduce 和五类 primitive 调用 | [`src/device/all_reduce.h`](../nccl/src/device/all_reduce.h) |
| Simple protocol primitive 实现 | [`src/device/prims_simple.h`](../nccl/src/device/prims_simple.h) |
| Host launch 函数声明 | [`src/include/enqueue.h`](../nccl/src/include/enqueue.h) |

## 19. 参考资料

- [NCCL 2.21.5 Collective Operations](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/usage/collectives.html)
- [NCCL 2.21.5 CUDA Stream Semantics](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/usage/streams.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- C++ 补充材料：`/mnt/d/CSlessons/NCCL/nccl_cpp_supplement_plan.md`

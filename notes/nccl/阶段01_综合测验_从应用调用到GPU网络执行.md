# NCCL 阶段 01 综合测验：从应用调用到 GPU—Proxy—网络执行

日期：2026-08-10
覆盖范围：NCCL 的用途、communicator 初始化、Host 任务准备、Kernel plan、GPU 执行、Proxy 线程、Ring AllReduce 和初步异常定位

## 测验目的

本测验检查的不是名词定义，而是能否沿着一次真实的 `ncclAllReduce()` 说明：

1. 为什么上层程序需要调用 NCCL；
2. 哪些参与者属于同一个通信组；
3. 初始化阶段准备了什么；
4. 一次 AllReduce 如何从 Host API 进入 GPU 和 Proxy 两条执行路线；
5. 数据传输停滞时，怎样根据运行状态判断问题所在阶段。

建议用时：60～90 分钟。第一遍尽量不查笔记；涉及源码结构的题目可以在第二遍查源码补充。

## 统一场景

有两台机器，每台机器有两张 GPU。训练程序采用“一进程一 GPU”的方式运行：

| 机器 | 进程 | 本机 GPU | communicator X 中的 rank |
|---|---|---:|---:|
| Node A | P0 | GPU 0 | 0 |
| Node A | P1 | GPU 1 | 1 |
| Node B | P2 | GPU 0 | 2 |
| Node B | P3 | GPU 1 | 3 |

四个进程使用同一个 `ncclUniqueId U` 创建 communicator X，`nranks = 4`。Channel 0 的 Ring 顺序规定为：

```text
rank 0 → rank 1 → rank 2 → rank 3 → rank 0
```

本测验中的普通网络发送采用 SIMPLE 协议和 NCCL 连接缓冲区，不考虑注册缓冲区、NVLS、CollNet 和网络设备卸载等特殊路径。

---

## 第 1 题：NCCL 在系统中的位置（8 分）

训练程序在四张 GPU 上执行数据并行。请从 `loss.backward()` 开始，写出上层框架调用 NCCL 并最终使用通信结果的完整外部流程。至少包含：

```text
训练框架 / DDP
本地梯度
NCCL集合通信API
GPU间通信
聚合后的梯度
参数更新
```

然后回答：以下工作哪些由 NCCL 负责，哪些不由 NCCL 负责？说明实际负责者。

1. 启动四个训练进程；
2. 决定在反向传播的什么位置同步梯度；
3. 根据 GPU 和网络拓扑选择通信路径；
4. 把不同 GPU 的数组执行 AllReduce；
5. 执行模型前向传播；
6. 启动通信 Kernel 并推进网络传输。

### 作答区



---

## 第 2 题：进程、GPU、rank 与 communicator（8 分）

根据统一场景回答：

1. P0 和 P2 能否拥有相同的操作系统进程 ID？rank 是否等于进程 ID？
2. Node A 的 GPU 0 和 Node B 的 GPU 0 为什么不会导致 rank 冲突？
3. communicator X 中能否同时存在两个 rank 0？为什么？
4. 如果另外创建 communicator Y，它能否也拥有 rank 0、1、2、3？
5. 四个进程里的 `ncclComm_t comm` 变量是否必须保存完全相同的指针值？它们共同表示的是什么？

回答时必须明确 rank 的有效范围和唯一性范围。

### 作答区



---

## 第 3 题：communicator 初始化因果链（10 分）

将下列步骤按实际依赖关系重新排列，并用箭头连接：

```text
A. 每个进程调用 ncclCommInitRank(&comm, 4, U, rank)
B. 建立 Channel、Ring 拓扑和 send/recv transport 连接
C. rank 0 调用 ncclGetUniqueId(&U)
D. 上层运行时确定各进程的 global rank、local rank 和 nranks
E. bootstrap 让参与者建立初始化阶段的联系
F. 训练框架、MPI 或其他外部机制把 U 分发给其余进程
G. 各 rank 交换 peerInfo 等初始化信息
H. communicator 可以用于调用 ncclAllReduce()
I. 每个进程选择自己控制的 CUDA device
```

然后回答：

1. `UniqueId` 标识的是某个 GPU、某台机器、某个 rank，还是一次 communicator 建立过程的共同会合身份？
2. `UniqueId` 是否就是最后得到的 `ncclComm*` 指针？
3. `peerInfo` 为什么不能在 rank、device 和进程信息都未确定时凭空产生？

### 作答区



---

## 第 4 题：一次 AllReduce 的调用契约（8 分）

四个 rank 上分别有以下 GPU 数组：

```text
rank 0: [1, 2]
rank 1: [2, 3]
rank 2: [3, 4]
rank 3: [4, 5]
```

每个进程都调用：

```c
ncclAllReduce(sendbuff, recvbuff, 2, ncclFloat, ncclSum, comm, stream);
```

回答：

1. 最初是谁调用这条 Host API？四个 rank 中是否只需要 rank 0 调用？
2. 操作完成后，每个 rank 的 `recvbuff` 是什么？
3. 如果训练程序需要平均梯度，还需要做什么？
4. `ncclAllReduce()` 返回是否等价于 GPU 通信完成？
5. `comm` 和 `stream` 两个参数分别回答了什么问题？

### 作答区



---

## 第 5 题：task、devWork 与 plan 的实际先后关系（12 分）

请根据当前本地 NCCL 源码，把下面的对象和函数排列成完整流程：

```text
ncclTaskColl
ncclTasksRegAndEnqueue()
struct ncclDevWorkColl devWork = {}
planner->collWorkQueue
ncclLaunchPrepare()
分配 ncclKernelPlan
scheduleCollTasksToPlan()
plan->workQueue
finishPlan()
uploadWork()
GPU Kernel
```

回答：

1. 从运行时对象的创建顺序看，`ncclDevWorkColl` 和 `ncclKernelPlan` 哪个先创建？
2. 为什么仍然可以说 plan “组织” devWork？
3. `ncclTaskColl`、`ncclDevWorkColl` 和 `ncclKernelPlan` 分别服务于谁、解决什么问题？
4. 为什么 plan 中不能只有 `ncclDevWorkColl`，还要保留 task、Proxy operation 和清理信息？

必须区分“对象创建顺序”和“对象组织关系”。

### 作答区



---

## 第 6 题：结构体、指针、句柄和容器（8 分）

阅读以下代码：

```c
struct ncclTaskColl* task;
struct ncclDevWorkColl devWork = {};
struct ncclWorkList* workNode;
struct ncclDevWorkColl* workPtr =
    (struct ncclDevWorkColl*)(workNode + 1);
struct ncclKernelPlan* plan;
ncclComm_t comm;
```

回答：

1. 哪些变量保存结构体对象，哪些变量保存指针？
2. `workNode + 1` 在这里为什么可以指向紧跟在 `ncclWorkList` 后面的 payload？
3. `workPtr` 和局部变量 `devWork` 是否一定是同一个内存对象？
4. `ncclComm_t comm` 为什么称为句柄？调用者通过它间接访问什么？
5. 在内存分配函数中同时传入 `&comm->memPool_ncclTaskColl` 和 `&comm->memPermanent`，是否表示两个成员是递进包含关系？正确理解是什么？

### 作答区



---

## 第 7 题：CPU 线程、GPU 线程与函数调用（8 分）

源码中有：

```c
state->thread = std::thread(ncclProxyProgress, proxyState);
```

回答：

1. `state->thread`、`ncclProxyProgress`、`proxyState` 分别是什么？
2. 直接调用 `ncclProxyProgress(proxyState)` 与创建新线程执行它有什么不同？
3. 同一进程中的用户调用线程和 Proxy Progress 线程共享哪些资源，又各自独立拥有什么？
4. 将以下函数按执行者分类：

```text
ncclAllReduce()
collTaskAppend()
ncclLaunchPrepare()
ncclKernelMain()
runRing()
ncclProxyProgress()
sendProxyProgress()
ncclNet->isend()
```

分类为：用户 Host 调用线程、GPU 线程、NCCL Proxy CPU 线程。

### 作答区



---

## 第 8 题：一次 plan 为什么有两条执行路线（8 分）

请补全并解释下图：

```text
ncclKernelPlan
├── plan->workQueue
│       ↓
│   ____________
│       ↓
│   ncclKernelMain()
│       ↓
│   runRing()
│
└── plan->proxyOpQueue
        ↓
    ____________
        ↓
    sendProxyProgress()
        ↓
    ncclNet->isend()/test()
```

回答：

1. 两个空格分别是什么执行者或入口？
2. 两条路线在哪里交换数据和状态？
3. 如果只有 GPU work 而没有需要的 Proxy work，跨机器 SIMPLE 网络发送为什么无法完成？

### 作答区



---

## 第 9 题：Channel 与 Ring AllReduce（8 分）

已知 Channel 0 的 Ring 顺序为：

```text
0 → 1 → 2 → 3 → 0
```

回答：

1. 对 rank 1 而言，`ring->prev` 和 `ring->next` 分别是谁？对 rank 3 呢？
2. Ring AllReduce 为什么可以分成 ReduceScatter 和 AllGather 两个阶段？每个阶段结束时，各 rank 分别拥有什么？
3. `Channel` 是 rank、线程、网络连接还是一条并行的数据传输通道？它内部为什么需要保存 Ring 拓扑和 peer 连接？
4. 使用多个 Channel 时，是把同一整块数据重复计算多次，还是把数据划分后并行处理？

不要求写每一轮的精确 chunk 下标，但必须讲清楚数据所有权的变化。

### 作答区



---

## 第 10 题：从 `runRing()` 到发送连接（8 分）

rank 1 在 Channel 0 上执行一次发送，忽略特殊注册路径。请解释：

```text
runRing(work)
  → 构造 Primitives
  → directSend(...)
  → loadSendConn(...)
  → conn = &channel.peers[peer]->send[0]
```

回答：

1. `work` 指向什么类型的对象，主要给 GPU 提供什么？
2. `Primitives<...> prims` 是任务、线程、队列，还是当前 GPU 线程使用的局部通信状态对象？
3. 此时的 `peer` 应当来自哪里？
4. `struct ncclConnInfo* conn` 是网络连接本身，还是连接描述结构体的指针？
5. `directSend()` 是否必然表示数据绕过 NCCL 缓冲区直接进入网卡？为什么？

### 作答区



---

## 第 11 题：GPU—Proxy—网络状态诊断（8 分）

一个发送 step 使用循环缓冲区槽位：

```c
slot = step % NCCL_STEPS;
```

`tail` 表示 GPU 已发布到哪里，`head` 表示 Proxy/网络已完成并释放到哪里。Proxy 还有：

```text
posted：已经允许或准备进入发送流水线的step数
transmitted：已经向网络接口提交的step数
done：网络已经确认完成的step数
```

请分别判断下列状态说明通信大致进行到哪里，下一步应由谁推进：

| 情况 | 已知状态 |
|---|---|
| A | `posted = 1`，`tail` 未推进，`transmitted = 0`，`done = 0` |
| B | `tail` 已推进，`connFifo[slot].size` 有效，`transmitted = 0` |
| C | `transmitted = 1`，`done = 0` |
| D | `done = 1`，`connFifo[slot].size = -1`，`head` 已推进 |

每种情况说明：

1. GPU 是否已经生产并发布数据；
2. Proxy 是否已经提交网络请求；
3. 网络是否已经完成；
4. 缓冲区槽位是否可以安全复用。

### 作答区



---

## 第 12 题：面向 Mycroft 复现目标的综合判断（6 分）

现在要分别构造三种性能异常：

```text
异常甲：GPU Kernel迟迟没有生产出待发送chunk
异常乙：GPU已经发布chunk，但Proxy线程长时间没有提交网络请求
异常丙：网络请求已经提交，但长时间没有完成
```

请为每种异常回答：

1. 最适合观察哪一组状态边界；
2. `tail`、`transmitted`、`done`、`head` 大致会停在哪里；
3. 异常更接近 GPU 执行、Proxy 调度还是网络传输；
4. 为什么仅观察 `ncclAllReduce()` 的进入和返回不足以区分这三种异常？

这里不要求给出最终插桩代码，只要求建立可验证的因果关系。

### 作答区



---

## 评分标准

| 能力 | 对应题目 | 分值 |
|---|---|---:|
| 理解 NCCL 的用途和调用场景 | 1、4 | 16 |
| 理解 communicator、rank 和初始化 | 2、3 | 18 |
| 理解 Host 对象及其真实先后关系 | 5、6 | 20 |
| 理解 CPU/GPU/Proxy 执行关系 | 7、8 | 16 |
| 理解 Ring、Channel 和连接路径 | 9、10 | 16 |
| 能够根据状态定位执行阶段 | 11、12 | 14 |
| 总分 |  | 100 |

评价重点：

- 是否说明了对象从哪里来、由谁使用；
- 是否区分创建顺序、包含关系和执行顺序；
- 是否区分入队、开始、提交和完成；
- 是否能沿状态变化定位 GPU、Proxy 或网络阶段；
- 不要求复述源码原句，但因果关系必须成立。

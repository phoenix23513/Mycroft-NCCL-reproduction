# 阶段04：接收端 Proxy 到 GPU 规约与转发

> 保留的历史学习资料，不定义当前任务或验收。执行范围与进度以[核心复现计划](../plans/Mycroft_26日开发路线图.md)为准；源码解释按本文标注的版本阅读，候选观测点仍需真实运行验证。

本文承接[阶段03：GPU—Proxy—NET 执行与 Mycroft 进度观测](阶段03_GPU_Proxy_NET执行与Mycroft进度观测.md)，把其中接收端的概览展开到实际代码：网络如何把数据写进接收 buffer，Proxy 如何通知 GPU，GPU 如何规约并继续发送，以及接收 slot 如何释放。

源码基线：仓库 `third_party/nccl` 中的 NCCL `v2.21.5-1`。主场景为 Ring AllReduce、NET Transport、SIMPLE 协议、普通连接 buffer，使用 `sub->reg == 0` 路径。共享 buffer 和注册用户 buffer 的差异在相关步骤中说明。

## 1. 本阶段在整体流程中的位置

前面已经建立从应用到执行的路径：

```text
应用调用 ncclAllReduce()
  -> Host 创建和组织 collective task
  -> 准备 GPU work 与 kernel plan
  -> 启动 CUDA kernel
  -> Channel 对应的 GPU block 执行 runRing()
  -> Primitives 读写连接 buffer
  -> 发送 Proxy 调用 isend()
```

本阶段沿着数据继续追踪：

```text
前驱 rank 的发送 Proxy
  -> 网络传输
  -> 当前 rank 的接收 buffer
  -> 接收 Proxy 发布 recvTail
  -> 当前 GPU 退出 waitPeer()
  -> GPU 读取、规约、写下一跳发送 buffer
  -> GPU 发布接收 head 和发送 tail
  -> 接收 Proxy 回收 slot；发送 Proxy 继续向后继发送
```

整个过程存在两种同步：网络异步请求用 `test()` 检查；GPU 与 Proxy 通过连接中的 head/tail 指针交换进度。

## 2. 用一个 slice 串起全过程

设 Ring 顺序为：

```text
rank 0 -> rank 1 -> rank 2 -> rank 3 -> rank 0
```

当前跟踪某个 chunk 内的一个 slice，Sum 规约的数据为：

```text
rank 0 发来的数据：             [ 1,  2,  3,  4]
rank 1 相同位置的本地输入：     [10, 20, 30, 40]
rank 1 规约后送往 rank 2：      [11, 22, 33, 44]
```

这只是该 chunk 的一段中间规约；rank 2、rank 3 的贡献还需要加入。`runRing()` 决定当前处理哪个 chunk；一个 chunk 内的 slice 则由 primitive 按协议参数继续处理。

```text
rank 0 GPU             rank 1 接收 Proxy       rank 1 GPU          rank 1 发送 Proxy
    |                         |                    |                      |
写发送 buffer                 |                    |                      |
更新发送 tail                 |                    |                      |
    |                         |                    |                      |
sendProxy -> isend() ------ 网络 ------> 接收 buffer |                      |
                              |                    |                      |
                         test：接收完成             |                      |
                         必要时 flush               |                      |
                         发布 recvTail ------------> waitPeer 退出         |
                                                   |                      |
                                              读接收数据                   |
                                              与本地输入相加               |
                                              写发送 buffer                |
                                                   |                      |
                         读取接收 head <------ 发布接收 head               |
                                              发布发送 tail ------------->|
                                                                       isend()
                                                                         |
                                                                       rank 2
```

接收 Proxy 可以预先调用 `irecv()`，因此图中的准备接收和远端准备发送可能并行发生。

## 3. 函数、结构体和指针关系

接收端函数位于 [`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc)，搜索：

```cpp
static ncclResult_t recvProxyProgress(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args)
```

| 名字 | C 风格解释 | 在本流程中解决的问题 |
|---|---|---|
| `recvProxyProgress` | CPU 函数 | 推进接收请求、数据可见性和 slot 回收 |
| `proxyState` | 指向 `ncclProxyState` 的指针 | 提供 Proxy 状态和 NET 插件函数表 |
| `args` | 指向 `ncclProxyArgs` 的指针 | 保存本次 Proxy 工作及其进度 |
| `args->subs` | 结构体内嵌的 `ncclProxySubArgs` 数组 | 保存各条连接的子工作 |
| `sub` | 指向数组中一个元素的指针 | 读取和更新某条接收工作的计数器 |
| `sub->connection` | 指向 Proxy 连接结构体的指针 | 找到 transport 和连接资源 |
| `resources` | 指向 `recvNetResources` 的指针 | 获取接收 buffer、NET 连接、head/tail 等 |
| `subGroup` | 指向一组连续 sub 的第一个元素的指针 | 组织插件支持的批量接收 |

结构关系：

```text
struct ncclProxyArgs
  |-- state / protocol / sliceSteps / nsubs / done
  |-- progress：函数指针
  \-- subs[]：内嵌数组
        \-- struct ncclProxySubArgs
              |-- channelId / base / nsteps
              |-- posted / received / transmitted / done
              \-- connection：指针
                    \-- transportResources：指针
                          \-- struct recvNetResources
                                |-- netRecvComm
                                |-- buffer 映射
                                |-- recvMem -> tail
                                \-- sendMem -> head
```

`args` 是一个 Proxy 工作容器。一次 AllReduce 可以关联多个发送/接收 Proxy 工作；不能把一个 `args` 直接当成整次 AllReduce。

代码中的：

```cpp
struct ncclProxySubArgs* sub = args->subs + s;
```

等价于 C 写法：

```c
struct ncclProxySubArgs* sub = &args->subs[s];
```

它取得已有数组元素的指针，没有创建新的 sub。

## 4. 谁调用 `recvProxyProgress()`

[`src/proxy.cc`](../../third_party/nccl/src/proxy.cc) 中的 `progressOps()` 遍历活跃工作并执行：

```cpp
NCCLCHECK(op->progress(proxyState, op));
```

`op->progress` 是函数指针，在 Proxy 工作转换时从对应 transport 的 `proxyProgress` 取得。NET 的接收 transport 注册 `recvProxyProgress`，因此这次间接调用进入接收端推进函数。

```text
Proxy progress CPU 线程
  -> progressOps()
  -> op->progress(proxyState, op)
  -> recvProxyProgress(proxyState, args)
```

它反复检查当前条件，每次有进展就更新状态。代码在各阶段之后有：

```cpp
if (args->idle == 0) return ncclSuccess;
```

`idle == 0` 表示这次有进展。这里的成功返回表示本轮推进调用成功，工作是否结束要查看 `args->state`。后续调用会继续推进未完成部分。

## 5. 初始化：建立本次工作的 step 范围

在 `args->state == ncclProxyOpReady` 分支中，核心代码为：

```cpp
sub->base = ROUNDUP(resources->step, args->chunkSteps);
resources->step = sub->base + sub->nsteps;
sub->posted = sub->received = sub->transmitted = sub->done = 0;
```

字段之间的关系：

| 字段 | 含义 |
|---|---|
| `resources->step` | 这条长期连接分配工作时使用的 step 位置 |
| `sub->base` | 本次子工作的起始 step，按 `chunkSteps` 对齐 |
| `sub->nsteps` | 本次子工作需要处理的协议 step 数 |
| 四个进度计数器 | 相对于本次 `base` 的累计进度 |

例如，连接旧 step 为 9，`chunkSteps=2`，`nsteps=6`：

```text
base                    = 10
本次step范围             = [10, 16)
resources->step更新为    = 16
本次四个相对计数器       = 0
```

`resources->step` 被推进是为后续工作保留序号，不能由此判断本次数据已经传完。

初始化还会把共享同一个 `netRecvComm` 的 sub 调整到一起，受插件支持的 `maxRecvs` 限制。一组 sub 可以通过一次 `irecv()` 提交多个 buffer；这与应用层 communicator 分组是不同层次。

## 6. 四个接收计数器对应四道边界

| 接收侧字段 | 在哪一步增加 | 代表已推进到哪里 |
|---|---|---|
| `posted` | `irecv()` 返回非空 request | 已提交接收请求 |
| `received` | 接收 request 的 `test()` 返回完成 | 网络接收已完成 |
| `transmitted` | 必要的 flush 完成后 | Proxy 已推进向 GPU 发布数据的阶段 |
| `done` | 普通 buffer 路径读到 GPU head 前进后 | GPU 已消费相应接收 slot |

主场景中的进度关系为：

```text
0 <= done <= transmitted <= received <= posted <= nsteps
```

这些数以协议 step 为单位。代码通常每次增加 `args->sliceSteps`，而不是固定加一；slice、step 和 Ring chunk 的单位要按执行参数区分。

同一个字段名在发送侧和接收侧解决的问题不同：

| 字段 | 发送侧 | 接收侧主场景 |
|---|---|---|
| `transmitted` | 有效发送 request 已提交 | 接收数据经过可见性处理，进入向 GPU 发布阶段 |
| `done` | 本地发送 request 完成 | GPU 已消费接收 slot |

## 7. 提交接收：先选 buffer，再调用 `irecv()`

### 7.1 选择 slot

对非共享 buffer，关键代码为：

```cpp
int stepSize = resources->buffSizes[p] / NCCL_STEPS;
char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
int buffSlot = (sub->base + sub->posted) % NCCL_STEPS;

ptrs[subCount] = localBuff + buffSlot * stepSize;
sizes[subCount] = stepSize * args->sliceSteps;
```

`localBuff` 是 Proxy 使用的 buffer 指针视图；目标存储可以是映射的 host buffer，也可以是 GDR 路径中的 GPU buffer。`cpu` 参数说明取得哪一侧的指针视图，不能仅凭它判断物理存储在 CPU 内存中。

若 `NCCL_STEPS=8`，`base=10`，`posted=2`：

```text
本次起始step = 12
buffSlot     = 12 % 8 = 4
目标指针      = localBuff + 4 * stepSize
```

普通共享 buffer 路径通过 `sharedBuffersGet()` 取得实际 offset，并写入 `connFifo[buffSlot].offset`。GPU 会据此读取相同数据；这时逻辑 slot 的编号和 buffer 的实际偏移通过 FIFO 元数据关联。

### 7.2 限制尚未消费的深度

```cpp
if (sub->posted >= sub->done + maxDepth) {
    subCount = 0;
    break;
}
```

`posted - done` 是已提交但尚未被 GPU 消费确认的 step 深度。达到窗口上限后停止继续提交，避免新接收覆盖仍在使用的存储空间。

### 7.3 调用 NET 插件

```cpp
void** requestPtr = subGroup->requests + (step % NCCL_STEPS);

NCCLCHECK(proxyState->ncclNet->irecv(
    resources->netRecvComm,
    subCount,
    ptrs,
    sizes,
    tags,
    mhandles,
    requestPtr));
```

| 参数 | 作用 |
|---|---|
| `netRecvComm` | 已建立的 NET 接收连接 |
| `subCount` | 这次批量接收的 buffer 数 |
| `ptrs` | 接收目标 buffer 指针数组 |
| `sizes` | 接收容量数组，单位为字节 |
| `tags` | 插件配对消息时使用的标记，本处来自 `tpRemoteRank` |
| `mhandles` | 对应 buffer 的内存注册句柄数组 |
| `requestPtr` | 指向请求指针存储位置的指针，插件将异步请求写到这里 |

`requestPtr` 的类型是 `void**`，因为插件要修改的是一个 `void*` 请求指针。

只有得到非空 request 后，源码才推进：

```cpp
if (*requestPtr) {
    sub->posted += args->sliceSteps;
}
```

此时 `posted > received`：接收已提交，数据到达仍需后续检查。

## 8. 网络接收完成：推进 `received`

```cpp
if (subGroup->posted > subGroup->received) {
    uint64_t step = subGroup->received;
    int done;

    NCCLCHECK(proxyState->ncclNet->test(
        subGroup->requests[step % NCCL_STEPS], &done, sizes));

    if (done) {
        // 处理实际收到的字节数、注册buffer等分支。
        sub->received += args->sliceSteps;
    }
}
```

这里省略了分组循环等代码，保留控制条件和进度变化。局部变量 `int done` 是插件本次返回的完成标志；`sub->done` 则是接收子工作的累计消费进度，两者类型和用途不同。

`test()` 确认接收 request 完成后，`received` 前进；普通 SIMPLE 路径还要经过可见性处理和 tail 发布，GPU 才开始读取这一片段。

## 9. 可见性处理：必要时 flush

网络设备直接写 GPU 显存时，NCCL 根据连接配置判断是否需要 flush：

```cpp
if (resources->useGdr) {
    needFlush |= resources->needFlush;
}
```

在数据量大于零、SIMPLE 协议且 `needFlush` 成立时，代码选择：

- `gdcFlush` 路径：执行一次特定读取，完成所需的可见性处理。
- 普通 flush 路径：调用插件的 `iflush()`，必要时保存异步 flush request。

后续阶段检查：

```cpp
int done = 1;
void* request = subGroup->requests[step % NCCL_STEPS];
if (request) {
    NCCLCHECK(proxyState->ncclNet->test(request, &done, NULL));
}
```

没有异步 flush request 时可直接继续；存在 request 时等待它完成。

```text
received前进
  -> 必要的flush完成
  -> 发布tail
  -> GPU读取数据
```

## 10. 发布给 GPU：推进 `transmitted`，写 tail

主场景的源码顺序为：

```cpp
sub->transmitted += args->sliceSteps;

__sync_synchronize();
struct recvNetResources* resources =
    (struct recvNetResources*)sub->connection->transportResources;

volatile uint64_t* recvTail = resources->gdcSync
    ? resources->gdcSync
    : &resources->recvMem->tail;

*recvTail = sub->base + sub->transmitted;
```

它先推进接收侧 `transmitted`，再执行同步并写 tail。tail 使用绝对 step，计数器使用相对 step。

例如：

```text
base = 10，sliceSteps = 1
transmitted从0增加到1
recvTail被写为11
```

GPU 根据 tail 判定当前 slice 已准备好。Proxy 在 `transmitted` 增加和 tail 写入之间存在很短的执行区间；在该位置采样时，应明确采集的是哪个动作之前或之后的状态。

## 11. GPU 为什么在读同一个 tail

接收连接建立时，[`net.cc`](../../third_party/nccl/src/transport/net.cc) 中 `recvConnect()` 绑定：

```cpp
recv->conn.head = &sendMem->head;
recv->conn.tail = gdcMem ? (uint64_t*)gdcMem : &recvMem->tail;
```

GPU primitive 的 [`loadRecvConn()`](../../third_party/nccl/src/device/prims_simple.h) 则为不同线程角色绑定相应指针：

```cpp
if (flags & RolePostRecv) {
    connStepPtr = conn->head;
}

if (flags & RoleWaitRecv) {
    connStepPtr = conn->tail;
}
```

因此：

```text
接收Proxy写 resources->recvMem->tail
                  |
                  +-- GPU接收conn.tail指向同一进度位置
                          |
                          +-- WaitRecv角色通过connStepPtr读取
```

GPU kernel 和 CPU Proxy 可以同时运行。它们通过共享进度指针协作，Proxy 写 tail 后 GPU 的等待循环观察到变化。

## 12. `recvReduceSend()`：同时等数据和发送空间

Ring 中间规约阶段在 [`all_reduce.h`](../../third_party/nccl/src/device/all_reduce.h) 调用：

```cpp
prims.recvReduceSend(offset, nelem);
```

`prims` 是 GPU primitive 对象；`recvReduceSend` 是该对象的成员函数。SIMPLE 版本定义为：

```cpp
__device__ __forceinline__ void recvReduceSend(
    intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
}
```

模板参数按 `genericOp` 定义顺序解释：

| 参数 | 值 | 本次操作要求 |
|---|---|---|
| `DirectRecv1` | 0 | 使用普通接收方式 |
| `DirectSend1` | 0 | 使用普通发送方式 |
| `Recv` | 1 | 接收前驱数据 |
| `Send` | 1 | 向后继发送 |
| `SrcBuf` | `Input` | 读取本 rank 输入作为另一个规约来源 |
| `DstBuf` | -1 | 此步骤没有本 rank 用户输出目标 |

`genericOp()` 首先安排本地输入指针，再调用 `waitPeer()`。等待角色分为接收和发送两个方向：

```text
WaitRecv角色：读接收连接tail，等前驱数据ready
WaitSend角色：读发送连接head，等下一跳发送buffer有空间
```

核心等待条件：

```cpp
while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0)
       < step + StepPerSlice) {
    connStepCache = loadStepValue(connStepPtr);
    if (checkAbort(spins)) break;
}
```

拆成便于理解的两种条件：

```text
接收：tail >= step + StepPerSlice
发送：head + NCCL_STEPS >= step + StepPerSlice
```

接收条件保证读到有效数据；发送条件保证写入的空间可以使用。即使前驱数据已到达，下游发送空间不足也会阻止这一轮规约/转发继续执行。

## 13. GPU 规约：两个输入，一个发送目标

`genericOp()` 中的主要次序为：

```text
设置本地输入/输出指针
  -> waitPeer()：等待连接条件并设置连接buffer指针
  -> subBarrier()：工作线程同步
  -> reduceCopy()：复制或规约
  -> barrier()：同步数据处理完成
  -> postPeer()：发布进度
```

对于本例的 `recvReduceSend()`：

```text
srcs[]：本地输入指针 + 前驱接收数据指针
dsts[]：后继连接的发送buffer指针
```

`reduceCopy()` 的来源数量和目标数量由模板参数、连接数量计算。Ring 单前驱、单后继的当前操作是两个来源、一个目标。

以下是 C 风格的数据语义示意，真实代码由多条 GPU 线程执行并带有协议同步：

```c
for (int i = 0; i < count; ++i) {
    send_slot[i] = recv_slot[i] + local_input[offset + i];
}
```

数据变化：

```text
recv_slot：    [ 1,  2,  3,  4]
local_input：  [10, 20, 30, 40]
send_slot：    [11, 22, 33, 44]
```

这里描述 `ncclSum`。实际 `RedOp` 决定规约运算；在用户最终结果生成位置还可能执行相应的 post-operation 处理。

## 14. `postPeer()`：归还接收空间，发布发送数据

数据处理完成并经过线程同步后调用：

```cpp
postPeer<Recv, Send>(0 < sliceSize);
```

核心代码为：

```cpp
if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
    step += StepPerSlice;
    if (Send && (flags & RolePostSend)
        && (dataStored || (flags & ConnFifoEnabled))) {
        fence_acq_rel_sys();
    }
    st_relaxed_sys_global(connStepPtr, step);
}
```

接收和发送角色通过各自的指针发布状态：

| GPU 角色 | `connStepPtr` 指向 | 写入含义 | 后续观察者 |
|---|---|---|---|
| `RolePostRecv` | 接收 connection 的 head | 接收 slice 已消费，空间可以复用 | 接收 Proxy |
| `RolePostSend` | 发送 connection 的 tail | 发送 slice 已准备好 | 发送 Proxy |

这些角色由不同 GPU 线程承担，各自维护对应连接的 step。`waitPeer()` 与 `postPeer()` 中都出现 `step += ...`，要结合执行线程角色理解，不能把它们当成同一个线程重复增加同一个计数器。

```text
前驱 -> 接收buffer -> GPU规约 -> 发送buffer -> 后继
           |                         |
      GPU发布head                GPU发布tail
           |                         |
      recvProxy回收              sendProxy提交isend
```

## 15. 接收 Proxy 读取 head，推进 `done`

普通 buffer 路径读取：

```cpp
volatile uint64_t* sendHead = &resources->sendMem->head;
uint64_t done = *sendHead;
```

局部变量名 `sendHead` 来自内存结构命名。前面的连接建立代码已经说明：这个位置绑定到本接收 connection 的 `conn.head`，由 GPU 的 `RolePostRecv` 更新。

源码按 head 回收已消费的进度，并限制不超过已发布的 `transmitted`：

```cpp
while (done > sub->base + sub->done &&
       sub->transmitted > sub->done) {
    // 插件支持时调用 irecvConsumed()，通知接收buffer已消费。
    sub->done += args->sliceSteps;
}
```

例如：

```text
base=10
transmitted=1
GPU head=11
sub->done从0推进到1
```

`posted - done` 随之下降，接收窗口释放容量，后续 `irecv()` 可以继续提交。

## 16. 一个 slice 的状态推演

设普通 buffer 路径中 `base=10`、`sliceSteps=1`、`nsteps=1`，已经完成初始化对齐，head/tail 初值均为 10。

| 时刻 | posted | received | transmitted | done | 接收 tail | 接收 head | 意义 |
|---|---:|---:|---:|---:|---:|---:|---|
| 初始化完成 | 0 | 0 | 0 | 0 | 10 | 10 | 本 slice 尚未提交 |
| `irecv` 返回有效 request | 1 | 0 | 0 | 0 | 10 | 10 | 网络接收已提交 |
| 接收 `test` 完成 | 1 | 1 | 0 | 0 | 10 | 10 | 数据已到达 |
| 可见性处理和 tail 发布完成 | 1 | 1 | 1 | 0 | 11 | 10 | GPU 可以读取 |
| GPU 完成处理并发布 head | 1 | 1 | 1 | 0 | 11 | 11 | GPU 已消费，Proxy 尚未确认 |
| Proxy 观察到 head | 1 | 1 | 1 | 1 | 11 | 11 | 接收子工作完成 |

表格记录各动作结束后的状态。真实执行允许多个 slice 同时处于不同阶段；计数器推进也可能跨越多个协议 step。

## 17. Proxy 工作结束如何判断

一个 sub 的判断：

```cpp
if (sub->done == sub->nsteps) {
    args->done++;
}
```

所有 sub 完成后的判断：

```cpp
if (args->done == args->nsubs) {
    args->state = ncclProxyOpNone;
}
```

这里有两个层次：

```text
sub->done：一个连接子工作的step进度
args->done：已经完成的sub数量
```

`args->state == ncclProxyOpNone` 说明这组 Proxy 工作结束。整次 AllReduce 的 GPU work、其他 Channel 和其他 Proxy 工作完成关系，需要继续沿 collective 生命周期追踪。

## 18. 两种 buffer 路径的差异

### 18.1 普通共享 buffer

主线中的 head/tail 协作仍存在，但 buffer 指针通过共享池和 `connFifo.offset` 关联：

```text
recvProxy选择共享buffer offset
  -> 写connFifo[slot].offset
  -> irecv写入对应位置
  -> 发布tail
  -> GPU依据offset取得接收指针
```

逻辑 slot 是循环元数据索引，实际数据位置由 offset 指定。

### 18.2 注册用户 buffer：`sub->reg != 0`

该分支可以直接使用注册的用户 buffer。代码还包含以下专门处理：

- 某些共享 SIMPLE 路径先等待 GPU 通过 FIFO 表示 kernel 已启动，再访问用户 buffer。
- 如果一次网络接收不足以处理全部字节，会调整 `recvbuff`、`nbytes` 和 `nsteps`。
- NET 可以拆成多个 step，但 GPU 侧只接受一次对应的发布；tail 在相应全部网络步骤完成后推进。
- 最后的消费确认使用 `sub->base + sub->nsteps`，而不是普通路径的 `*sendHead`。

因此，“接收 `sub->done` 增加证明 GPU 已读完普通 slot”适用于本文主场景；注册用户 buffer 分支必须按其专门语义解释。

## 19. 用接收状态分析停滞

| 观察到的持续差距 | 本地已知事实 | 继续检查的依赖 |
|---|---|---|
| `posted > received` | 请求已提交，网络接收尚未完成 | 前驱是否产生数据并提交发送、网络和连接是否推进 |
| `received > transmitted` | 接收已完成，尚未完成发布阶段 | flush request、可见性处理、Proxy 执行状态 |
| `transmitted > done` | 数据已发布，普通路径的 GPU 消费确认未完成 | GPU 的接收等待、下一跳发送空间、GPU 调度和处理状态 |
| `posted - done` 达到窗口深度 | 接收窗口已满 | GPU 消费为什么没有推进，以及下游是否阻塞 |

一个具体传播链：

```text
rank 1 向 rank 2 的发送空间不足
  -> rank 1 recvReduceSend 的 WaitSend 等待
  -> rank 1 暂时不消费已经收到的 slice
  -> rank 1 接收 head 不前进
  -> rank 1 接收 Proxy 的 done 不前进
  -> 接收窗口逐渐填满
  -> 对 rank 0 的后续传输形成反压
```

Mycroft 需要这些依赖来区分“在哪个位置看到停滞”和“哪个依赖最先停止推进”。单看 rank 1 的接收 `done` 停止，不能直接确定根因在 rank 1 GPU；应结合下游发送侧和对端状态继续追踪。

## 20. 源码定位索引

以下行号对应当前 `v2.21.5-1` 基线；文件变化后优先使用搜索词。

| 内容 | 文件 | 行号 | 搜索词 |
|---|---|---:|---|
| sub 和 args 类型 | [`src/include/proxy.h`](../../third_party/nccl/src/include/proxy.h) | 62、92 | `struct ncclProxySubArgs`、`struct ncclProxyArgs` |
| Proxy 工作函数绑定 | [`src/proxy.cc`](../../third_party/nccl/src/proxy.cc) | 398 | `args->progress =` |
| Proxy 间接调用 | [`src/proxy.cc`](../../third_party/nccl/src/proxy.cc) | 692、698 | `progressOps`、`op->progress` |
| 接收 head/tail 绑定 | [`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc) | 413、417 | `recv->conn.head`、`recv->conn.tail` |
| 接收推进函数 | [`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc) | 1184 | `static ncclResult_t recvProxyProgress` |
| base 和计数器初始化 | [`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc) | 1214、1217 | `sub->base = ROUNDUP` |
| buffer 和接收提交 | [`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc) | 1243、1278 | `sub->posted >=`、`ncclNet->irecv` |
| 接收完成和 flush | [`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc) | 1302、1363 | `ncclNet->test`、`ncclNet->iflush` |
| 推进 transmitted 并发布 tail | [`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc) | 1383、1393 | `sub->transmitted +=`、`*recvTail =` |
| GPU 消费确认 | [`src/transport/net.cc`](../../third_party/nccl/src/transport/net.cc) | 1410、1421 | `sendHead`、`sub->done +=` |
| GPU 接收指针绑定 | [`src/device/prims_simple.h`](../../third_party/nccl/src/device/prims_simple.h) | 439、452、458 | `loadRecvConn`、`connStepPtr =` |
| GPU 发送指针绑定 | [`src/device/prims_simple.h`](../../third_party/nccl/src/device/prims_simple.h) | 494、501、507 | `loadSendConn` |
| 等待和进度发布 | [`src/device/prims_simple.h`](../../third_party/nccl/src/device/prims_simple.h) | 123、175 | `waitPeer`、`postPeer` |
| GPU 数据处理 | [`src/device/prims_simple.h`](../../third_party/nccl/src/device/prims_simple.h) | 186、238、271、280 | `genericOp`、`reduceCopy` |
| 规约转发 primitive | [`src/device/prims_simple.h`](../../third_party/nccl/src/device/prims_simple.h) | 770 | `void recvReduceSend` |
| Ring 算法调用 | [`src/device/all_reduce.h`](../../third_party/nccl/src/device/all_reduce.h) | 53 | `prims.recvReduceSend` |

## 21. 当前学习边界与后续衔接

本阶段讲解已覆盖单个 slice 的接收提交、网络完成、可见性处理、tail 发布、GPU 规约转发和 head 回收，并将这些动作对应到源码。

接下来沿 [`runRing()`](../../third_party/nccl/src/device/all_reduce.h) 的五段 primitive，使用具体的 4-rank、4-chunk 向量推演完整算法：

```text
send
  -> recvReduceSend
  -> directRecvReduceCopySend
  -> directRecvCopySend
  -> directRecv
```

完整 ReduceScatter/AllGather 数值推演、多 Channel 完成关系、整次 collective 完成判定尚待后续讲解。本笔记记录源码解释与推演；当前没有新增动态运行或插桩验证结果。

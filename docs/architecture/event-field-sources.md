# Event 字段在 NCCL 2.21.5 中的来源

本文把 Mycroft 论文中的观测字段、项目版本化 Event 契约和固定版本 NCCL 源码连接起来。源码基线是官方 tag `v2.21.5-1`，commit `ab2b89c4c339bd7f816fbc114a4b05d386b66290`。

状态含义：

- **已确认**：字段或状态转换能从固定版本源码直接证明；
- **候选**：源码位置明确，但仍需 Day 16—20 的真实 NCCL 动态实验确认跨 rank 一致性或运行时语义；
- **未映射**：论文没有给出足够定义，不能靠同名字段猜测。

## 三种身份不要混在一起

一次通信同时存在三类身份：

1. 逻辑身份：哪个 communicator、哪个 rank、哪个 collective operation；
2. 本机硬件身份：当前进程使用哪张 GPU、哪个 PCI 设备；
3. 传输身份：哪个 channel、连接和 QP 正在搬运数据。

`ncclComm_t` 指针只在本进程地址空间有效，`cudaDev` 只是本进程可见设备的本地序号，QP number 也只在其 HCA 和生命周期内有意义。它们都不能单独充当跨进程全局 ID。

## 字段来源表

| 论文/项目字段 | NCCL 2.21.5 来源 | 创建者与生命周期 | 跨 rank / 唯一性 | 结论 |
|---|---|---|---|---|
| `IP` | `src/include/socket.h:27` 的 `ncclSocketAddress`；IB 插件的控制连接也使用 socket address | bootstrap、Proxy 或 NET 插件建连时产生，连接存活期间有效 | 不保证是 RDMA 数据面地址；Socket、IB、RoCE 的含义不同 | **候选，transport-specific**。不能把任意 socket 地址统一解释成数据面 IP |
| `comm_id` / `communicator_id` | `src/init.cc:1538` 从 `ncclUniqueId` 计算 `comm->commHash`；成员定义在 `src/include/comm.h:245` | communicator 初始化时产生，随 `ncclComm` 存活 | 同一初始化输入的 rank 得到相同 hash；64-bit hash 仍存在理论碰撞，指针值不可跨进程使用 | **候选**：Event 中编码为 `capture_id:commHash`，不能只记录 `ncclComm_t` 地址 |
| `Gid` | 论文 Table 2 只列出名字，没有定义；NCCL IB 插件另有 `ibv_gid`、`spn/iid` | 论文语义未知；RDMA `GID` 只在 IB/RoCE 建连中产生 | 两者没有证据证明是同一概念 | **未映射**。禁止仅凭拼写把论文 `Gid` 等同于 RDMA `GID` |
| `GPU_id` | `comm->cudaDev`、`nvmlDev`、`busId` 位于 `src/include/comm.h:248-252`，由 `src/init.cc:332-339` 填充 | communicator 初始化时记录本 rank 使用的设备 | `cudaDev`/`nvmlDev` 是本机序号；`busId` 也必须和主机身份组合 | **候选**：同一采集任务内使用 `(host identity, busId)`；不能只用 `cudaDev` |
| `rank` | `comm->rank`，`src/include/comm.h:246`；`src/init.cc:318` 赋值 | communicator 初始化至销毁 | 只在对应 communicator 内唯一 | **已确认**：必须与 `communicator_id` 组合 |
| `Channel_id` / `channel` | `channel->id` 在 `src/channel.cc:18` 赋值；`proxyOp->channelId` 在 `src/enqueue.cc:1806` 赋值 | communicator channel 初始化后存在，调度时复制进 Proxy op | 数字只在 communicator 内有意义；同一 channel 可承载很多 operation | **已确认**：Event 中必须和 communicator、operation、rank 组合 |
| `QP_id` | `src/transport/net_ib.cc:636` 的 `ncclIbQpInfo.qpn`；本地 QPN 在 `:1022` / `:1335` 从 `ibv_qp::qp_num` 取得 | IB/RoCE 连接建连时创建，QP 销毁后失效 | 一个连接可以有多个 QP；QPN 不是永久全局 ID | **候选，IB-specific**：至少需要连接方向、net device、peer、channel、QP index 和 local QPN，不能只留一个整数 |
| `op_seq` | collective Proxy op 在 `src/enqueue.cc:331` 获得 plan-local `opCount`；`:1100-1146` 用 `sharedRes->collOpCount` 转成历史序号；Proxy 侧保存在 `ncclProxyArgs.opCount` | collective 被调度并上传 Proxy op 时产生；persistent plan 每次上传都会推进历史计数 | collective 合法调用顺序一致时应跨 rank 对齐；无 Proxy 的路径、communicator split/share 和 graph replay 仍需实测 | **强候选**：collective 使用 `proxyOp->opCount >> 1`，并与 `commHash` 组合；Day 16 必须用真实运行动态验证 |
| `msg_size` | API `count`/`datatype` 在 `ncclInfo`；`src/include/info.h:78` 计算 `nBytes`/`workBytes`；`src/enqueue.cc:1793` 计算 `proxyOp->nbytes` | API 入队后得到逻辑字节数，调度后得到每次 Proxy 传输粒度 | 相同 collective 的逻辑大小应跨 rank 一致；每 channel/slice 大小不一定相同 | **已确认有多个层级**：若记录 operation 大小用 `ncclInfo.nBytes`，若记录网络请求大小用 `proxyOp.nbytes`，字段名必须区分 |

## `op_seq` 为什么只能叫强候选

`ncclKernelPlan::collOpCount` 是 plan 内从零开始的 collective 序号。`uploadProxyOps()` 把它平移到 `ncclSharedResources::collOpCount` 的历史末端，再交给 Proxy：

```text
plan-local opCount
        |
        v
uploadProxyOps(): (shared collOpCount << 1) + oldId
        |
        v
ncclProxyArgs.opCount
```

最低位是 collective/P2P 标签，所以 collective 的分析序号候选是 `opCount >> 1`。它比 `comm->opCount` 更合适：后者在 `src/proxy.cc:903-914` 每次 Proxy launch 增加一次，一次 launch 可能包含多个任务。

仍需 Day 16 用两个真实 rank、连续多个 AllReduce、CUDA Graph replay 各跑一次，验证 `(commHash, opCount >> 1)` 在目标工作负载中是否一一对应同一 CollOp。验证前不能把它写成无条件事实。

## `msg_size` 必须先说明层级

同一个 collective 至少有三种“大小”：

- `ncclInfo.nBytes`：operation 的逻辑数据规模；
- 调度到某个 channel 的 work/chunk 大小；
- `ncclProxyOp.nbytes`：一次 Proxy sub-operation/slice 使用的大小。

因此采集层若记录 operation 大小，应采用逻辑 `nBytes`；流量和吞吐分析应另记 transfer bytes。把三者都叫 `msg_size` 会让 MinData 和带宽计算失真。

## E02 三进度量的真实源码候选

Day 08—10 的 E02 是发送侧抽象，不是对 `ncclProxySubArgs` 的逐字段复刻。对应关系如下：

| E02 累计量 | NCCL 发送侧候选 | 证据与限制 | 状态 |
|---|---|---|---|
| `total_chunks` | `sub->nsteps / args->sliceSteps` | `nsteps` 是协议 step 数，不保证等于论文实现所称 chunk；需要统一采样单位 | 候选 |
| `gpu_ready` | GPU 更新 connection FIFO size/tail 后，`sendProxyProgress()` 的 readiness 条件成立 | `src/device/prims_simple.h:139,181` 写 size/tail；`src/transport/net.cc:1084-1118` 检查 SIMPLE/LL/LL128 readiness。没有一个现成的 `gpu_ready` 累计字段 | 候选，必须插桩派生 |
| `rdma_transmitted` | `sub->transmitted` | `isend()` 返回非空 request 后在 `src/transport/net.cc:1121-1124` 推进 | 源码语义已确认；归一化为 E02 counter 仍待 Day 17 的真实 state log 验证 |
| `rdma_done` | `sub->done` | `ncclNet->test()` 报告完成后在 `src/transport/net.cc:1137-1154` 推进 | 源码语义已确认；归一化映射待 Day 17 的真实 state log 验证，且不能解释成远端 GPU 已消费 |

`sub->posted` 不是 `gpu_ready`：它表示 Proxy 已把 buffer/credit 提供给 GPU，发生在 GPU 数据真正 ready 之前。

接收侧 `recvProxyProgress()` 还有 `posted/received/flushed/transmitted/done`，同名 `transmitted` 和 `done` 的语义与发送侧不同。因此 E02 的不变量只对应发送侧简化模型，不能直接套在接收侧计数器上。

## Event v1 与 v2 的兼容边界

Day 13 冻结的 Event v1 继续可读，其内容是分析器的公共身份和两种 payload：

- `(communicator_id, op_seq, collective, rank, channel)`；
- Ring action 或三进度快照。

进入 Day 14 前发现 v1 无法表达真实时间上的停滞、吞吐和 operation 间隔，因此发布兼容的 Event v2，而不改变 v1：

- `NCCL_MONOTONIC_NS` 表示由采集进程的单调时钟取得的纳秒时间；
- `OPERATION_COMPLETION` 的 `event.time.value` 是完成时间，payload 保存同一时钟域的 `started_at_ns` 和 operation 逻辑 `message_bytes`；
- completion 是 operation 级事件，因此 `channel=null`；其他事件仍必须有非负 channel；
- v2 progress snapshot 可以使用真实单调时间，也继续允许 E02 的合成 tick；v1 的时间域规则保持不变。

进程单调时钟不是跨主机同步时钟。原始纳秒值只能在同一 rank 的恢复时间线内直接排序；跨 rank 分析应先计算各自的持续时间、吞吐或间隔，再比较这些派生指标。不能用不同主机的原始 monotonic 值建立全局先后关系。

`IP`、`Gid`、`GPU_id` 和 `QP_id` 仍属于采集元数据，不强塞进每一种 Event payload。Day 16—19 的真实采集链可用它们建立 rank/channel/connection 元数据表。completion 的真实聚合插桩点同样要在 Day 16 的真实运行中动态确认，当前契约只规定分析输入语义，不宣称采集实现已经完成。

## 论文与本项目的边界

Mycroft 论文 Table 2 列出字段类别，并说明从 NCCL Proxy 周期性采集累计进度，但没有公开每个字段在某个 NCCL 版本中的确切成员映射。本表是针对 NCCL 2.21.5 的独立源码映射，不声称复刻论文未公开实现。

论文：<https://doi.org/10.1145/3731569.3764848>

# NCCL 2.21.5 候选插桩点

本文记录 Day13 得到的候选位置；M2 已按这些位置提供 adapter/patch，尚待真实 CUDA 构建与运行验证。按 [v0.5 计划](../../docs/plans/Mycroft_26日开发路线图.md) 的 M1—M3，用普通 RING/SIMPLE NET/IB 真实运行验证。每个位置都基于 submodule 固定的 `v2.21.5-1` 源码。Graph、完整 P2P 和多协议覆盖移出当前范围。

## 候选点总览

| 编号 | 位置 | 能取得的数据 | 用途 | 当前结论 |
|---|---|---|---|---|
| T0 | `src/init.cc` 的 communicator 初始化完成路径，`commHash` 已在 `:1538` 生成后 | `commHash/rank/nRanks/cudaDev/nvmlDev/busId` | 建立 capture 内的 communicator、rank、GPU 元数据 | 候选；只执行一次，开销低 |
| T1 | `src/enqueue.cc:786` 调度路径调用 `ncclInfoSetDerived(nextInfo, ...)` 之后 | collective 类型、count、datatype、`nBytes`、stream | 记录 API/operation 逻辑大小 | 候选；此处还没有最终 Proxy `opCount`，需要和 T2 关联 |
| T2 | `src/enqueue.cc:1100` `uploadProxyOps()` 中，临时转换 `q->opCount` 后、`ncclProxySaveOp()` 前 | 全局化的 `opCount`、channel、collective、protocol、nsteps、nbytes | 把 operation 身份送入 Proxy | **有 Proxy 路径的首选 operation/channel 插桩点**；不能保证覆盖普通 P2P |
| T3 | `src/proxy.cc:350` `ncclProxyOpToArgs()` | `opCount` 与每个 sub 的 channel、peer、nsteps、nbytes | 验证 T2 信息确实进入 Proxy progress | 候选；不应与 T2 重复长期记录 |
| T4 | `src/transport/net.cc:1030` `sendProxyProgress()` | readiness 条件、`transmitted`、`done` | 生成发送侧周期性进度快照 | **首选进度插桩区域**；不能把 `posted` 当 GPU ready |
| T5 | `src/transport/net.cc:1184` `recvProxyProgress()` | `posted/received/transmitted/done`；flush 完成后推进 transmitted | 补充所选延迟用例需要的对端/接收证据 | sub->flushed 字段存在，但该 NET 路径未独立维护；不套用 E02 发送侧定义 |
| T6 | `src/transport/net_ib.cc` 的 connect/accept，QPN 在 `:1022/:1335` 写入 metadata | link layer、device、LID 或 RoCE GID、一个或多个 QPN | 建立 IB/RoCE connection 元数据 | 候选且 IB-specific；Socket transport 不使用 QP |

## T2：operation 与 channel 关联

Day16 框架起步已确认源码路径限制：`addProxyOpIfNeeded()` 只有在 `ncclProxySaveOp(..., &needed)` 确认 needed 为 true 时保存队列条目；`src/proxy.cc:525` 的 `SaveProxy()` 在 progress 回调为空时直接返回。默认 P2P transport 的 progress 为空，显式开启 CUDA memcpy 路径时发送侧可以例外安装回调。Day15 的真实连接为 P2P/CUMEM，因此不能只依赖 Proxy 队列条目覆盖全部 operation/channel，也不能把空队列解释成操作没有发生或已经完成。v0.5 先验证双节点 NET/RDMA，不为此新增 P2P GPU 采集框架。已有源码结论见 [`day16/README.md`](day16/README.md)。

`addTunedCollToPlan()` 给同一 collective 的各 channel Proxy op 写入相同的 plan-local `opCount`。`uploadProxyOps()` 再把它转换为 communicator 历史中的编号，并调用 `ncclProxySaveOp()`。随后代码会把 `q->opCount` 恢复为 plan-local 值，所以必须在转换后、恢复前捕获。

候选记录：

```text
capture_id
commHash
op_seq_candidate = q->opCount >> 1
rank
channelId
coll / protocol / pattern
nsteps / sliceSteps / nbytes
```

不要使用 `comm->opCount` 替代：它按 Proxy launch 计数，不是按 CollOp 计数。

## T4：发送侧三段进度

发送路径的真实顺序是：

```text
Proxy posted buffer/credit
        |
GPU 写好 FIFO/tail，readiness 成立
        |
ncclNet->isend() 返回 request，transmitted 推进
        |
ncclNet->test() 完成，done 推进
```

因此候选累计快照为：

```text
gpu_ready_steps       # 需要新增轻量计数；readiness 首次成立时推进
rdma_transmitted      = sub->transmitted
rdma_done             = sub->done
total_steps           = sub->nsteps
```

对外转换成 chunk 前必须除以或按 `sliceSteps` 归一化，且四个量必须使用同一单位。`sub->posted` 只表示 Proxy 给 GPU 的可用槽位/credit，不代表 GPU 数据已准备好。

M2 框架的 `normalize_send_steps()` 只检查普通计数的整数 slice 和累计顺序，adapter 已提供接入代码，真实运行仍待验证。`sub->reg` 分支可以动态增加 `nsteps`，并存在单 GPU step 对多个网络 step 的区别；接口保留路径标记，当前转换器拒绝其未经确认的映射。readiness 必须去重并在 isend 前维护，即使 request 暂未返回或延迟提交，周期采样仍须继续。

完成条件候选是所有 sub 的 `done == nsteps`，而不是单个 channel 的一个 request 完成。

## T6：网络连接元数据

IB 插件的一个 connection 可以创建多个 QP，`ncclIbConnectionMetadata.qpInfo[]` 保存各 QPN。RoCE 使用 GID 的 `spn/iid`，native InfiniBand 走 LID。建议只在建连时写一次元数据：

```text
transport = IB or RoCE
direction = send or recv
netDev / devName
local rank / remote rank
channelId
qp_index / local_qpn / remote_qpn
lid or gid (according to link_layer)
```

运行时进度事件只引用紧凑的 connection ID，避免在 Proxy 热循环重复写长地址结构。

## Event v2 的采集要求

Event v2 为后续分析规定输入契约，但没有把候选点伪装成已完成插桩：

- 进度事件应在 T4 使用采集进程的同一单调时钟，写入 `NCCL_MONOTONIC_NS`；
- 论文的 completion log 表示 CollOp 完成；所有 channel/sub 完成只是当前候选必要条件，是否足以代表 CollOp 完成必须在 M2 动态确认。普通单操作计划仍需核对 GPU 完成、abort/error 和观测 stream。确认前不得生成 `OPERATION_COMPLETION`；
- `started_at_ns` 必须与完成时间来自同一进程、同一时钟域；
- `message_bytes` 使用 operation 级 `ncclInfo.nBytes`，不能用单个 Proxy slice 的 `nbytes` 冒充；
- completion 是 operation 级记录，所以 `channel=null`；channel 级 progress 仍保留实际 channel；
- 不同 rank/主机的原始 monotonic 纳秒值不可直接比较，只能先在各 rank 内计算 duration、throughput 或 interval。

## 当前必须完成的动态验证（M2—M3）

1. 两个 rank 连续运行至少三个 AllReduce，确认相同 CollOp 的 `(commHash, opCount >> 1)` 跨 rank 对齐；
2. 至少一次多 channel 运行，确认同一 CollOp 的所有 channel 共享 `op_seq` 而 `channelId` 不同；
3. 动态确认当前普通单操作计划的整体完成观测；不把 API 返回、FIFO 读取确认或 Proxy request 完成直接作为 completion；
4. 对 SIMPLE 协议记录 readiness、`transmitted` 和 `done`，检查累计顺序和单位；
5. 在真实 IB/RoCE 环境验证必要连接元数据和对端证据；Socket 仅作必要排障，不伪造 QP/GID 或称其进度为 RDMA；
6. 比较打开/关闭插桩的正确性和开销，热路径不得进行阻塞 I/O 或动态格式化日志。

只有通过这些动态检查的候选字段，才能升级为“已确认的真实 trace 语义”。

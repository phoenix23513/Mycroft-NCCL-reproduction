# NCCL 2.21.5 候选插桩点

本文记录 Day 13 得到的候选位置，不在 Day 13 修改 NCCL；Day 16—20 将按真实运行结果逐项验证和更新。每个位置都基于 submodule 固定的 `v2.21.5-1` 源码。

## 候选点总览

| 编号 | 位置 | 能取得的数据 | 用途 | 当前结论 |
|---|---|---|---|---|
| T0 | `src/init.cc` 的 communicator 初始化完成路径，`commHash` 已在 `:1538` 生成后 | `commHash/rank/nRanks/cudaDev/nvmlDev/busId` | 建立 capture 内的 communicator、rank、GPU 元数据 | 候选；只执行一次，开销低 |
| T1 | `src/enqueue.cc:786` 调度路径调用 `ncclInfoSetDerived(nextInfo, ...)` 之后 | collective 类型、count、datatype、`nBytes`、stream | 记录 API/operation 逻辑大小 | 候选；此处还没有最终 Proxy `opCount`，需要和 T2 关联 |
| T2 | `src/enqueue.cc:1100` `uploadProxyOps()` 中，临时转换 `q->opCount` 后、`ncclProxySaveOp()` 前 | 全局化的 `opCount`、channel、collective、protocol、nsteps、nbytes | 把 operation 身份送入 Proxy | **首选 operation/channel 插桩点** |
| T3 | `src/proxy.cc:350` `ncclProxyOpToArgs()` | `opCount` 与每个 sub 的 channel、peer、nsteps、nbytes | 验证 T2 信息确实进入 Proxy progress | 候选；不应与 T2 重复长期记录 |
| T4 | `src/transport/net.cc:1030` `sendProxyProgress()` | readiness 条件、`transmitted`、`done` | 生成发送侧周期性进度快照 | **首选进度插桩区域**；不能把 `posted` 当 GPU ready |
| T5 | `src/transport/net.cc:1184` `recvProxyProgress()` | `posted/received/flushed/transmitted/done` | 将来分析接收侧或远端阻塞 | 本轮不映射到 E02；语义另行定义 |
| T6 | `src/transport/net_ib.cc` 的 connect/accept，QPN 在 `:1022/:1335` 写入 metadata | link layer、device、LID 或 RoCE GID、一个或多个 QPN | 建立 IB/RoCE connection 元数据 | 候选且 IB-specific；Socket transport 不使用 QP |

## T2：operation 与 channel 关联

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
- 论文的 completion log 表示 CollOp 完成；所有 channel/sub 完成只是当前候选必要条件，是否足以代表 CollOp 完成必须在 Day 16 动态确认。确认前不得生成 `OPERATION_COMPLETION`，`event.time.value` 的真实观测点也不得预设；
- `started_at_ns` 必须与完成时间来自同一进程、同一时钟域；
- `message_bytes` 使用 operation 级 `ncclInfo.nBytes`，不能用单个 Proxy slice 的 `nbytes` 冒充；
- completion 是 operation 级记录，所以 `channel=null`；channel 级 progress 仍保留实际 channel；
- 不同 rank/主机的原始 monotonic 纳秒值不可直接比较，只能先在各 rank 内计算 duration、throughput 或 interval。

## Day 16—20 必须完成的动态验证

1. 两个 rank 连续运行至少三个 AllReduce，确认相同 CollOp 的 `(commHash, opCount >> 1)` 跨 rank 对齐；
2. 至少一次多 channel 运行，确认同一 CollOp 的所有 channel 共享 `op_seq` 而 `channelId` 不同；
3. CUDA Graph replay 重复执行，确认每次 replay 是否得到新序号；
4. 对 SIMPLE 协议记录 readiness、`transmitted` 和 `done`，检查累计顺序和单位；
5. 分别在 Socket 和 IB/RoCE 环境验证 transport-specific 字段，不要求 Socket 伪造 QP/GID；
6. 比较打开/关闭插桩的正确性和开销，热路径不得进行阻塞 I/O 或动态格式化日志。

只有通过这些动态检查的候选字段，才能升级为“已确认的真实 trace 语义”。

# M2 最小采集：保留的 Day16 框架

当前状态：框架已搭建，C++ 采集函数仍返回 `NotImplemented`，真实结果核对入口仍退出 2；结果打包工具可独立使用。尚无 Day16 NCCL patch、采集结果或 GPU 验收，不能把 Day15 的应用标记改名为 completion log。

执行依据已更新为 [v0.5 核心复现计划](../../../docs/plans/Mycroft_26日开发路线图.md)：当前先执行 M1 双节点 NET/RDMA 路径确认，再执行 M2 普通 RING/SIMPLE 的最小采集。本文保留既有接口和源码结论；Graph、grouped collectives、完整 P2P GPU 标记/managed memory 方案均已移出当前执行范围，不再是本框架的验收门槛。

## 框架职责与实现边界

目标：复用已搭建接口、源码观测边界、验证入口及单文件结果传输，为 M2 的目标 NET 路径采集服务。当前下一步是 M1，不能继续把无 Proxy 的 P2P 兼容扩展作为前置任务。

输入：固定 NCCL 2.21.5 源码、Day15 原生 AllReduce workload 及已取得的真实选择日志。M2 计划输出：同一操作的身份/channel/peer 关联、基于同进程单调时钟的 Event v2 completion、周期 state log、必要接收端证据、构建来源和运行状态；本次仅提供这些输出的接口骨架及打包工具。

最小知识：一次 collective 可拆到多个 channel；API 返回和 Proxy 子任务完成与整个本地 collective 完成有不同语义；单调时间在各进程内使用；operation 逻辑字节数与单个 slice 的传输大小不同。

框架检查命令见下方“本地检查”；预期为接口语法和打包测试通过、真实核对入口明确返回未实现。M2 最终验收必须来自目标路径的项目修改版 NCCL：两个 rank、普通连续多个操作、实际多 channel 身份、时间/大小正确、数值结果正确，完成条件有可追溯的真实依据，并包含周期 state log 与必要对端状态。具体五步及完成标准只在路线图定义。

## 文件和职责

```text
instrumentation/nccl-2.21.5/day16/
├── include/operation_trace.h  # key、begin、channel、completion 与状态接口
├── src/operation_trace.cpp    # 采集和导出待实现，不触碰 NCCL 源码
├── verify_capture.py         # 复用 Event v2 格式检查，完整真实核对待实现
├── check_framework.sh        # 本地 C++ 语法与 Python schema 导入检查
└── tests/test_result_bundle.py

cluster/crater/scripts/package_day16_results.py
```

`OperationKey` 包含 `comm_hash`、`op_seq_candidate` 和 rank。`op_seq_candidate` 尚未确认，M2 须动态验证普通 NET/IB 操作的跨 rank 对应；Graph replay 不受支持。`record_channel_membership()` 将同一个 key 与各 channel 关联；operation 级 completion 的 Event v2 `context.channel` 为 null。

`record_operation_begin()` 保存操作逻辑大小与开始时间；`record_operation_completion()` 必须由经过确认的本地整个操作完成观测点调用。`export_event_v2()` 在 workload 停止采集后导出，begin 与 end 使用同一进程单调时钟。它们是进程内草案；共享内存 ABI 已移出当前执行范围。

M2 热路径实现必须写入预分配的有界内存，不格式化 JSON、不执行阻塞文件 I/O；缓冲区满时报告 dropped。当前未实现状态是显式返回值，不能忽略后继续声称已采集。

代码 TODO 对应 M2 的具体缺口：预分配与 dropped 统计、目标 NET 身份/channel/peer、可靠整体 completion、周期进度与单位、后台或受控停止后导出及真实证据核对。现有头文件仅覆盖 begin/channel/completion 草案，尚无 state/peer 采集接口；计划要求已明确，接口需在 M2 实现时按实际源码与动态证据补齐。目录名 `day16/` 和既有工具名保留以兼容调用，不表示继续旧逐日路线。

## 已知路径限制：P2P 不必有 NET Proxy 进度

Day15 TRACE 的实际选择是 RING/LL（16 B、16 KiB）及 RING/SIMPLE（4 MiB），连接为 P2P/CUMEM。固定源码中：

1. `src/enqueue.cc:176` 的 `addProxyOpIfNeeded()` 调用 `ncclProxySaveOp(..., &needed)`；只有 needed 为 true 才将条目放进计划的 Proxy 队列。
2. `src/proxy.cc:525` 的 `SaveProxy()` 在 `connector->proxyConn.proxyProgress == NULL` 时直接返回；`ncclProxySaveOp()` 先把 needed 设为 false。
3. `src/transport/p2p.cc:756` 的默认 P2P transport 没有 `proxyProgress`；显式开启 P2P CUDA memcpy 路径时，发送侧可以另行安装 progress 函数。这不等同于 NET 的进度回调。

这些是源码结论；实际每个 channel 是否进入 Proxy 队列仍要在插桩运行中观察。因此仅在 T2/T3 记录 Proxy op，不能无条件覆盖 Day15 P2P workload，也不能据此宣称已经得到所有操作的完成记录。

已沿 `addCBDCollToPlan()` → `appendWorkElemColl()` → kernel plan → `ncclLaunchKernel()` → GPU work 读取源码，下节保留结论。这说明 Day15 workload 不能直接验证 NET 的三进度量，所以 M1 改用真实双节点 NET/RDMA；本版不实现 P2P 专用采集器。`ncclLaunchFinish()` 在 host 侧释放/连接 stream 等依赖，其函数名不证明 GPU 已执行完。M2 完成观测限定普通单操作计划，仍须确认实际 stream、abort/error 和覆盖范围。

## 已完成的源码追踪：身份怎样经过多个 channel

已读取固定版本源码；下表是源码结论，不是新增 GPU 运行证据。

| 位置 | 状态或数据变化 | 对插桩的意义 |
|---|---|---|
| `enqueue.cc:438`，`addCBDCollToPlan()` | 在 channel 循环前取 `plan->collOpCount++`，左移一位作为 Proxy collective 标签值 | 同一 collective 的各 channel 使用同一 plan-local ordinal；下一 collective 才递增 |
| `enqueue.cc:461`，`setCollWorkElem()` / `appendWorkElemColl()` | 先设置该 channel 的 count/offset，再拷贝 work 到 channel 队列 | 可以关联 operation 与实际参加的 channel；逻辑总字节数仍取 `collInfo->nBytes` |
| `include/device.h:227`，`ncclWorkElem` | 有 count、workCount、workOffset 等执行字段，没有 `commHash` 或完整 operation 序号 | 不能假设 GPU work 已自动携带我们的完整操作身份，需要另建关联或可复查扩展 |
| `enqueue.cc:565`，`finishPlan()` | 从非空 work 队列构造 channelMask/channelCount；Proxy 队列是否非空另算 hasProxyOps | work 与 Proxy 是两类队列，不能用 Proxy 队列为空判断 GPU 没有工作 |
| `enqueue.cc:1351`，`ncclLaunchKernel()` | grid.x 等于计划 channelCount，参数包含 channelMask 与 workHead | launch 对象是整个 plan，不无条件等于单个 collective |
| `device/common.h:125`，`ncclKernelMain()` | GPU block 根据 mask 中第几个置位映射到 channel，读取该 channel 的 work | blockIdx.x 是压缩后的 block 编号，不总等于 channelId |

例如某个 mask 只包含 channel 0、2、5，GPU blocks 0、1、2 分别处理 channel 0、2、5。必须记录实际 channelId，不能直接记录 blockIdx.x。这是解释映射的示例，不是 Day15 实测的 channel 集合。

`appendWorkElemColl()` 可以把多个 collective work element 合进同一个 `ncclWork`，一个 plan 也可以容纳多个 collective。于是 operation、work element、work FIFO slot、kernel launch 都有不同粒度。观察整个 kernel 完成可以覆盖该 plan 中的工作，但不能据此声称其中各操作有不同的精确 GPU 结束时间。

### 普通执行与 Graph 执行的序号边界

`uploadProxyOps()` 开头先保存旧的 `comm->sharedRes->collOpCount` 为 base，再加上 `plan->collOpCount`，随后才遍历 Proxy 队列。有 Proxy collective 条目时：

```text
plan_local_ordinal = oldId >> 1
historical_candidate = base + plan_local_ordinal
```

因此满足 `!persistent && persistentRefs == 0 && !ncclCudaLaunchBlocking` 的普通分支，会在 `ncclLaunchKernelAfter_NoCuda()` 调用 `hostStreamPlanTask()`，继而执行这次 base 推进；即使 Proxy 队列为空，推进仍发生，只是没有 q 条目供 T2 循环采集。不能把“无法从 Proxy 条目取到 key”说成“历史计数一定没推进”。多 plan 调度时，base 应绑定到该 plan 的实际推进时刻，不能在提前构建所有计划时反复读取同一个旧 base。

Graph capture 使 plan.persistent 为 true。`ncclLaunchPrepare()` 在这条分支仅对 hasProxyOps 的 plan 安排 `hostStreamPlanCallback()`；`ncclLaunchKernelAfter_NoCuda()` 的普通调用分支也被绕过。所以无 Proxy 的 persistent plan 不能直接依赖这条既有回调为每次 replay 分配新序号。未来扩展 Graph 支持时需要另行验证执行实例身份；不能把 capture 时的 plan-local 编号永久当作每次 replay 的 operation 编号。本版只保留这一限制，不安排 replay 实验。

此外，Day15 INFO 的 `AllReduce: opCount` 来自 `comm->opCount`。它在 `ncclProxyStart()` 中按调用推进，和这里的 collective 历史计数不是同一个字段。Day15 串行单 collective 场景中显示 0—8，不构成其在 grouped collectives 或 Graph replay 中一一对应 operation 的证明。

### 完成信号：workFifoDone 为什么不够

`device/common.h:181` 的顺序是：

```text
读取 work 描述到 GPU shared memory
    → workFifoDone = doneAcks（最后一个 FIFO work 的读取确认）
    → SpecializedRunWork().run(...) 或 ncclDevFuncTable[...]()
    → block 内同步，继续下一个 work 或退出
```

`workFifoDone` 用于 host 端 `waitWorkFifoAvailable()` 回收描述符槽位；它在当前 work 的通信计算前更新，故不能作为该 channel 的通信完成，更不能作为整个 CollOp completion。头部的 isLast 只是该 channel 在本次 kernel 的最后一份 work 描述。

后续完成观测应覆盖整个目标操作的 GPU 工作及全部参与 channel，并和身份关联；必须处理 abort/error，避免把提前退出当成成功完成。CPU 观测时间可用于本进程时差，但要明确是观测时刻，不能伪称精确 GPU 结束时间。`ncclLaunchFinish()` 和 Proxy request completion 同样不能直接替代这个条件。

尚未确定完成采集实现，也未修改 NCCL。上述 Graph/P2P 结论保留为支持范围限制，本版不再要求同时覆盖这些路径。下一项是 M1 实际 NET/RDMA 验证，随后 M2 在普通 RING/SIMPLE 路径确认身份与 completion/state 语义。

## 本地检查

从仓库根目录执行，不需要 CUDA/GPU：

```bash
bash instrumentation/nccl-2.21.5/day16/check_framework.sh
python3 -m unittest discover -s instrumentation/nccl-2.21.5/day16/tests -p 'test_*.py' -v
```

框架检查应打印 `day16_framework=PASS scope=CPU_syntax_and_schema_import_only`、`day16_capture=NOT_IMPLEMENTED`、`day16_gpu_acceptance=NOT_RUN`。打包测试使用明确标记的文件 fixture，只验证传输工具，不产生 NCCL 验收证据。

核对入口尚未实现，可观察其明确退出状态：

```bash
PYTHONPATH=src python3 instrumentation/nccl-2.21.5/day16/verify_capture.py \
  .build/day16-upload/day16-results
```

当前应返回 2 并打印 `capture_verification=NOT_IMPLEMENTED`，不写结果、不把任意合法 Event JSON 当成真实完成证据。后续要同时核对 patch/build hash、观测语义、丢失计数、workload 结果与逐操作身份，不能只做 JSON 格式验证。

## 单个结果包与接收目录

本地 `.build/day16-upload/` 已提前创建。Windows 文件管理器先打开 `\\wsl.localhost\Ubuntu-22.04`，进入本仓库的 `.build/day16-upload`；用户以后仅下载并复制一个 `day16-results.tar.gz`，由 Codex 解包核对。

已实现的打包工具接受以下结果布局，缺失项会记录在包内 `bundle-manifest.json` 中；`run-status.txt` 必须存在：

```text
<RESULT_DIR>/
├── run-status.txt
├── environment.txt
├── capture-manifest.json
├── analysis.txt
├── workload/{run.log,rank0.log,rank1.log}
├── trace/{rank0.jsonl,rank1.jsonl,channel-map.jsonl}
└── nccl-build/{build-manifest.txt,run-status.txt,verification.log,saved-verification.log}
```

将其打成新文件：

```bash
python3 cluster/crater/scripts/package_day16_results.py "<RESULT_DIR>" \
  --output "<RESULT_DIR>.tar.gz"
```

包内统一以 `day16-results/` 为根，保留目录结构。仅包含明确列出的必要文本证据；如果有编译日志，仅保存末尾 128 KiB。不会附带库、头文件或中间产物，也不另写必须下载的 SHA 文件；archive SHA256 打印在终端。已有包拒绝覆盖，先写临时文件再原子发布。

失败作业同样能打包已产生的日志，包内保留原始非零退出状态；打包成功不代表实验通过。未来真实采集运行脚本必须在退出时先保存作业状态，再调用此工具。目前还没有该运行脚本或可运行的插桩包，不应为这份骨架创建 GPU 作业。

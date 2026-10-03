# M3：真实软件延迟与三组对照

执行依据是[当前计划的 M3](../../../docs/plans/Mycroft_26日开发路线图.md#m3真实软件延迟与对照证据)。M2 r3 已取得双物理节点 NET/IB 采集技术证据，用户授权进入 M3。本目录提供非阻塞注入、NCCL 增量补丁、三组运行及离线核对框架；本地验证不等于 CUDA 编译或真实延迟效果通过。

## 问题、假设和对照

问题：延迟一个已知可发送请求，是否能在真实通信记录中观察到提交停滞，并在释放后恢复正确完成？这为 M4 提供有已知异常来源的真实输入。

| 组别/目录 | 库 | 采集与延迟 | 核对 |
|---|---|---|---|
| A / `baseline` | 固定原始源码新构建的未插桩 TRACE 库（构建/加载已验证） | 采集关闭，延迟关闭 | 双节点 NET/IB、RING/SIMPLE、各九次结果正确 |
| B / `capture` | 新构建的 M3 库 | 采集开启，`delay_ns=0` | M2 原有身份/进度/completion 对账、零丢失，未主动延迟 |
| C / `delay` | 与 B 相同路径/hash 的库 | 采集开启，目标延迟开启 | 命中、实际等待窗口、解除后恢复、结果正确 |

三组都执行 float32 Sum AllReduce，16 B、16 KiB、4 MiB 各三次；每组启动全新的进程和 communicator。一次双节点作业组合按 A→B→C 顺序执行，在同一对节点、GPU、镜像、工具链和 NET 设置下对照；核对失败即停止后续组。历史 M1 结果不冒充新对照测量。

已知：M2 的单操作计划、真实 channel/peer 和进度对账已通过；M3 补丁已在 Crater CUDA 12.5.82 环境编译并完成版本/加载/导出接口检查；注入仅作用于 `isend` 调用前。本轮已验证：目标命中、等待期间持续采样、对端 channel0 受阻及释放后恢复；自动根因诊断属于 M4，尚未完成。第一版默认 rank0/op6/channel0 的第一个可发送请求，等待 100 ms；op6 是第一次 4 MiB 操作，M2 已观察到两个 channel。100 ms 是可调初值，本轮实际等待 100.000906 ms、保存 9,725 条目标区间样本，足以观察本次等待与恢复；不外推到其他配置。没有命中、等待样本不足、丢失或数值错误，均拒绝该次实验核对。

## 核心代码如何工作

```text
原有 GPU readiness 检查通过
  → M2 保存观察到的可提交边界
  → M3 是否允许本次 isend？
      非目标/开关关闭：正常提交
      第一次命中：保存 begin 和 deadline；暂不提交
      尚未到期：暂不提交；继续循环和已有请求完成检查
      到期：保存 release；正常提交，以后不重复延迟
  → 原有网络 test/done 与周期 state 采样继续
  → stream 同步、数值检查、Proxy join
  → 导出真实轨迹和 injection.json
```

- `include/send_delay.h`：无分配、无 sleep、无 I/O 的一次性截止时间开关，调用者提供单调 ns；只匹配 rank/op/channel，并保持目标请求的 step。异常时让通信继续，但将注入证据标错。
- `nccl_adapter/mycroft_m3.cc`：在真实可发送分支读取 tag 和 step；记录命中/释放时刻，在两个边界强制保存 M2 快照。热路径只更新内存；`injection.json` 在 M2 成功确认生产者 join 后才导出。
- `nccl-2.21.5-m3.patch`：在 M2 补丁后的源码上追加一个 host 源文件，并用开关包住 `isend` 和对应提交量更新。等待分支不 `return`、不 `continue` 整个 sub，因此已有网络请求仍可检查完成。
- 原生 workload：按显式环境参数调用 `mycroftM3Configure`，完成 M2 导出后调用 `mycroftM3Finish`。普通 M1/M2 不设置这些参数；runner 清除继承的 M3 开关。
- `verify_experiment.py`：复用 M1 运行核对及 M2 逐 rank capture 核对，另核对三组来源、环境、注入控制和目标窗口。

目标 C 的等待区间必须实际有至少两条 target send 样本、样本跨度至少一个采样周期、readiness 高于提交量且提交量保持在被暂缓的 step；释放不能早于 deadline，之后网络最终状态和本地整体 completion 必须出现。报告保存实际等待时长、样本跨度和最大间隔，不声称调度器能精确保持 10 μs 间隔。其他路径可能因依赖受阻；只有目标路径主动执行延迟。对端因果分析留给 M4，不用注入配置代替诊断。

## 构建与运行入口

复用 M2 的安全源码准备、CUDA 构建、版本/导出接口检查和结果打包机制。`prepare_m3.py` 先应用 M2 补丁，再应用 M3 增量补丁；新产物标为 `m3_instrumented`，保存九项源码输入和十二项准备后文件 hash。沿用 `m2-build.json`/`m2-source-manifest.json` 文件名以复用接口，`build_kind` 明确区分版本；不能把旧 M2 库当作包含 M3 的库。B/C 的新 M3 库已构建一次并共用；A 当前使用另行恢复的未插桩 TRACE 对照库，两份库均通过 CPU 构建/加载核对。上游 submodule 保持干净。

本地打包：

```bash
python3 cluster/crater/scripts/package_m3.py
```

生成一个 `.build/m3-upload/m3-experiment.tar.gz`。在 CUDA 12.5 devel 环境解包后，从 `m3-experiment` 根目录运行：

```bash
python3 cluster/crater/scripts/build_m3.py --work-dir <共享目录>/M3/m3-build-r1 --jobs 2
```

CPU 构建不申请 GPU/RDMA；成功产物为 `<共享目录>/M3/m3-build-r1/nccl/`。构建器检查两个新增 M3 导出接口，加载版本 22105，保存失败证据；不执行 collective。

GPU 执行沿用 M2 已验证的两个 Custom 作业，各 2 CPU、8 GiB、1 V100、1 RDMA，同一拓扑，各固定不同节点，Bash/root。两侧从解包根目录分别执行完整的 rank 参数形式：

```bash
python3 cluster/crater/scripts/run_m3.py --rank 0 --run-dir <共享目录>/M3/m3-controls-r1 --baseline-nccl-root <共享目录>/M3/m3-baseline-r1/nccl --baseline-sha256 175ee5e09cb0ca6a908a7335d518d649e12ceb5257abc96200206083851be816 --nccl-root <共享目录>/M3/m3-build-r1/nccl --target-rank 0 --target-operation 6 --target-channel 0 --delay-ms 100 --peer-timeout 900
```

```bash
python3 cluster/crater/scripts/run_m3.py --rank 1 --run-dir <共享目录>/M3/m3-controls-r1 --baseline-nccl-root <共享目录>/M3/m3-baseline-r1/nccl --baseline-sha256 175ee5e09cb0ca6a908a7335d518d649e12ceb5257abc96200206083851be816 --nccl-root <共享目录>/M3/m3-build-r1/nccl --target-rank 0 --target-operation 6 --target-channel 0 --delay-ms 100 --peer-timeout 900
```

这些是接口说明，含占位符，不能原样粘贴到 Crater。历史路径未通过存在性检查后，用户要求按没有可用原始库处理；新的未插桩对照库已在 M3 专用目录构建并通过持久化加载检查，当前两侧完整命令已据本轮证据生成。默认原始库 hash 为已验证 M1 TRACE 库的 `e4437dd0b48e3ab426b17b394043a0d162f1c9040b32bcf34b5b866e6286132a`。必要时用 `--baseline-sha256` 指定另一个独立验证过的库，不能绕过原始构建来源检查。

原始对照库恢复已完成：上传本地保留的固定原始源码包，复用 Day14 CPU 构建/持久化加载入口，获得新的未插桩 TRACE 库。本步骤不重跑历史 M1 GPU，也未重建 B/C 的 M3 库。新库来源、编译与加载检查通过，实际 hash 为 `175ee5e09cb0ca6a908a7335d518d649e12ceb5257abc96200206083851be816`；当前命令显式设置 `--baseline-sha256`，不将新库冒充旧 M1 二进制。A/B/C 三组真实 GPU 对照技术核对已通过，用户已明确授权进入 M4，M3 已验证。个人绝对路径和完整 GUI 启动命令保存在忽略的 `.build/m3-upload/`：`crater-m3-rank0-command.txt`、`crater-m3-rank1-command.txt`；两侧各一个 Custom 作业即可顺序执行三组。

两侧共用新的 suite 目录，每组有独立 run_id、rank 目录和 NCCL ID；原有目录和包不得覆盖。启动两侧后不单独再提交 B/C。rank0 最终打一个 `m3-controls-r1.tar.gz`，成功失败均保留已有证据；只下载此文件到预先创建的 `.build/m3-download/`。worker 不另打包。每个通信子进程上限沿用 240 秒，注入范围 1..2000 ms。

本地离线复核：

```bash
python3 instrumentation/nccl-2.21.5/m3/verify_experiment.py <解包目录>/m3-results
```

报告的 call→stream_sync 时间只属于同一 rank 的 CPU 观测；每种消息量只有三次样本，保存原值及中位数，不作为生产性能结论。`injection.json` 是核对期望位置的实验记录，不供 M4 当作根因答案。三组数值和技术证据通过后还需结合真实传播轨迹复核 M3，不自动标为 M4/M5 完成。

## 本轮真实结果与解释

2026-10-03 的三组两侧进程、数值和完整技术核对均通过。本地复核与远端分析逐字一致；93 项文件齐全，三组共 54 项数值检查正确，B/C 采集零丢失。两侧物理节点确实不同，同一 rank 跨组使用相同 GPU/节点和记录的环境；实际 transport 为 NET/IB。

目标发送在约 100 ms 中持续表现为 readiness=2、submitted=done=0，释放后完成至 8。rank1 对应接收已投递到 8，但 received/done=0 保持约 97.754 ms；其发送在 readiness=submitted=done=4 保持约 97.619 ms。rank0 的 channel1 在本地约 3.035 ms 完成收发，说明本轮并未阻塞整个 Proxy。结合已知干预与双方连接，此轨迹支持“目标发送暂缓，使对端及同一 channel 后续工作受阻”；自动从日志识别主动候选留给 M4。

op6 的本地 CPU API→stream_sync 观测：B 为 rank0 0.851 ms、rank1 0.892 ms；C 为 100.627 ms、98.275 ms。C 的 op7/op8 回到 0.74–0.82 ms。三次中位数会掩盖唯一延迟操作，必须保留逐操作时间；这些不是精确 GPU 时间，也不能比较跨主机原始单调时间戳。A/B 小样本不足以给出稳定插桩开销结论。

原始证据、独立复核和逐路径统计保存在忽略的 `.build/m3-download/controls-r1/`，其中 `实验结果解读.md` 面向用户解释，`observations.json` 保存精确口径。技术证据已通过，用户已明确授权进入 M4，M3 已验证；原始报告的 PENDING_USER_REVIEW 保留，M4/M5 尚未由本轮验证。

## 本地检查与证据边界

```bash
python3 -m unittest discover -s instrumentation/nccl-2.21.5/m3/tests -p 'test_*.py'
```

检查覆盖实际 C++ gate、固定源码上的两层补丁、注入窗口负例和三组运行协调；模拟子进程只验证控制流程，手写轨迹只验证核对规则，均不冒充 CUDA 编译、GPU 运行或 Mycroft 诊断结果。核心路线、进度与验收只由当前计划规定。

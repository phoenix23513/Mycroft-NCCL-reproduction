# Mycroft 核心复现执行计划

- 版本：v0.5（仅执行核心复现必需步骤）
- 当前规则：仓库根目录 `AGENTS.md`
- 历史版本：v0.1—v0.4 的完整内容保留在 Git 历史中，关键变更见第 15 节；文件名保留以兼容现有链接
- NCCL 目标版本：2.21.5
- 最终目标：在真实 NCCL 2.21.5 双节点 RDMA 运行中，完成最小的日志采集、异常触发和依赖驱动 RCA 闭环
- 执行规模：剩余工作按五个必做里程碑验收，不再要求逐项完成原 Day16—26；不承诺固定开发日数量
- 当前状态：Day15 已完成并提交/push（`4638847`）。已有 Day16 采集接口与结果打包骨架，现复用于 M2。M1 补齐运行库后的 Crater 重试完成双节点 NET/IB AllReduce，原作业因日志交错导致核对失败，修复后的原始日志离线复核通过。M1 已验证并提交到 main（`f333d6d`）；用户明确进入 M2。M2 进程内有界记录、周期辅助、丢失统计和停止后导出已实现并通过 CPU 检查；NCCL adapter/patch、构建/双 rank 启动/打包和真实核对代码已实现；Crater 插桩库 CUDA 构建与加载检查已通过，r3 双节点 GPU 采集及技术核对已通过，完整 M2 验收待用户复核。

## 1. 计划要解决的问题

本项目把 Mycroft 复现作为开发主线，不再把“先系统学完 NCCL”作为启动条件。每个阶段都从一个可运行功能出发；只有实现遇到实际缺口时，才补充解决该缺口所需的 NCCL、C/C++、Python、Docker、PyTorch、Kubernetes 或 RDMA 知识。

每次只推进一个边界明确、可验证的子任务；同一里程碑的实现、解释和实验可以连续推进，不为凑开发日增加等待或独立作业。解释过、阅读过或通过自测均不等于完成；只有代码、必要检查、固定演示和真实运行证据一致时，子任务才算通过。commit/push 仍需用户明确指令。

从 Day14 起改变执行原则：先构建、运行和插桩真实 NCCL 2.21.5，再实现消费这些真实轨迹的分析器。E01、E02 和手写 Event 仍可作为确定性单元回归，但不能单独充当真实实验或核心复现的验收证据。

文档与代码按以下职责统一：`AGENTS.md` 管理稳定协作规则；本文唯一规定范围、顺序、验收和动态进度；实验 README 说明现有接口、可运行命令及限制；代码 TODO 标明里程碑和具体实现缺口。下层文件不得把已移出范围的功能重新设为必做项。历史学习资料只保留原有知识与版本背景。

本文是当前唯一的开发路线与动态进度文档：第 2 节定义当前范围，第 9 节保留 Day01—15 的历史任务及新的 M1—M5 任务卡。稳定的协作、Git、文档和安全规则统一由 `AGENTS.md` 管理。用户已要求只执行必做步骤，本版替代旧版的剩余安排和验收范围。

## 2. 当前项目范围与完成定义

### 2.1 剩余必须完成的五步

1. **M1：真实目标通信路径**。两个物理节点各一个原生 rank/GPU，加载项目构建的固定 NCCL；以 `NET/IB` 和连接证据确认 RDMA，无静默回退。
2. **M2：可信日志**。动态验证 operation/rank/channel/peer 身份、逻辑字节数、可靠 completion 和周期进度的来源与单位；输出可恢复的 Event v2 及必要元数据。
3. **M3：正常与异常对照**。在指定 rank/op/channel 的发送提交前注入可恢复的软件延迟；正常、仅插桩、插桩加延迟三组数值正确，日志差异可解释。
4. **M4：Trigger 与 RCA**。用真实日志实现时间窗口、Trigger、MinOp/MinData 和所选异常场景的状态/依赖分析，输出候选、受影响 rank、证据与缺口。
5. **M5：端到端验收**。正常不误报，异常按预期触发且候选符合注入位置；结果可重跑，构建、transport、丢失及基础开销有证据。

复用已完成的 E01/E02 回归夹具、E03 事件契约、Day14 构建和 Day15 workload，不重做其教学实验。E04 是分析核心，E06 提供真实运行与日志；E05 不再是当前闭环的前置条件。

### 2.2 范围边界

本版复现 Mycroft 的观测、异常触发和依赖驱动 RCA 方法，分析可以在真实运行结束后离线执行。结果属于小规模方法复现，不声称已实现论文的持续在线监控系统或原系统性能。

第一版固定为一个 communicator、每 rank 一个 GPU、Float32 AllReduce、RING/SIMPLE、普通串行调用；必须核对实际选择和计划粒度，不能仅依赖环境变量推断。保留实际多 channel，但不要求兼容所有调度形式。NET/IB 的元数据和单位按实际连接采集，不把 Socket 计数叫 RDMA 计数。

以下内容移出当前执行范围，不安排对应作业或作为验收门槛：完整 P2P GPU 结束标记/managed memory 采集、Graph replay、grouped collectives、多 stream、多 communicator 扩展、多算法/协议、E05 稳定共享内存 ABI/循环缓冲区/独立 reader、慢 reader 压力测试、生产后端以及完整长时间性能矩阵。保留现有代码骨架和回归测试；用户以后明确要求扩展时另行规划。

Socket 仅在 RDMA 环境排障需要时使用，不设独立完整验收阶段，也不替代最终 RDMA 证据。正确性、可信字段、采集丢失检查、少量同环境运行的基础开销对照和单文件结果传输仍是必需项。

### 2.3 完成状态

| 状态 | 定义 |
|---|---|
| 未开始 | 尚未开始该子任务或里程碑 |
| 开发中 | 骨架已建立，但验收尚未全部通过 |
| 已验证 | 自动测试、固定演示、运行证据和用户解释全部通过 |
| 环境阻塞 | 可做部分已完成，但被集群、GPU、网络、权限或镜像条件阻塞 |

本版不沿用旧版 L1/L2 作为当前完成标签，以免把缩减范围误写成旧版实时系统已验收。M4 通过后称“真实轨迹诊断闭环已验证”；M5 通过后称“核心方法复现已验证（双节点 RDMA、离线分析）”。只编译、只有合成事件、只有 Socket 或尚未取得真实报告时均不能标为完成。历史 L1/L2 定义保留在 Git 历史中，E05 与完整实时系统尚未验收。

## 3. 任务执行方式

本节说明里程碑内的子任务如何启动和验收，沿用 `AGENTS.md` 的稳定协作规则；用户最新明确指令优先。

### 3.1 子任务开始方式

Codex 先说明子任务卡，在已获用户授权的范围内直接推进；会改变路线或边界的疑问先确认。任务卡必须包含：

- 本子任务解决的实际问题；
- 输入与输出；
- 只有存在技术歧义、安全风险或验收混淆时才说明排除项；
- 需要补充的最小知识；
- 文件与接口框架；
- 适当的自动检查或验证脚本，以及关键演示命令；
- 预期现象；
- 验收标准；
- 建议 commit message。

### 3.2 分工

默认由 Codex 优先搭建：

- 目录、构建配置和接口声明；
- 测试框架、固定输入、fixtures 和失败断言；
- CLI 参数解析骨架；
- 必要的实验 README 小节和可公开的小型结果样例；
- 空实现、明确的 `TODO` 和错误返回；
- 用户明确要求时，直接实现、补全或修复核心逻辑。

默认由用户参与推导和验收：

- Ring step/chunk 数据传播；
- 因果依赖和延迟传播；
- 三进度量状态转移；
- 事件分组、触发器、`MinOp`、`MinData` 和 RCA；
- NCCL 2.21.5 真实字段确认与插桩更新；
- Crater 页面配置、作业提交、日志查看和最终验收。

以上是默认教学分工，不是代码权限边界。用户可以随时要求 Codex 实现某个函数；实现后仍需讲清数据流、状态变化和验收证据。

### 3.3 逐级提示

用户求助时默认按以下顺序提供帮助：

1. 解释问题和相关知识，不给实现；
2. 给执行流程、数据变化或伪代码；
3. 给局部代码片段；
4. 只有用户明确要求“直接修改”或“给出完整实现”时，才补全核心逻辑。

### 3.4 前进条件

- Codex 执行本地检查并准备实验；用户在 Crater 执行必要作业、观察演示并确认关键结论；
- 用户明确表示验收通过或进入下一步后，才将当前里程碑标记为完成；
- 测试通过但关键状态或证据尚未解释清楚时，继续当前任务，不机械增加考试或独立作业；
- 不允许删除或弱化测试来绕过验收；
- 验收失败只修当前缺口，不重做整条路线。
- 单元测试可以使用合成 fixture，但真实运行里程碑验收必须包含由项目构建的真实 NCCL 2.21.5 产生的证据。
- 真实 NCCL、GPU 或集群条件不可用时标记环境阻塞；不得回退到 E01、E02 或手写 JSONL 并宣称里程碑通过。

## 4. 最小执行节奏

每个子任务先解释它为最终诊断提供哪项证据，再实现、检查并演示。纯逻辑开发在本地完成；需要真实 GPU/NET 的检查合并进必要的 Crater 作业，不为每个小模块单独上传或运行。

里程碑交付必须包含可运行代码、必要检查、准确命令和可观察证据。文件和原始结果由 Codex 整理，用户参与关键判断及平台操作，不逐个下载日志。只更新路线图、当前实验 README 和必要的小型样例，不增加日报。Git 验收边界与 commit/push 授权保持不变。

## 5. 技术路线和环境基线

| 范围 | 技术 |
|---|---|
| E01 | C11、CMake、CTest；必要时由 Python 黑盒测试读取 JSONL |
| E02 | Python 3.10+、类型标注、`unittest`；仅作为确定性 fixture |
| E03—E04 | Python 3.10+、Event v2、JSONL、真实 NCCL 轨迹集成测试 |
| E06 当前采集 | C++17、CUDA/NCCL、预分配进程内缓冲、后台/受控结束后导出；复用已验证的原生 g++ 构建 |
| E05（范围外） | POSIX shared memory、稳定 ABI、独立 reader；仅在用户另行启动时规划 |
| 平台探测 | 已验证的 Crater 平台镜像：Python 3.12、PyTorch 2.6.0a0（NVIDIA 24.12）、CUDA 12.6、V100 |
| NCCL 接入 | NCCL 2.21.5、Ubuntu 22.04、GCC/G++ 11、CUDA devel 镜像 |

Day03/04 当时使用通过 Gloo 与 CUDA 探针的 Crater 平台镜像；Day14/15 原生 NCCL 构建与运行已验证的环境为 CUDA 12.5.82、Ubuntu 22.04.4、G++ 11.4.0。更换环境时运行必要的最小探针，并在对应实验 README 中记录实际版本。Day 14 起所有运行实验使用原生程序直接链接项目构建的 NCCL 2.21.5，不以 PyTorch 自带 NCCL 或模拟器作为插桩与分析验收依据。

## 6. 仓库结构

下列结构是阶段性导航，不要求提前创建空目录；文件只有在直接服务开发、运行、理解或验收时才加入仓库。

```text
mycroft-nccl-reproduction/
├── README.md
├── LICENSE
├── .gitignore
├── CMakeLists.txt
├── pyproject.toml
├── docs/
│   ├── plans/
│   │   └── Mycroft_26日开发路线图.md
│   ├── architecture/
│   ├── crater/
│   └── legacy-nccl-source-study-notes/  # 历史资料，不定义当前路线
├── experiments/
│   ├── e01_ring_dependency/
│   └── e02_progress_state_machine/
├── src/
│   └── mycroft/
│       ├── schema/
│       ├── trace/
│       └── analysis/
├── workloads/
│   └── minimal_allreduce/
├── instrumentation/
│   └── nccl-2.21.5/
│       ├── README.md
│       ├── tracepoints.md
│       └── patches/
├── cluster/
│   └── crater/
│       ├── README.md
│       ├── probes/
│       ├── images/
│       ├── jobs/
│       └── scripts/
├── tests/
│   ├── fixtures/
│   ├── integration/
│   └── expected/
├── results/
│   └── samples/
├── scripts/
└── third_party/
    └── nccl/
```

`third_party/nccl/` 在 Day 13 固定到 NCCL 2.21.5 的明确 tag/commit；Day 14 起编译该版本，M2 在功能分支中加入可复查插桩。实际 NCCL 修改通过 fork/submodule 或可复查 patch 保存；不能在未跟踪的嵌套 Git 工作区中修改后丢失历史。

## 7. 系统主数据流

```text
真实 NCCL 2.21.5
        |
        v
completion/state tracepoint
        |
        v
本进程有界内存 -> 后台/受控停止后导出 Event v2 JSONL -> E04 离线 Trigger/RCA

E01/E02 ---------------------------> 单元回归输入（不作为运行验收）
```

论文实现使用共享内存、异步 agent 和后端分析。本版缩到方法验证：采集仍在真实通信进行时周期执行，分析可以在结果导出后回放各 rank 的历史窗口。保存采集截止时间、丢失和失败状态，不能仅用最终一次快照猜测停滞。每个 rank 的原始单调时间只用于本 rank 时差；跨 rank 按 operation 和已验证单位关联。只有用户以后明确启动扩展时，才另行规划 E05 的持续在线传输。

## 8. 已完成工作与剩余五步

| 历史开发日 / 当前里程碑 | 阶段 | 可提交结果 |
|---|---|---|
| 01 | 仓库基线 | 公开仓库骨架、正式笔记迁移、统一测试入口 |
| 02—04 | Crater 基础 | 页面概念、CPU 双 Pod、单 GPU 基线 |
| 05—07 | E01 | 独立实现的 4-rank Ring 依赖模拟器 |
| 08—10 | E02 | GPU/Proxy/Network 三进度状态机 |
| 11—13 | E03 | Event v1/v2、乱序恢复、NCCL 2.21.5 映射 |
| 14—15（已完成） | E06 基线 | 固定 NCCL 构建、原生双 rank 正确性与真实选择日志 |
| M1（已验证） | 目标路径 | 双物理节点、项目 NCCL、真实 NET/IB 与正确性 |
| M2 | 最小采集 | RING/SIMPLE 的身份、completion、周期进度及必要对端证据 |
| M3 | 异常输入 | 正常/仅插桩/软件延迟对照与真实日志 |
| M4 | 分析核心 | 同一证据集上的 Trigger、MinOp/MinData、最小 RCA |
| M5 | 最终验收 | 双节点 RDMA 方法闭环、基础开销与可重跑证据 |

## 9. 历史任务卡与当前里程碑

以下 Day01—15 保留当时的任务与结果，其涉及后续 Day 编号的安排属于旧版背景；剩余工作以本节 M1—M5 为准。已有 Day16 框架保留复用，不等于 M1/M2 已完成。

### Day 01：创建公开仓库和可运行基线

**实际问题**：为后续每日提交建立干净、可复现、不会泄露本机信息的项目边界。

**Codex 搭建**：

- 创建第 6 节目录骨架；
- 提供 `.gitignore`、根 `CMakeLists.txt`、`pyproject.toml` 和测试入口；
- 迁移当时的范围计划、逐日计划、`NCCL_源码学习日志.md` 和 `notes/` 下正式 Markdown；
- 提供中英双语 README 骨架和 `scripts/check.sh`；
- 不迁移旧 E01、二进制、临时文件和 `参考答案/`。

**用户完成**：

- 决定仓库名和许可证；
- 检查所有迁移文件是否适合公开；
- 创建公开 GitHub 仓库并 push。

**验收与预期现象**：

- 全新 clone 后一个命令能完成空项目构建和测试；
- `git status` 干净；
- GitHub 不含绝对路径、凭据、二进制、参考答案或嵌套 NCCL 工作树。

**建议 commit**：`chore: bootstrap reproducible Mycroft project`

### Day 02：Crater 从零认识平台和单容器 CPU 冒烟作业

**实际问题**：先理解本项目最终运行环境中“提交一个作业”到底创建了什么，并亲自完成一次低资源作业的配置、提交、观察和清理闭环。

**只讲普通用户范围**：Crater、Kubernetes、Volcano、物理 node、Pod、container、image、job、role、replica、mount、environment、job phase、log。

**文件框架**：`docs/crater/Crater平台入门与实验手册.md`。在产生真实探测代码或平台导出配置之前，不预建假模板。

**Codex 搭建**：一份合并手册，包含必要概念、GUI 填写值、预期现象、排错顺序和脱敏规则。

**用户完成**：用自己的话说明镜像与容器、Pod 与物理节点、副本与进程的区别；提交一个 1 CPU、2 GiB、0 GPU 的 Custom Job；查看 phase、Pod、Node 和日志；验证共享目录结果文件；停止或清理作业；导出原始配置并只在仓库中保留脱敏模板。

**验收与预期现象**：日志出现 `hello from crater` 和容器运行信息；共享目录生成内容为 `day02 success` 的结果文件；作业正常结束且用户能找到清理入口。用户能说明为什么“两个容器”不必然是“两台机器”，以及 Target Node Allow List 只限制可选节点、不保证自动分散。

**本日不使用**：GPU、PyTorch DDP、NCCL、RDMA、Target Node Control 和自定义镜像构建。

**建议 commit**：`feat(crater): complete first CPU smoke job`

### Day 03：Crater CPU 双 Pod 与共享目录探测

**实际问题**：不用 GPU 验证 Master/Worker、环境变量、服务发现和 Gloo 集合通信。

**文件框架**：`cluster/crater/probes/cpu_ddp_probe.py`、`tests/python/test_cpu_ddp_probe.py`。

**Codex 搭建**：探测接口、标准库单元测试、GUI 填写值和预期日志。

**核心逻辑、理解与平台操作**：实现 Gloo process group 和 CPU AllReduce；在页面配置 Master 1/Worker 1/GPU 0；通过 GUI 上传单个探针文件并运行。

**验收与预期现象**：`WORLD_SIZE=2`，rank 0/1 共同完成 AllReduce；rank 0 日志输出 `result=3.0`，作业正常退出。即使平台未展示 Worker 日志，该结果也证明 rank 1 已加入通信并贡献数值 2。

**建议 commit**：`feat(crater): validate two-pod CPU control plane`

### Day 04：Crater 单 Pod 单 GPU 基线

**实际问题**：独立确认 GPU、驱动、CUDA 和 PyTorch 镜像可用，为后期多机 NCCL 排除基础环境问题。

**最小补充知识**：GPU request、GPU model、`CUDA_VISIBLE_DEVICES`、宿主驱动与容器 CUDA toolkit 的区别。

**文件框架**：`cluster/crater/probes/gpu_probe.py`、`tests/python/test_gpu_probe.py`。

**Codex 搭建**：GPU 探针接口、本地假模块测试、页面配置值和输出契约。

**核心逻辑、理解与平台操作**：检查 `nvidia-smi`、PyTorch CUDA、device count、GPU tensor 运算和同步；选择一张 V100，资源不足时选择 A100。

**验收与预期现象**：Pod 只看到分配的 GPU；PyTorch tensor 运算正确；记录 GPU、driver、CUDA、PyTorch 和镜像版本；作业数分钟内完成并正常释放资源。

**建议 commit**：`feat(crater): verify single-GPU runtime baseline`

**Gate Crater Basics**：用户能独立填写基础作业字段、提交作业、查看 Pod/日志、停止作业，并解释本地代码如何通过镜像、启动命令和挂载进入 Crater。

### Day 05：E01 数据模型与 ReduceScatter

**实际问题**：用代码证明多个 chunk 怎样沿固定 Ring 被逐步规约，并验证 rank 数量与 chunk 数量是两个独立概念。

**最小补充知识**：4-rank、8-chunk 单 channel Ring，chunk lane、chunk owner、step、同步轮次；结构体、结构体指针和数组的生命周期。

**文件框架**：

```text
experiments/e01_ring_dependency/
├── CMakeLists.txt
├── README.md
├── include/ring_sim.h
├── include/trace_event.h
├── src/main.c
├── src/ring_sim.c
└── tests/test_reduce_scatter.c
```

**Codex 搭建**：输入结构、函数声明、固定 4-rank fixture、失败测试和 CLI 骨架。

**核心逻辑与理解重点**：初始化 chunk、计算发送/接收 rank、为两条 chunk lane 实现三个 ReduceScatter step，并保存每个 step 的状态。

**验收与预期现象**：

- ReduceScatter 后每个 rank 恰好持有两个完成规约的 chunk；
- chunk owner 和直接求和结果一致；
- AddressSanitizer/基本内存检查无越界。

**建议 commit**：`feat(e01): implement ring reduce-scatter steps`

### Day 06：E01 AllGather、最终结果与 JSONL

**实际问题**：让规约结果传播到所有 rank，并把依赖过程变成机器可读事件。

**Codex 搭建**：JSONL writer 接口、schema 断言、最终结果测试和 README 命令模板。

**核心逻辑与理解重点**：实现 AllGather 转发、保存完整结果；在每个 step 发出包含 `op_seq/rank/channel/phase/step/chunk/action/peer/timestamp` 的事件。

**验收与预期现象**：

- 四个 rank 得到相同 AllReduce 结果；
- JSONL 每行可独立解析；
- 正常轨迹中每个 chunk 的 step 单调递增；
- 删除或交换任一必要事件会触发依赖一致性测试失败。

**建议 commit**：`feat(e01): add all-gather and JSONL trace output`

### Day 07：E01 因果依赖、延迟注入与阶段验收

**实际问题**：证明最晚完成的 rank 不一定是主动异常 rank。

**文件扩展**：`src/delay.c`、`tests/test_delay_propagation.c`、`results/samples/e01/`。

**Codex 搭建**：延迟 CLI 参数、根事件 fixture、因果图测试和 E01 README 验收小节。

**核心逻辑与理解重点**：实现直接依赖、同 rank 顺序依赖、延迟传播和根事件标记；不能给所有受影响事件直接增加人为延迟。

**验收与预期现象**：

- 只有一个事件被主动注入；
- 跨 rank 后继事件的开始时间被推迟；
- 数值结果不变；
- 输出能明确区分 injected root 与 affected ranks；
- 用户能依据事件解释“最慢 rank 为什么可能只是受害者”。

**建议 commit**：`feat(e01): trace causal delay propagation`

**Gate E01**：全部通过后，阶段 0—4 从“已讲授”变为“已验证”。

### Day 08：E02 正常三执行者抽象 fixture

**实际问题**：用三个归一化累计量构造有先后约束的确定性发送侧分析 fixture，不声称复刻 NCCL 执行时序。

**最小补充知识**：只区分发送侧的数据准备、请求提交和本地请求完成；具体 NCCL 字段来源留到 Day 13，动态正确性留到 Day 17 的真实 NCCL state log。

**文件框架**：

```text
experiments/e02_progress_state_machine/
├── README.md
├── progress_sim.py
├── cli.py
└── tests/test_normal_progress.py
```

**Codex 搭建**：三个 actor 接口、tick/scheduler 骨架、正常 fixture 和单调性断言。

**核心逻辑与理解重点**：实现抽象 fixture 中 `GPU_ready >= RDMA_transmitted >= RDMA_done` 的正常推进和逻辑步记录；真实字段来源由 Day 13 映射，单位与运行语义由 Day 17 的真实 NCCL state log 验证。

**验收与预期现象**：正常轨迹最终三者相等，任何时刻均不违反单调性和先后不变量。

**建议 commit**：`feat(e02): model normal GPU proxy network progress`

### Day 09：E02 三层合成停滞

**实际问题**：让相同的“collective 变慢”表象产生不同的内部状态证据。

**Codex 搭建**：`FaultSpec`、三类参数化测试和固定输出目录。

**核心逻辑与理解重点**：实现 GPU 未准备、Proxy 未发送、Network 未完成三类延迟/停滞，并保证故障只作用于指定执行者。

**验收与预期现象**：

- GPU 故障：三个进度量共同停止或整体滞后；
- Proxy 故障：`GPU_ready > RDMA_transmitted`；
- 网络完成故障：`RDMA_transmitted > RDMA_done`；
- 每类轨迹仍满足物理先后约束。

**建议 commit**：`feat(e02): inject component-specific progress faults`

### Day 10：E02 状态分类与阶段验收

**实际问题**：从状态轨迹给出“知道什么、还不能断言什么”，避免把相关性当根因。

**文件扩展**：`classify.py`、`tests/test_state_classification.py`，并在 E02 README 记录验收现象。

**Codex 搭建**：论文四类状态的期望表和模糊边界测试。

**核心逻辑与理解重点**：实现未开始、未发送、未送达、GPU 未继续准备的分类，并输出本地原因、远端可能原因和证据不足项。

**验收与预期现象**：给定固定轨迹，分类稳定；接收端证据缺失时不能把发送端唯一判为根因。

**建议 commit**：`feat(e02): classify progress states with evidence bounds`

### Day 11：E03 统一 Event v1

**实际问题**：让 E01 的依赖事件和 E02 的进度状态使用同一数据契约。

**文件框架**：

```text
src/mycroft/schema/
├── __init__.py
├── event.py
├── validate.py
└── version.py
tests/schema/
```

**Codex 搭建**：`Event` 数据类、序列化接口、schema version、合法/非法 fixture 和兼容性测试。

**核心逻辑与理解重点**：确定必填字段、身份字段、operation 字段、progress 字段和依赖字段的验证规则；编写 E01/E02 adapter。

**验收与预期现象**：两类实验输出均能通过同一 validator；缺 rank、op identity 或非法进度关系时给出明确错误。

**建议 commit**：`feat(schema): define versioned unified trace events`

### Day 12：E03 乱序、多 communicator、多 operation 恢复

**实际问题**：不能依赖文件到达顺序判断事件属于哪次 collective。

**文件框架**：`src/mycroft/trace/group.py`、`timeline.py`、`tests/trace/`。

**Codex 搭建**：两组 communicator、多个 operation/channel 的打乱 fixture 和期望分组。

**核心逻辑与理解重点**：定义 operation/flow key，完成乱序分组、去重、缺口报告和时间线恢复。

**验收与预期现象**：随机打乱输入多次，恢复结果完全一致；重复事件不会重复计算；缺事件被标记而不是静默忽略。

**建议 commit**：`feat(trace): recover timelines independently of arrival order`

### Day 13：E03 NCCL 2.21.5 字段来源和候选插桩点

**实际问题**：把模拟字段连接到指定版本源码，避免使用 master 字段替代事实。

**最小补充知识**：只追踪 communicator/rank、operation sequence、channel、send/recv proxy progress、connection/QP 身份相关路径。

**文件框架**：

```text
instrumentation/nccl-2.21.5/
├── README.md
└── tracepoints.md
docs/architecture/event-field-sources.md
third_party/nccl/              # pin 到 NCCL 2.21.5 的明确 tag/commit
```

**Codex 搭建**：字段来源表模板和源码引用格式。

**核心逻辑与理解重点**：添加官方 NCCL 源码引用并固定到 2.21.5 的明确 tag/commit（不沿用当前未固定的 master 工作区）；逐项确认 `IP/comm_id/Gid/GPU_id/channel_id/QP_id/op_seq/msg_size` 的创建者、结构、生命周期、跨 rank 一致性和唯一性；记录三个进度量的候选位置及“已确认/待确认”。

**验收与预期现象**：仓库记录的 tag/commit 可复查且工作区干净；每项结论都指向该版本的文件、函数、结构体成员或明确的数据流；所有猜测均标为待确认。

**建议 commit**：`docs(e03): map trace identity to NCCL 2.21.5`

**Gate E03**：Event v1 保持冻结并继续可读。进入 Day 14 前发现 v1 缺少真实时间、completion 和字节数，因此按本规则发布兼容 Event v2；不能静默改变 v1 语义。

### Day 14：E06 可复现 NCCL 2.21.5 构建与加载基线

**实际问题**：先证明项目能稳定构建并加载指定版本的真实 NCCL，后续所有运行实验都建立在这份二进制上。

**输入与输出**：输入为固定到官方 `v2.21.5-1` 的源码；输出为可复现的构建脚本、`libnccl.so`、版本清单和最小动态加载检查。

**文件框架**：

```text
cluster/crater/images/Dockerfile
cluster/crater/scripts/build_nccl.sh
instrumentation/nccl-2.21.5/README.md
third_party/nccl/
```

**核心逻辑与理解重点**：使用 CUDA devel 环境编译未插桩 NCCL；记录 tag、commit、CUDA、编译器和构建参数；通过 `ncclGetVersion`、动态链接路径和 smoke program 同时证明实际加载的是项目构建版本。


**验收与预期现象**：真实 `libnccl.so` 构建成功；smoke program 能加载并返回 2.21.5；动态链接检查指向项目产物。若缺 CUDA devel 或构建资源，标记环境阻塞，不能用模拟器代替验收。

**执行状态**：真实构建与加载基线已完成，用户已明确进入 Day 15。Crater 在 Ubuntu 22.04.4、G++ 11.4.0、CUDA 12.5.82 下从固定且干净的源码完成未插桩 NCCL 构建；临时与持久化库的 `ncclGetVersion` 均返回 `22105`，实际加载路径与指定产物一致，构建和持久化验证退出码均为 0。已保留工具链、构建参数、ELF SONAME 和库 SHA256 的脱敏证据。本地 WSL 仍无 CUDA devel；提供的 Dockerfile 尚未实际构建。此验收不包含 GPU collective。命令、样例与限制见 [`instrumentation/nccl-2.21.5/README.md`](../../instrumentation/nccl-2.21.5/README.md)。

**建议分支/commit**：从 Day 14 起使用功能分支；`build(e06): reproduce NCCL 2.21.5 toolchain`

### Day 15：E06 原生同机双 rank AllReduce 基线

**实际问题**：绕开 PyTorch 自带 NCCL，证明项目构建的 NCCL 能执行真实 collective。

**文件框架**：

```text
workloads/minimal_allreduce/
├── CMakeLists.txt
├── include/bootstrap.h
├── src/bootstrap.cpp
├── src/main.cpp
└── tests/
```

**核心逻辑与理解重点**：rank 0 创建并分发 `ncclUniqueId`；两个原生进程各绑定一张 GPU，初始化 communicator，在同一物理节点运行多种消息大小和连续多个 AllReduce，并同步校验结果。

**验收与预期现象**：两个 rank 数值结果正确；至少三个连续 operation 完成；`ldd` 或等价证据明确指向 Day 14 构建的 NCCL 2.21.5；记录算法、协议和 transport 日志。拿不到同机双 GPU 时标记环境阻塞，不退回 E01/E02 作为验收。

**执行状态**：已完成 API 返回、GPU 等待与 Ring 数据流讲解，用户明确授权补全实现。原生 C++ workload 提供文件 bootstrap、GPU 绑定、双进程启动、AllReduce 提交、stream 同步和全部元素校验；消息大小为 16 B、16 KiB、4 MiB，各连续三次，输入随应用 operation 序号变化以发现旧结果。Crater CUDA 12.5.82 下使用 g++ 编译，并在同机双 V100 上运行；首轮两个 rank 各九次操作全部正确，库版本、路径及 hash 与 Day14 一致，退出码均为 0，连接为 P2P/CUMEM。应用阶段标记尚不是 NCCL 插桩事件。独立 TRACE 补证结果见下一段；完整仓库检查通过，包括八十六项 Python、五项 C++ 测试及两组真实日志摘录的离线复查。程序、命令、证据与限制见 [`workloads/minimal_allreduce/README.md`](../../workloads/minimal_allreduce/README.md)。

**已完成的 TRACE 补证**：独立 `TRACE=1` 库已在 Crater 从同一固定且干净的源码完成构建、加载及双 GPU 运行；构建、持久化加载、作业和两个 rank 的退出码均为 0。构建验证日志给出的库 SHA256 为 `e4437dd0b48e3ab426b17b394043a0d162f1c9040b32bcf34b5b866e6286132a`，与 workload 环境记录一致；使用这个独立构建 hash 的离线复核已通过。rank0 的九次选择日志全部位于对应 API 调用区间：16 B 使用 RING/LL、1 channel；16 KiB 使用 RING/LL、2 channels；4 MiB 使用 RING/SIMPLE、4 channels，均为 CBDColl 调度路径，连接记录为 P2P/CUMEM。上游该日志只由 rank0 输出，不把字段补造给 rank1；两个 rank 各九次结果均正确。二十一项 workload 辅助测试、二十二项 NCCL 构建/包 guard 已通过。真实技术证据已齐全，Day15 已提交并推送；后续任务按 M1—M5 推进。

**建议 commit**：`feat(e06): run native two-rank NCCL baseline`

**提交状态**：公开样例保留两轮真实实验的脱敏摘录，并用同一离线核对脚本复查；未提交原始日志、库、上传包或个人旧学习笔记。Day15 已提交为 `4638847`，推送到 `origin/day15/native-allreduce-baseline`。随后创建 `day16/operation-completion`，Day16 修改独立推进。

### M1：目标 NET/RDMA 路径验证（已验证）

**目标与输入输出**：复用固定源码、已验证 CUDA devel 环境和 Day15 原生 workload，在两个物理节点各运行一个 rank/GPU。输出正确性、项目库来源、实际节点分布、transport 和连接证据。

**实现边界**：准备最小双节点启动/unique ID 交换入口和一次资源探针，不重新开发完整平台教学。核对 GPU、IB 设备、link layer、memlock、两端通信条件；确认 `NET/IB` 且未静默回退。明确 Master/Worker 各一个 Pod/进程，分别落在不同物理节点；资源配置和路径从实际环境取得。优先复用已有构建产物；修改或需要 RDMA 开发依赖时再构建新库，保存来源。

**最小知识与用户参与**：两个 Pod 不等于两个节点；用户能根据节点证据和 NET 日志判断走了哪条路径。Codex 准备命令、打包和解释，用户在 Crater 执行并复制一个结果包。

**文件/接口**：复用原生 workload 和有界结果打包；`run_m1.py` 每 Pod 启动一个 rank，`rdma_probe.py` 预检环境，`verify_m1.py` 核对真实来源、身份、网络和数值。新增显式单 GPU 模式，两侧通过同一新共享目录交换 ID/就绪状态，rank0 自动打一个结果包。运行前提、双 Role 配置和准确命令见 [M1 实验 README](../../workloads/minimal_allreduce/M1_README.md)。

**验收**：两个 rank 所有元素正确、退出码为 0；项目 NCCL 版本/加载路径/hash 一致；两个真实物理节点；明确 NET/IB 和有效连接。缺少 RDMA 或节点资源时保留诊断包、标记环境阻塞；可做本地开发，但不能拿 Day15 P2P 或 Socket 顶替验收。

**当前状态**：已验证；证据齐全后用户明确进入 M2。首轮缺少 `libibverbs.so.1`，停在资源预检；用户通过 Crater 派生镜像补齐运行库后，清理远端旧目标并复用了 r1 名称。重试归档 SHA256 为 `22262421a9dfdb765ac3e3e0fc64a7c0d5c6e6d336e25bd1f29e045f24feb59d`；两侧 probe/build/workload 均退出 0，各完成九次 AllReduce，项目 TRACE 库 hash 与独立构建一致，两个不同物理节点/不同 GPU、同一 communicator、NET/IB 双向连接证据齐全。rank0 的实际计划为 RING/SIMPLE：16 B 和 16 KiB 各使用 1 channel，4 MiB 使用 2 channels。原作业整体退出 1，原因是应用的分段 stdout 输出被完整 Proxy TRACE 插入，核对器误报 `operation stages differ`。已修复离线核对器，按原始字节片段恢复 rank0 两行、rank1 五行，18 项数值检查及完整技术复核通过；保留原始日志和失败状态，不要求重跑 GPU。两侧 memlock 仍为 64 KiB，但本轮实际通信未被阻塞，不新增平台配置前置任务。已把未来应用记录改为整行一次 `fwrite`，该 C++ 输出修复尚未在 Crater 编译运行。本次全仓库检查通过（102 项 Python、5 项 C++），两组 Day15 真实摘录与一组 M1 双节点 NET/IB 真实摘录均通过离线复核。公开 M1 摘录位于 `results/samples/e06/m1/`，原失败状态与恢复行计数保留，脱敏与筛选规则可复查。原运行与重试分别保存在忽略的 `.build/m1-download/`，重试离线报告为 `.build/m1-download/r1-retry/offline-recheck.txt`。M1 已提交并推送到 main（`f333d6d`）；用户随后明确授权搭建 M2 框架，当前推进 M2。

### M2：最小 completion/state log

**目标与输入输出**：在 M1 已验证的普通 RING/SIMPLE NET 路径采集真实记录，输出逐 rank Event v2、身份/连接元数据、采集范围与丢失统计。

**最小实现**：

- T1/T2 关联 `commHash`、历史 collective 序号候选、rank、channel、peer/direction 与逻辑字节数。两个 rank 连续多个 AllReduce 动态对账，不直接沿用 INFO 的 `comm->opCount`。
- 在 `sendProxyProgress()` 派生 readiness，采样 `transmitted/done/nsteps`，统一 step/slice 单位并验证单调性。只为已验证的 IB 路径赋予 RDMA 含义。
- 仅补充解释所选故障所需的接收侧/对端状态；发送与接收同名字段分别定义。若证据不能区分本地异常和远端阻塞，输出并列候选和证据缺口。
- 在真实执行 stream 上验证整个本地 collective 的完成观测，并检查 abort/error。第一版限定单操作计划，必须动态证明这一条件；不支持的计划形式显式标记，不伪造 completion。Proxy request、API 返回和 `workFifoDone` 都不能直接替代整个操作完成。
- 开始和结束观测使用同一进程单调时钟，记录具体观测点和可能的调度/观测延迟；不声称精确 GPU 执行耗时，不比较跨主机原始纳秒值。
- 写入初始化时预分配的有界内存，后台批量导出或受控停止后导出；热循环不做阻塞 I/O/JSON 格式化。采集满时报丢失，不等待 reader。进程崩溃且内存尚未导出时明确记录证据缺失，不能因此宣称通信失败。

**文件/接口**：复用 `instrumentation/nccl-2.21.5/day16/` 接口、Event v2、`tracepoints.md` 与结果打包工具；patch 在主仓库可复查保存，上游固定源码保持干净。不增加 GPU work 字段、完整 P2P 采集器、Graph 序号或共享内存 ABI。

**验收与预期现象**：真实连续 AllReduce 至少三次，每 rank 每操作恰有一条 completion；身份/大小/channel 对账通过。合适消息量或明确记录的采样周期下，至少一个真实操作得到两条 state log；采样量不因通信停滞而停止。completion 观测有真实执行依据，所有元素正确，采集丢失为零或足以明确拒绝该次诊断。源码位置、支持范围和真实证据保存在同一实验 README/小型样例中。

**当前已知结论**：Day16 静态追踪确认 CBDColl 序号在 channel 循环外产生；GPU work 没有完整 operation 身份，work/FIFO slot/plan 的粒度不同；普通无 Proxy 路径也可能推进历史计数，但无 q 条目可采集；`workFifoDone` 在通信执行前更新。Graph 与 P2P 覆盖问题保留为限制，本版不实现其扩展。M2 开发中：复用 Day16 目录实现预分配有界记录、多生产者独立槽写入、周期到期保留未变化快照、停止后统计和导出；发送/接收原始计数及身份/plan 元数据均保存。step→slice、关联缺失/重复、累计倒退和丢失检查已可在 CPU 上运行。导出 manifest 明确标为 adapter/completion 未验证且不具备真实诊断资格，不能把手写 fixture 作为运行证据。NCCL adapter/patch 已接入身份、实际 channel/peer、Proxy 状态、同进程 CLOCK_MONOTONIC、实际 launch stream 的 CUDA event 与 abort/error 观测，并确认 Proxy 线程已 join 才导出。独立构建保留固定原源码 archive、patch/新增源码/产物 hash；双 rank runner 复用 M1，真实核对对账构建、加载、数值、raw/Event 及应用阶段时间。上传包和成功/失败单文件结果交付入口已有代码；当前下一项为复用已构建插桩库进行同范围双节点 GPU 采集验证。本地已通过 129 项 Python、5 项 C++ 及历史真实日志离线回归，其中 M2 的 32 项检查包含源码包/patch 准备及核对负例；这些检查与真实运行证据分别判断。2026-10-03 收到 Crater 的 `m2-build-r1.tar.gz`，归档 SHA256 为 `b6af08d24ae7aa940af17ae0837cbdebf2d4c870bf93a7873f6b9132bea8315e`；Ubuntu 22.04.4、G++ 11.4.0、CUDA 12.5.82 下插桩库构建及加载检查退出 0，版本为 `22105`，三个 M2 接口导出。离线重新应用上传包的 patch，原源码、五项插桩输入和九项准备后源码 hash 均与构建记录一致；结果包无缺失文件。新库 SHA256 为 `485db701953f4b1dd499bdfb5de8abe285646441c7bfaa35290fdd6f3e67ed4e`，实际库留在远端供 GPU runner 再次校验；本地未加载该二进制。原包与离线报告保存在忽略的 `.build/m2-download/`。构建包明确 `gpu_execution=NOT_RUN`。随后收到首轮 `m2-capture-r1.tar.gz`，归档 SHA256 为 `b0f9e622b371525969eed7ad3221cec53281c07e5aed8896e926800e0d58125a`：两侧 probe、workload 编译和运行均退出 0，18 项结果正确，所加载库与已核对构建一致；但两侧 DMI 与 kernel boot 标识均相同，属于同一主机上的两个不同 GPU/Pod，故物理双节点检查失败，原作业退出 1。保留原始结果与失败结论；独立调用逐 rank capture 检查后，身份/channel/peer、九条 completion、raw/Event、单位与单调性、最终状态及零丢失均通过，两侧 send/recv 样本分别为 2171/2249 和 224/286，周期中间窗口分别为 2147、200；实际 channel 数为 1,1,1,1,1,1,2,2,2，跨 rank 身份一致。这些仅是同主机 NET/IB 采集证据，不能冒充双物理节点验收。原包及离线报告保存在 `.build/m2-download/capture-r1/`。M2 未验收；下一步修正物理节点调度后复用现有上传包和已构建库，使用新的采集编号，无需重复构建。

**当前调度与运行阻碍**：用户确认 PyTorch 双 Role 的节点控制由整个作业共用；Custom 单机批处理支持 RDMA 与节点白名单。已用两个 Custom 作业分别指定不同节点，共用 `m2-capture-r2` 与已构建的 `m2-build-r1/nccl/`。返回归档 SHA256 为 `cb644fc6b2880bdc7730a59a780edd9c51a35ff647ae0cafb18e939ea6be6f0a`，本地核对一致：两侧 run_id、构建来源和记录的库 hash 一致，probe 与 workload 编译退出 0，但两侧 native workload 均退出 -11（SIGSEGV）。两侧已有 `comm_init_begin`，没有 `comm_init_return` 或 `m2_capture=STARTED`；首个明确错误均为 `ibv_create_cq failed with error Cannot allocate memory`，十四项 capture 文件尚未生成，M2 核对未执行。两侧 DMI 标识均不可读取（null，不是相同硬件标识），kernel boot 标识不同，不沿用 r1 的同主机判断。原始证据和离线报告保存在 `.build/m2-download/capture-r2/`。

用户确认 r2 两侧 Custom Shell 均为普通用户，且页面支持 root；随后按最小对照两侧改用 root，复用现有包和库，以新编号 r3 运行。Crater [官方 RDMA 文档](https://raids-lab.github.io/crater/zh/docs/admin/more/rdma/)记录的锁内存权限和运行时限制问题是本次排查依据。root 身份不保证有效 `IPC_LOCK` 权限；结果包未记录有效 capabilities，且 r2 无崩溃回溯，ENOMEM 的具体原因及 SIGSEGV 位置仍未确定，保留失败证据，不把推断写成根因已证实。

**当前真实证据（r3）**：2026-10-03 返回 `m2-capture-r3.tar.gz`，本地 SHA256 为 `5dd367f9e0742cf32d1cad7dad22728cbee1602d808bf29fd5fe9586ddb02ed7`（未取得本轮远端 hash，未宣称两端 hash 对比通过）。包内 33 项必要文件齐全；两侧 run_id、保存的构建来源及记录的库 hash 与已核对构建一致。两侧 probe/build/workload 和作业整体均退出 0，初始化正常返回、采集正常启动，CQ ENOMEM 与 SIGSEGV 均未出现。两侧 DMI 标识和 kernel boot 标识均不同，双物理节点证据通过；各侧 boot 标识与 r2 对应一致，memlock soft/hard 仍为 64 KiB，因此本轮成功不能解释成提高了 memlock 上限。两侧各九次 AllReduce，18 项数值检查正确，每侧九条 completion；身份、channel/peer、大小、单操作 plan、真实执行 stream、raw/Event 对账、单位、单调性、最终状态和零丢失均通过。rank0 send/recv 样本为 2996/3026，周期中间窗口 2972；rank1 为 341/424，周期中间窗口 317。独立运行本地 `verify_capture.py` 退出 0，输出与包内分析逐字一致：`m2_technical_checks=PASS`、`capture_verification=PASS`。completion 仍只是本地整体 collective 的 CPU 完成观测，readiness 是观察到的可提交 slice 边界，不声称精确 GPU 时间或硬件故障证明。原归档位于 `.build/m2-download/`，解包结果和离线报告位于 `.build/m2-download/capture-r3/`。当前技术证据已通过，完整 M2 验收待用户复核/明确进入下一步；不新增重跑或构建任务，研究范围和验收标准保持不变。

### M3：真实软件延迟与对照证据

**目标与输入输出**：在 M2 已验证路径注入一个位置已知的延迟，输出未插桩正常、仅插桩正常、插桩加延迟三组同环境证据。

**核心实现**：默认关闭开关，按 rank/op/channel 精确选择。在 `ncclNet->isend` 前用单调时钟 release deadline 延迟目标请求提交，同时继续 Proxy progress 与周期采样，禁止阻塞 sleep 冻结整个 Proxy。延迟结束后通信恢复。此用例只表示真实 NCCL 中的软件延迟，不等同物理 GPU/NIC/RDMA 故障。

**验收**：三组数值均正确；仅目标路径主动延迟，其他 rank 的受阻变化可以沿连接解释；关闭开关恢复正常。保存足够覆盖正常与延迟区间的周期记录、completion、退出码和采集截止时间。最终文件归档不能掩盖中途采样窗口，也不能因为作业结束后有 completion 就抹去停滞区间。

**文件与命令**：复用同一 NCCL patch、runner 和核对入口，正常/延迟用例在一次受控作业中连续执行并自动归档；命令在 runner 实现后写入实验 README，不新增独立日报。

### M4：Trigger、MinOp/MinData 与最小 RCA

**目标与输入输出**：分析 M2—M3 的真实轨迹，输出异常时间窗口、类型、根因候选、受影响 rank、证据行和缺口。分析器只接收日志/元数据，注入配置仅用于核对期望答案，不作为诊断输入。

**核心实现**：

- 窗口接收明确采集截止时间/水位；有活跃 state log 而没有 completion 时触发停滞；按同 rank 的观测 duration、吞吐和操作间隔与正常基线比较，阈值可配置。
- `CheckMinOp` 先定位 operation 落后者；只有 operation 对齐且字段单位一致时才比较 `CheckMinData`，保留并列与缺失记录。
- 根据发送侧状态及必要对端证据恢复所选用例的依赖，区分主动异常候选与受影响者。最晚 rank 不自动等于根因；软件注入证据不支持绝对硬件故障结论。
- 丢事件、采集崩溃、不支持路径和无法对齐的身份降低证据充分性，不自动转成通信故障。

**文件/接口**：在现有 `src/mycroft/analysis/` 实现 window、trigger、MinOp/MinData、rca、evidence 和最小 CLI。只增加能验证边界与真实证据回放的检查，E01/E02 fixture 不能替代真实验收。

**验收**：正常真实轨迹不触发；延迟轨迹在预期中间窗口触发，候选与注入方向/位置相符或明确输出证据不足；改变输入到达顺序不改变候选。用户能依据日志解释“为什么该路径主动等待、其他 rank 为什么受影响”。通过后标记“真实轨迹诊断闭环已验证”。

### M5：核心方法端到端验收

**目标**：用固定且可重跑的命令，在两个物理节点的真实 RDMA 路径验证完整链路。复用已有可信结果，不为阶段编号重复运行；只有修改、失败或证据缺口才补跑。

**最低矩阵**：未插桩正常、仅插桩正常、插桩加软件延迟。同环境少量重复运行，记录观测定义、运行时间、事件数、丢失数和异常报告；不要求长时间压力、慢 reader 或生产性能结论。

**通过条件**：

1. 三组所有元素与退出码正确，实际加载项目 NCCL 2.21.5 且构建来源可追溯；
2. 两个物理节点、NET/IB、连接元数据与未回退均有证据；
3. operation/channel 身份、消息大小、进度单位和完成观测通过真实核对；
4. 正常不误报，异常按预期触发，RCA 给出符合注入位置的候选、受影响者和证据边界；
5. 用于诊断的日志无丢失；有缺失时必须拒绝或明确不足，补跑后再验收；
6. 基础插桩开销有同环境记录，不把小样本外推为生产性能；
7. 用户观察演示并能解释证据，从干净 checkout 按文档可重跑。

**结果交付**：每次作业成功或失败均保存已有日志与状态，自动生成一个结果压缩包。Codex 提前创建本地接收目录并给出 Windows 路径，用户下载一个文件，Codex 解包核对并讲解。

**完成标签**：“核心方法复现已验证（双节点 RDMA、离线分析）”。本版不宣称 E05、持续实时分析、Graph/完整 P2P 兼容、论文规模或原系统开销已复现。Git commit/push 仍由用户明确授权。

## 10. 当前验收矩阵

| 里程碑 | 必须回答的问题 | 必须存在的证据 |
|---|---|---|
| M1 | 真实 NCCL 在哪两个节点、通过哪条网络执行？ | 节点分布、库来源、NET/IB/连接、正确性和退出码 |
| M2 | 每个字段与 completion 分别代表什么？ | 源码位置、真实身份/大小对账、进度单位、完成观测、采集丢失统计 |
| M3 | 哪个路径主动延迟，其他路径如何受影响？ | 三组对照、目标注入位置、中间窗口的 state/completion 与数值结果 |
| M4 | 为什么触发，为什么这些 rank/path 是候选？ | 正常不误报、异常触发、MinOp/MinData、状态/对端依赖证据与缺口 |
| M5 | 从干净环境能否重现真实 RDMA 诊断闭环？ | 完整命令、双节点 NET/IB、真实 Event v2、RCA、基础开销和无丢失证据 |

## 11. GitHub 工作流

### 11.1 Day01—13 历史工作流（不作为当前执行规则）

- 每个验收通过的开发日至少形成一个通过测试的 commit，Day 01—Day 13 直接提交 `main`；
- commit 前运行统一检查脚本；
- 不提交失败测试、WIP、二进制或大型输出；
- commit 和 push 必须由用户明确发出指令；push 默认由用户执行；
- commit 前核对暂存边界，不把下一 Day 的工作混入当前提交；
- 推荐格式：`type(scope): subject`。

### 11.2 Day14 起的真实 NCCL 与 M1—M5

- Day 14 起涉及构建、workload、E05/E06 底层运行时或 NCCL 源码时使用短分支；Crater 配置或探针是否单独建分支由任务风险决定；
- 分支只覆盖一个里程碑或明确子任务；
- 用户完成自验后创建 PR；
- PR 描述包含目的、测试、预期现象、实际现象和风险；
- 合并前保持测试通过，不用修改测试来迎合实现。

### 11.3 公开仓库安全

- 不提交 GitHub token、Harbor 凭据、集群 cookie、SSH key；
- 不提交 Crater 内部域名、真实用户名、个人目录或节点名到公共模板；
- 实验配置使用 `<USER_HOME>`、`<MASTER_NODE>`、`<IMAGE>` 等占位符；
- 大型轨迹保留在本地/集群，只提交小型可复查 sample；
- 每次 push 前检查 `git diff --cached`。

## 12. 环境阻塞和计划变更

出现以下情况时标为环境阻塞：

- Crater 无法申请两个同型号 GPU；
- 两个 Pod 无法分布到不同物理节点；
- RDMA 开关、IB 设备或 memlock 不可用；
- CUDA devel 镜像无法构建 NCCL 2.21.5；
- 用户没有查看 Pod/Node/log 的权限。

阻塞时必须：

1. 保存最小诊断结果和错误日志；
2. 明确已经验证到哪一层；
3. 区分代码错误、配置错误和平台条件；
4. 不把 Socket 成功写成 RDMA 成功；
5. Day14 起不把模拟器、手写 Event 或仅单元测试成功写成真实里程碑或核心复现完成；
6. 等待用户或管理员处理后继续同一任务。

任何范围、验收或顺序变化都必须在本路线图中留下简短变更记录，并在得到用户确认后执行；稳定协作规则的变化更新 `AGENTS.md`。

## 13. Day 01 历史迁移边界

Day 01 只迁移了正式 Markdown，并排除了旧 E01、参考答案、临时笔记、嵌套 NCCL 工作树、本机绝对路径和构建产物。本节仅保留这一历史决策；后续文件是否加入仓库由 `AGENTS.md` 的最小必要原则和公开安全规则判断。

## 14. 依据

- [Mycroft SOSP 2025 论文](https://minlanyu.seas.harvard.edu/writeup/sosp25.pdf)
- [Mycroft arXiv 页面](https://arxiv.org/abs/2509.03018)
- [NVIDIA NCCL 官方仓库](https://github.com/NVIDIA/nccl)
- [NVIDIA nccl-tests 官方仓库](https://github.com/NVIDIA/nccl-tests)
- [Crater 官方仓库](https://github.com/raids-lab/crater)
- [Crater Volcano 集成说明](https://raids-lab.github.io/crater/zh/docs/admin/deployment/volcano/)
- [Crater RDMA 说明](https://raids-lab.github.io/crater/zh/docs/admin/more/rdma/)
- [Volcano PyTorch 插件说明](https://volcano.sh/docs/userguide/user_guide_how_to_use_pytorch_plugin/)

## 15. 计划变更记录

用户已确认原 v0.1 计划并启动 Day 01。随后用户提出先熟悉开发与运行工具，再进入 E01 知识和核心实现；本次 v0.2 调整已经用户明确确认。

v0.2 只改变顺序，不改变 E01—E06 的范围、总开发日数量或最终 L2 验收：

- 原 Day 18—20 的 Crater 页面、CPU 双 Pod、单 GPU 基线前移为 Day 02—04；
- 原 E01—E05 顺延为 Day 05—20；
- 双物理节点 Socket/RDMA 保持在 Day 21—22；
- E06 保持在 Day 23—26；
- Day 01—17 直接提交 `main`，Day 18 起对 E05/E06 底层代码使用分支和 PR。


v0.3 在不改变 E01—E06、26 个开发日和 L2 验收目标的前提下：

- 建立 `AGENTS.md` 作为稳定规则入口；
- 从工作树删除早期 L1/学习路线文档，由 Git 历史保留原始内容，并将本文确立为当前执行路线；
- 统一结对分工、最小文档、Git 授权、实际 Crater 环境和动态状态规则。


v0.4 根据用户要求重排 Day 14—26，并保持总开发日数量和最终范围不变：

- Day 14—17 前置项目 NCCL 2.21.5 构建、原生双 rank workload、completion log 和周期 state log；
- Day 18—20 用真实事件完成共享内存/reader，并在真实 NCCL 控制流中生成确定性软件延迟证据；
- Day 21—23 才基于真实轨迹实现 Trigger、MinOp/MinData 和 RCA，L1 不再接受纯合成数据；
- Day 24—26 完成插桩版 NCCL 的双节点 Socket、RDMA 和 L2；
- E01、E02 和手写 Event 保留为单元回归，不再作为后续实验验收证据；
- 从 Day 14 起底层工作使用功能分支和 PR。


v0.5（2026-10-02）根据用户“修订计划文档，只执行必须要做的几步”的明确要求，替代旧版剩余路线：

- 保留 Day01—15 的历史成果和 Day16 未完成骨架，不重做已通过实验，不把骨架升级为已验收。
- 将原 Day16—26 收缩为 M1 目标路径、M2 可信采集、M3 异常对照、M4 检测/RCA、M5 端到端验收五个里程碑；不再承诺 26 个开发日。
- 前移真实双节点 RDMA 路径确认，解决 Day15 P2P 与论文 NET/Proxy 观测之间的衔接；Socket 只用于必要排障。
- 缩到普通串行 Float32 AllReduce、单 communicator、RING/SIMPLE；取消本版 Graph/grouped/多 stream、多协议、完整 P2P GPU 采集的必做门槛。
- E05 共享内存 ABI、循环缓冲区、独立 reader 及压力测试移出当前执行范围；使用进程内有界采集和文件导出，先验证真实轨迹上的离线诊断方法。
- 保留真实正确性、身份/单位/完成语义、必要对端证据、丢失统计、基础开销对照和单文件结果交付。
- 当前完成标签改为“核心方法复现已验证（双节点 RDMA、离线分析）”；不把范围缩减后的结果写成旧版 L2 或完整实时系统已完成。
- 原计划的生产采集工程与兼容性扩展仍有用途，但不再作为当前核心方法复现的前置条件；以后需要时由用户明确启动。
- 按用户指令删除失效的 Day14 交接、`runtime/README.md` 和重复的 `notes/nccl/`；历史学习资料统一保留在 `docs/legacy-nccl-source-study-notes/`。
- 同步 AGENTS、实验 README 与代码 TODO：取消固定逐日推进和旧 L2/E05 目标，TODO 对应 M2，历史任务与下载流程明确标注；该次统一完成时下一步是 M1，当时尚无双节点入口或真实采集。

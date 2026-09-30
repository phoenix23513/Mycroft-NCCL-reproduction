# Mycroft 26 日开发路线图

- 版本：v0.4（Day 14 起真实 NCCL 优先）
- 当前规则：仓库根目录 `AGENTS.md`
- 历史版本：v0.1/v0.2/v0.3 的完整内容保留在 Git 历史中，关键变更见第 15 节
- NCCL 目标版本：2.21.5
- 最终目标：完成 L2 NCCL 插桩原型，并在 Crater 多机 GPU 环境中完成真实验证
- 工作强度：每个开发日 4—5 个专注小时，不绑定自然日期
- 预计规模：26 个开发日；集群排队、权限申请和平台故障等待不计入开发日
- 当前状态：Day 13 与 Event v2 契约纠错已完成；v0.4 已将真实 NCCL 构建前移，下一步是 Day 14 构建与加载基线，尚未执行任何 Day 14 实验

## 1. 计划要解决的问题

本项目把 Mycroft 复现作为开发主线，不再把“先系统学完 NCCL”作为启动条件。每个阶段都从一个可运行功能出发；只有实现遇到实际缺口时，才补充解决该缺口所需的 NCCL、C/C++、Python、Docker、PyTorch、Kubernetes 或 RDMA 知识。

每天只完成一个有明确边界的子任务，并在验收通过后形成至少一个可运行 commit。解释过、阅读过或通过自测均不等于完成；只有代码、自动测试、固定演示和运行证据一致时，子任务才算通过。

从 Day 14 起改变执行原则：先构建、运行和插桩真实 NCCL 2.21.5，再实现消费这些真实轨迹的分析器。E01、E02 和手写 Event 仍可作为确定性单元回归，但不能再单独充当任何实验日、L1 或 L2 的验收证据。

本文是当前唯一的开发路线与动态进度文档：第 2 节定义 E01—E06 范围，其余章节把范围展开为每日任务。稳定的协作、Git、文档和安全规则统一由 `AGENTS.md` 管理；历史计划不再约束当前执行。

## 2. 当前项目范围与完成定义

### 2.1 必须完成

1. **E01：4-rank Ring 依赖模拟器**
   - 正确完成 ReduceScatter 和 AllGather；
   - 输出 JSONL 事件；
   - 注入单点延迟并展示依赖传播；
   - 区分主动异常 rank 和后续受阻 rank。
2. **E02：GPU/Proxy/Network 三段状态机**
   - 模拟 `GPU_ready`、`RDMA_transmitted`、`RDMA_done`；
   - 覆盖 GPU、Proxy 和网络延迟；
   - 产生可供分析器使用的归一化状态轨迹；
   - 只作为确定性分析 fixture，不证明真实 NCCL 字段映射或执行时序。
3. **E03：统一事件身份与 NCCL 2.21.5 源码映射**
   - 混合多 communicator、多 operation、多 channel 后仍能正确分组；
   - 每个真实身份字段和候选插桩点均有 NCCL 2.21.5 源码依据。
4. **E04：Mycroft 分析器 MVP**
   - 停滞触发、慢速触发、`MinOp`、`MinData`；
   - 根因候选、受影响 rank、证据链和置信边界；
   - 真实 NCCL 正常/延迟轨迹的集成验收，以及仅用于边界回归的固定 fixture。
5. **E05：共享内存循环缓冲区与独立 reader**
   - 固定事件 ABI；
   - 写入方不等待 reader；
   - 覆盖、丢事件和版本错误均可观察；
   - reader 将真实 NCCL 事件输出为可直接交给 E04 的 Event v2。
6. **Crater 平台上手和多机基线**
   - 理解普通用户需要操作的页面与对象；
   - CPU 双 Pod、单 GPU、双节点 Socket、双节点 RDMA 逐层验证。
7. **E06：接入 NCCL 2.21.5**
   - Day 14 编译未插桩基线，Day 16 起持续运行可复查的修改版 NCCL；
   - 原生最小双 rank AllReduce 直接加载项目构建的 `libnccl.so`，并在插桩后重复验证；
   - 真实事件进入 E05 并由 E04 分析；
   - 至少一个确定性软件延迟用例能被检测和定位；
   - 记录正确性、开销和丢事件结果。

### 2.2 范围边界

本项目聚焦 Mycroft 的 NCCL 观测、异常检测和根因分析关键路径，不扩展为生产监控平台或论文全部大规模实验的复刻。具体任务只有在存在技术歧义、安全风险或验收混淆时才说明排除项。

### 2.3 完成状态

| 状态 | 定义 |
|---|---|
| 未开始 | 尚未建立当天骨架 |
| 开发中 | 骨架已建立，但验收尚未全部通过 |
| 已验证 | 自动测试、固定演示、运行证据和用户解释全部通过 |
| 环境阻塞 | 可做部分已完成，但被集群、GPU、网络、权限或镜像条件阻塞 |

L1 只有在 Day 23 的真实 NCCL 单机双 rank 端到端验收通过后完成。L2 只有在 Day 26 的 Crater 多机真实验证通过后完成；仅完成代码或仅能编译时必须标为“L2 实现完成，集群验证未完成”。

## 3. 任务执行方式

本节只描述逐日任务如何启动和验收；若具体措辞与 `AGENTS.md` 或用户最新明确指令冲突，以后两者为准。

### 3.1 每日开始方式

Codex 先提供当天任务卡，并在用户确认任务边界后才修改骨架文件。任务卡必须包含：

- 当天解决的实际问题；
- 输入与输出；
- 只有存在技术歧义、安全风险或验收混淆时才说明排除项；
- 需要补充的最小知识；
- 文件与接口框架；
- 自动测试和固定演示命令；
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
- 循环缓冲区并发、覆盖和序号逻辑；
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

- 用户必须亲自执行当天验收；
- 只有用户明确说“验收通过，进入下一步”，Codex 才能开始下一天；
- 测试通过但用户无法解释关键状态变化时，不进入下一天；
- 不允许删除或弱化测试来绕过验收；
- 验收失败只修当前缺口，不重做整条路线。
- Day 14 起，单元测试可以使用合成 fixture，但当天运行验收必须包含由项目构建的真实 NCCL 2.21.5 产生的证据。
- 真实 NCCL、GPU 或集群条件不可用时标记环境阻塞；不得回退到 E01、E02 或手写 JSONL 并宣称当天通过。

## 4. 每个开发日的固定节奏

建议将 4—5 小时分配为：

1. 30—45 分钟：理解任务、接口、输入输出和验收；
2. 2.5—3 小时：实现核心逻辑；
3. 30—60 分钟：调试和自动测试；
4. 30 分钟：固定演示、必要的 README 更新和 Git diff；
5. 最后：验收通过并得到用户明确指令后创建可运行 commit；push 默认由用户执行。

每天结束时必须存在：

- 可运行代码；
- 自动测试或确定性的验证脚本；
- README 中的运行命令；
- 必要证据写入实验 README、自动测试或小型样例，不默认创建每日 devlog；
- 用户明确要求后创建至少一个不含凭据、二进制和大型日志的 commit。

## 5. 技术路线和环境基线

| 范围 | 技术 |
|---|---|
| E01 | C11、CMake、CTest；必要时由 Python 黑盒测试读取 JSONL |
| E02 | Python 3.10+、类型标注、`unittest`；仅作为确定性 fixture |
| E03—E04 | Python 3.10+、Event v2、JSONL、真实 NCCL 轨迹集成测试 |
| E05—E06 | C++17、CMake、CTest、POSIX shared memory、原子变量 |
| 平台探测 | 已验证的 Crater 平台镜像：Python 3.12、PyTorch 2.6.0a0（NVIDIA 24.12）、CUDA 12.6、V100 |
| NCCL 接入 | NCCL 2.21.5、Ubuntu 22.04、GCC/G++ 11、CUDA devel 镜像 |

Day 03/04 优先复用已经通过 Gloo 与 CUDA 探针的 Crater 平台镜像；更换镜像时必须重新运行最小探针并在对应实验 README 中记录实际版本。Day 14 起所有运行实验使用原生程序直接链接项目构建的 NCCL 2.21.5，不以 PyTorch 自带 NCCL 或模拟器作为插桩与分析验收依据。

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
│   └── crater/
├── notes/
│   └── nccl/
├── experiments/
│   ├── e01_ring_dependency/
│   └── e02_progress_state_machine/
├── src/
│   └── mycroft/
│       ├── schema/
│       ├── trace/
│       └── analysis/
├── runtime/
│   ├── include/mycroft_trace/
│   ├── src/
│   ├── tools/
│   └── tests/
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

`third_party/nccl/` 在 Day 13 固定到 NCCL 2.21.5 的明确 tag/commit；Day 14 起编译该版本，Day 16 起才在功能分支中加入可复查插桩。实际 NCCL 修改通过 fork/submodule 或可复查 patch 保存；不能在未跟踪的嵌套 Git 工作区中修改后丢失历史。

## 7. 系统主数据流

```text
真实 NCCL 2.21.5
        |
        v
completion/state tracepoint
        |
        v
E05 本 Pod 共享内存 -> 独立 reader -> Event v2 JSONL -> E04 Trigger/RCA

E01/E02 ---------------------------> 单元回归输入（不作为运行验收）
```

论文实现使用本机共享内存、异步 agent 和后端分析。本项目复现其中关键路径：共享内存只在 Pod/主机本地使用，每个 rank 输出独立事件流，E04 再按 communicator、operation、rank 和 channel 恢复并分析。Day 14 起的实验结论必须来自上图的真实 NCCL 主路径；E01/E02 只负责让边界测试保持确定、快速和可重复。

## 8. 26 个开发日总览

| 开发日 | 阶段 | 当日可提交结果 |
|---:|---|---|
| 01 | 仓库基线 | 公开仓库骨架、正式笔记迁移、统一测试入口 |
| 02—04 | Crater 基础 | 页面概念、CPU 双 Pod、单 GPU 基线 |
| 05—07 | E01 | 独立实现的 4-rank Ring 依赖模拟器 |
| 08—10 | E02 | GPU/Proxy/Network 三进度状态机 |
| 11—13 | E03 | Event v1/v2、乱序恢复、NCCL 2.21.5 映射 |
| 14—17 | E06 前置 | NCCL 构建、原生 workload、真实 completion/state trace |
| 18—19 | E05 | 真实事件共享内存、非阻塞 reader 和压力运行 |
| 20 | E06 故障输入 | 真实 NCCL 控制流软件延迟和证据集 |
| 21—23 | E04 | 基于真实轨迹的 Trigger、MinOp/MinData、RCA 和 L1 |
| 24—25 | Crater 多机 | 插桩版 NCCL 双物理节点 Socket/RDMA 端到端 |
| 26 | E06 | 真实 NCCL 开销、丢事件和 L2 验收 |

## 9. 每日任务卡

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

**建议 commit**：`feat(e06): run native two-rank NCCL baseline`

### Day 16：E06 真实 completion log 与 operation identity

**实际问题**：先从真实 NCCL 运行中取得论文所述 completion log，并动态确认 operation 身份和时间语义。

**核心逻辑与理解重点**：在真实 enqueue/proxy 生命周期中关联 `commHash`、`opCount`、rank、channel 和 `ncclInfo.nBytes`；寻找并验证 CollOp 完成聚合点。start/end timestamp 必须来自同一进程单调时钟。热路径先写入预分配的有界内存记录，workload 结束后再导出，不在 Proxy 热循环格式化 JSON 或执行阻塞文件 I/O。

**验收与预期现象**：Day 15 workload 产生真实 Event v2 completion 记录；同一 CollOp 跨 channel 的身份一致；连续 operation 的序号稳定；开始时间不晚于结束时间；字节数与 workload 输入一致。若只能证明 Proxy 子操作完成而不能证明 CollOp 完成，事件必须按真实语义命名，不能冒充 completion log。

**建议 commit**：`feat(e06): trace real NCCL operation completion`

### Day 17：E06 真实周期 state log 与进度字段验证

**实际问题**：用真实 NCCL Proxy 运行验证 Day 13 的三进度候选，而不是继续从 E02 推断。

**核心逻辑与理解重点**：第一版限定 Ring + SIMPLE 发送路径；在 CollOp 进行期间按配置周期采样；生产目标约 100 ms，验证时可缩短周期或增大真实消息量，但必须记录实际配置。动态验证 GPU readiness 派生量、`sub->transmitted`、`sub->done` 和 `nsteps/sliceSteps` 的单位、单调性与生命周期；明确它们表示本地发送侧进度，不解释成远端 GPU 已消费。

**验收与预期现象**：足够长的真实 AllReduce 至少产生两条 state log 和一条 completion log；日志只在 operation 活跃期间出现；每个已验证字段能追溯到实际源码转换；不支持的协议或路径显式标记。E02 只可用于单元回归，不能作为本日运行证据。

**建议 commit**：`feat(e06): validate real NCCL progress tracepoints`

### Day 18：E05 真实事件 ABI 与共享内存生命周期

**实际问题**：把 Day 16—17 已验证的真实记录放入稳定二进制 ABI，并让 NCCL 进程与独立 reader 映射同一段本机共享内存。

**文件框架**：

```text
runtime/
├── CMakeLists.txt
├── include/mycroft_trace/event_abi.h
├── include/mycroft_trace/shm_region.h
├── src/shm_region.cpp
└── tests/test_event_abi.cpp
```

**核心逻辑与理解重点**：固定 POD/standard-layout 事件、尺寸、对齐和版本；实现 create/open/close/unlink；NCCL 初始化时按显式开关创建 writer 区域，独立进程只读打开。ABI 字段来自真实 completion/state log，不为 E02 tick 设计生产字段。

**验收与预期现象**：运行真实 AllReduce 时 writer 和 reader 映射成功；版本或容量不匹配时拒绝读取；关闭 tracing 时 NCCL 基线行为不变；异常退出后有明确清理方式。

**建议 commit**：`feat(e05): connect real NCCL events to shared memory`

### Day 19：E05 非阻塞 SPSC、reader 与真实压力运行

**实际问题**：让真实 NCCL 高频写事件时不等待 reader，并把二进制记录转换成 Event v2 JSONL。

**文件框架**：`ring_buffer.h/.cpp`、`runtime/tools/trace_reader.cpp`、`runtime/tests/`。

**核心逻辑与理解重点**：实现 reserve/publish/read、write/read sequence、acquire/release、wrap-around、覆盖策略、dropped counter、批量读取和优雅停止。单元测试可以使用固定记录覆盖边界，但运行验收必须由连续真实 AllReduce 产生数据。

**验收与预期现象**：正常真实负载零丢失；故意放慢 reader 时 writer 不阻塞且 dropped counter 可观察；reader 输出通过 Event v2 validator；事件数能与真实 workload operation 数和采样周期对账。

**建议 commit**：`feat(e05): stream real NCCL traces through nonblocking reader`

### Day 20：E06 真实控制流软件延迟与证据集

**实际问题**：在实际 NCCL 2.21.5 控制流中制造可重复异常，得到后续分析器的真实输入。

**核心逻辑与理解重点**：加入默认关闭、按 rank/op/channel 精确选择的延迟开关。最低验收范围是在 `ncclNet->isend` 前设置基于单调时钟的 release deadline：到期前本轮不提交请求但继续运行 Proxy progress 和周期采样，禁止用阻塞 `sleep` 冻结整个 Proxy 线程。只有源码位置和日志能够证明语义时才增加其他注入点。该实验是修改版真实 NCCL 的软件延迟，不宣称等价于物理 GPU、NIC 或 RDMA 故障。

**验收与预期现象**：基线、仅插桩、插桩加延迟三组真实运行数值均正确；只有目标路径主动等待；completion/state log 出现可解释差异；关闭开关后行为恢复；保存小型脱敏 Event v2 样例供回归测试。

**建议 commit**：`feat(e06): inject deterministic delay into real NCCL path`

### Day 21：E04 基于真实轨迹的时间窗口与 Trigger

**实际问题**：从 Day 19—20 的真实 completion/state log 中找出需要进入根因分析的时间点。

**文件框架**：

```text
src/mycroft/analysis/
├── window.py
├── trigger.py
└── result.py
tests/analysis/test_trigger.py
```

**核心逻辑与理解重点**：窗口必须接收明确的采集水位或截止时间；state log 持续存在但窗口内没有 completion 时触发 failure；使用真实 completion 的 start/end/bytes 计算吞吐和 CollOp interval；减半/翻倍阈值可配置，并从正常真实运行建立基线。

**验收与预期现象**：正常真实轨迹不触发；Day 20 延迟轨迹在预期窗口触发；输出只包含异常时间、类型、输入数值和阈值，不提前声称根因。合成 fixture 只用于边界单元测试，不能单独完成本日验收。

**建议 commit**：`feat(e04): trigger on real NCCL trace windows`

### Day 22：E04 基于真实多 rank 轨迹的 MinOp 与 MinData

**实际问题**：在同一次真实运行的多个 rank 中先找 operation 落后者，再比较同一 operation 的已验证进度。

**核心逻辑与理解重点**：实现 `CheckMinOp` 和 `CheckMinData`；只比较 Day 17 已证明单位一致的字段；保留并列候选、缺失事件、dropped record 和不支持路径，不把最晚 rank 自动当根因。

**验收与预期现象**：Day 20 的真实多 rank 轨迹得到可复查候选；operation 落后时优先输出 MinOp；operation 一致时才使用 MinData；改变输入到达顺序不改变结果；数据缺口降低置信度而不是被静默忽略。

**建议 commit**：`feat(e04): locate lagging ranks in real NCCL traces`

### Day 23：E04 真实轨迹 RCA 与 L1 验收

**实际问题**：把真实 Trigger、MinOp/MinData、发送侧状态和依赖证据组成最小可解释 RCA。

**文件框架**：`rca.py`、`evidence.py`、真实样例的 expected result 和 CLI。

**核心逻辑与理解重点**：依据论文状态表输出主动异常候选、受影响 rank、证据链和证据缺口；只对 Day 17—20 已动态确认的路径作结论，未采集接收侧或硬件证据时必须保留边界。

**验收与预期现象**：同机真实 NCCL 基线不报故障；真实软件延迟运行输出与注入位置相符的候选而非绝对硬件结论；报告能追溯到原始 Event v2；用户能解释为什么最慢 rank 可能只是受影响者。

**Gate L1**：L1 现在要求真实 NCCL 单机双 rank 端到端证据；纯 E01/E02 或手写 Event fixture 不能通过。

**建议 commit**：`feat(e04): complete real-trace L1 analyzer`

### Day 24：Crater 双物理节点 Socket 端到端

**实际问题**：把同机已验证链路迁移到两个物理节点，并明确使用 TCP Socket transport。

**核心逻辑、理解与平台操作**：两个 Pod 各运行一个直接链接项目 NCCL 2.21.5 的原生 rank；启用 tracing、reader 和 E04；使用调度约束确认不同物理节点；设置诊断用 `NCCL_IB_DISABLE=1` 与 `NCCL_DEBUG=INFO`。

**验收与预期现象**：AllReduce 正确；动态链接指向项目 NCCL；日志出现 `NET/Socket`；两端 Event v2 可恢复为同一批 operation；正常运行不触发异常，软件延迟运行能够触发。两个 Pod 落在同一节点时不通过。

**建议 commit**：`feat(crater): validate instrumented NCCL over two-node socket`

### Day 25：Crater 双物理节点 RDMA 端到端

**实际问题**：在保持 workload、NCCL、插桩和分析器不变的条件下，只把 transport 从 Socket 切换到 RDMA。

**核心逻辑、理解与平台操作**：检查 IB 设备、link layer、memlock 和 NCCL NET 日志；移除 Socket 强制开关；记录真实 connection/QP 元数据并验证其生命周期；继续运行正常和软件延迟两组。

**验收与预期现象**：两个物理节点、同型号 GPU、AllReduce 正确；日志明确出现 `NET/IB` 且没有静默回退；Event v2 与 E04 端到端可用；无法获得 RDMA 时标为环境阻塞，不能用 Socket 或模拟轨迹冒充。

**Gate Crater**：用户能核对 Pod/Node、动态库、transport、rank 日志和脱敏证据。

**建议 commit**：`feat(crater): validate instrumented NCCL over RDMA`

### Day 26：E06 开销、丢事件和 L2 最终验收

**实际问题**：证明真实 NCCL 插桩不仅能产生日志，而且能在不破坏正确性的前提下支持检测和定位。

**核心逻辑、理解与平台操作**：在相同环境运行未插桩基线、开启插桩和开启软件延迟三组矩阵；比较正确性、运行时间、事件数、dropped counter、Trigger 和 RCA；不把小样本外推为生产性能结论。

**验收与预期现象**：

1. 三组真实 NCCL AllReduce 数值均正确；
2. 正常真实轨迹不触发故障；
3. 延迟用例在预期窗口触发；
4. RCA 输出候选、受影响 rank、证据和边界；
5. 动态链接、NCCL 版本和 Socket/RDMA 路径均有证据；
6. 丢事件为零，或报告明确数量及对结论的影响；
7. 从干净环境可按文档重现实验。

**建议 commit/PR**：`feat(e06): complete real NCCL L2 validation`

**Gate L2**：用户完成演示并核对真实 NCCL、两个物理节点、RDMA、事件链和分析报告后，项目才标为 L2 已验证。

## 10. 阶段验收矩阵

| Gate | 必须回答的问题 | 必须存在的证据 |
|---|---|---|
| Crater Basics | 本地代码如何通过镜像、启动命令和挂载进入作业？ | 脱敏配置模板、CPU 双 Pod 日志、单 GPU 探测报告 |
| E01 | 为什么最慢 rank 不一定是根因？ | 正常/延迟 JSONL、因果链、数值测试 |
| E02 | 哪个执行者没有推进，还缺什么证据？ | 三类故障轨迹、状态分类测试 |
| E03 | 事件为何属于同一次 op/flow？ | 乱序恢复测试、2.21.5 字段来源表 |
| NCCL Baseline | 是否真正加载并运行项目 NCCL 2.21.5？ | 版本、动态链接、同机双 rank 正确性 |
| Trace Semantics | completion/state 字段是否来自已验证真实路径？ | 源码位置、真实 Event v2、operation 与字节数对账 |
| L1 | 分析器为什么输出该根因候选？ | 真实正常/延迟轨迹、expected RCA、证据链 |
| E05 | reader 慢时真实 NCCL writer 会发生什么？ | 真实压力运行、dropped counter、Event v2 输出 |
| Crater Multi-node | 两个 Pod 是否真在两台机器，项目 NCCL 走哪条网络？ | Node 字段、动态库、Socket/IB 日志、正确性结果 |
| L2 | 真实插桩是否支持定位且不破坏 NCCL？ | 动态链接、真实 JSONL、延迟 RCA、开销报告 |

## 11. GitHub 工作流

### 11.1 Day 01—Day 13

- 每个验收通过的开发日至少形成一个通过测试的 commit，Day 01—Day 13 直接提交 `main`；
- commit 前运行统一检查脚本；
- 不提交失败测试、WIP、二进制或大型输出；
- commit 和 push 必须由用户明确发出指令；push 默认由用户执行；
- commit 前核对暂存边界，不把下一 Day 的工作混入当前提交；
- 推荐格式：`type(scope): subject`。

### 11.2 Day 14—Day 26

- Day 14 起涉及构建、workload、E05/E06 底层运行时或 NCCL 源码时使用短分支；Crater 配置或探针是否单独建分支由任务风险决定；
- 分支只覆盖一个阶段或明确子任务；
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
5. Day 14 起不把模拟器、手写 Event 或仅单元测试成功写成当天实验通过、L1 或 L2 完成；
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

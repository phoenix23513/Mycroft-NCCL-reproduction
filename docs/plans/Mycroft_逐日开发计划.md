# Mycroft 逐日开发计划

- 版本：v0.2（已确认；Crater 基础前置）
- 依据计划：`Mycroft_复现冻结计划.md` v1.0
- NCCL 目标版本：2.21.5
- 最终目标：完成 L2 NCCL 插桩原型，并在 Crater 多机 GPU 环境中完成真实验证
- 工作强度：每个开发日 4—5 个专注小时，不绑定自然日期
- 预计规模：26 个开发日；集群排队、权限申请和平台故障等待不计入开发日
- 当前状态：Day 02 已验收；Crater 单容器 CPU 冒烟作业现象符合预期，等待确认 Day 03 任务边界

## 1. 计划要解决的问题

本项目把 Mycroft 复现作为开发主线，不再把“先系统学完 NCCL”作为启动条件。每个阶段都从一个可运行功能出发；只有实现遇到实际缺口时，才补充解决该缺口所需的 NCCL、C/C++、Python、Docker、PyTorch、Kubernetes 或 RDMA 知识。

每天只完成一个有明确边界的子任务，并在验收通过后形成至少一个可运行 commit。解释过、阅读过或通过自测均不等于完成；只有代码、自动测试、固定演示和运行证据一致时，子任务才算通过。

本计划不替代冻结计划。冻结计划继续控制 E01—E06 的研究边界，本计划只把它展开成可执行的每日开发任务。

## 2. 冻结的项目范围

### 2.1 必须完成

1. **E01：4-rank Ring 依赖模拟器**
   - 正确完成 ReduceScatter 和 AllGather；
   - 输出 JSONL 事件；
   - 注入单点延迟并展示依赖传播；
   - 区分主动异常 rank 和后续受阻 rank。
2. **E02：GPU/Proxy/Network 三段状态机**
   - 模拟 `GPU_ready`、`RDMA_transmitted`、`RDMA_done`；
   - 覆盖 GPU、Proxy 和网络延迟；
   - 产生可供分析器使用的状态轨迹。
3. **E03：统一事件身份与 NCCL 2.21.5 源码映射**
   - 混合多 communicator、多 operation、多 channel 后仍能正确分组；
   - 每个真实身份字段和候选插桩点均有 NCCL 2.21.5 源码依据。
4. **E04：Mycroft 分析器 MVP**
   - 停滞触发、慢速触发、`MinOp`、`MinData`；
   - 根因候选、受影响 rank、证据链和置信边界；
   - 固定故障用例和自动测试。
5. **E05：共享内存循环缓冲区与独立 reader**
   - 固定事件 ABI；
   - 写入方不等待 reader；
   - 覆盖、丢事件和版本错误均可观察；
   - reader 输出可直接交给 E04。
6. **Crater 平台上手和多机基线**
   - 理解普通用户需要操作的页面与对象；
   - CPU 双 Pod、单 GPU、双节点 Socket、双节点 RDMA 逐层验证。
7. **E06：接入 NCCL 2.21.5**
   - 编译修改版 NCCL；
   - 原生最小双 rank AllReduce 直接加载修改版 `libnccl.so`；
   - 真实事件进入 E05 并由 E04 分析；
   - 至少一个确定性软件延迟用例能被检测和定位；
   - 记录正确性、开销和丢事件结果。

### 2.2 明确不做

- 不复刻论文的 Kafka、云数据库和生产 Web 后端；
- 不复刻 32 张 A100 或更大规模的全部论文实验；
- 不把 Tree、CollNet、NVLS、LL、LL128 的完整实现纳入主线；
- 不系统学习 Kubernetes 管理、Crater 部署或 RDMA verbs；
- 不承诺硬件破坏性故障注入、网卡限速、PCIe 降级或 GPU 限功率；
- 不把当前 `参考答案/` 作为设计、实现或测试输入；
- 不把当前 E01 C 程序迁入新仓库，它只留在旧目录作为历史草稿。

### 2.3 完成状态

| 状态 | 定义 |
|---|---|
| 未开始 | 尚未建立当天骨架 |
| 开发中 | 骨架已建立，但验收尚未全部通过 |
| 已验证 | 自动测试、固定演示、运行证据和用户解释全部通过 |
| 环境阻塞 | 可做部分已完成，但被集群、GPU、网络、权限或镜像条件阻塞 |

L1 只有在 Day 17 通过后完成。L2 只有在 Day 26 的 Crater 多机真实验证通过后完成；仅完成代码或仅能编译时必须标为“L2 实现完成，集群验证未完成”。

## 3. 结对开发契约

### 3.1 每日开始方式

Codex 先提供当天任务卡，并在用户确认任务边界后才修改骨架文件。任务卡必须包含：

- 当天解决的实际问题；
- 输入、输出和不做范围；
- 需要补充的最小知识；
- 文件与接口框架；
- 自动测试和固定演示命令；
- 预期现象；
- 验收标准；
- 建议 commit message。

### 3.2 分工

Codex 可以完成：

- 目录、构建配置和接口声明；
- 测试框架、固定输入、fixtures 和失败断言；
- CLI 参数解析骨架；
- README/实验报告模板；
- 空实现、明确的 `TODO` 和错误返回；
- 经用户明确要求后的代码审查与局部修复。

用户负责完成：

- Ring step/chunk 数据传播；
- 因果依赖和延迟传播；
- 三进度量状态转移；
- 事件分组、触发器、`MinOp`、`MinData` 和 RCA；
- 循环缓冲区并发、覆盖和序号逻辑；
- NCCL 2.21.5 真实字段确认与插桩更新；
- Crater 页面配置、作业提交、日志查看和最终验收。

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

## 4. 每个开发日的固定节奏

建议将 4—5 小时分配为：

1. 30—45 分钟：理解任务、接口、输入输出和验收；
2. 2.5—3 小时：实现核心逻辑；
3. 30—60 分钟：调试和自动测试；
4. 30 分钟：固定演示、README、开发日志和 Git diff；
5. 最后：提交并 push 一个可运行 commit。

每天结束时必须存在：

- 可运行代码；
- 自动测试或确定性的验证脚本；
- README 中的运行命令；
- `docs/devlog/day-XX.md` 中的简短证据；
- 至少一个不含凭据、二进制和大型日志的 commit。

## 5. 技术路线和环境基线

| 范围 | 技术 |
|---|---|
| E01 | C11、CMake、CTest；必要时由 Python 黑盒测试读取 JSONL |
| E02—E04 | Python 3.10+、类型标注、`pytest`、JSONL |
| E05—E06 | C++17、CMake、CTest、POSIX shared memory、原子变量 |
| 平台探测 | PyTorch 2.4.1、CUDA 12.4、Python 3.10 |
| NCCL 接入 | NCCL 2.21.5、Ubuntu 22.04、GCC/G++ 11、CUDA devel 镜像 |

若 Crater 不提供完全一致的 PyTorch/CUDA 组合，优先选择平台维护的 Python 3.10、CUDA 12.4 邻近稳定组合；实际选择必须记录在实验报告中。最终 E06 使用原生程序直接链接 NCCL 2.21.5，不以 PyTorch 自带 NCCL 作为插桩验收依据。

## 6. 目标 GitHub 仓库结构

仓库名称和许可证在 Day 01 创建前由用户决定。建议结构如下：

```text
mycroft-nccl-reproduction/
├── README.md
├── LICENSE
├── .gitignore
├── CMakeLists.txt
├── pyproject.toml
├── docs/
│   ├── plans/
│   │   ├── Mycroft_复现冻结计划.md
│   │   └── Mycroft_逐日开发计划.md
│   ├── architecture/
│   ├── crater/
│   ├── reports/
│   ├── progress/
│   └── devlog/
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

`third_party/nccl/` 在 Day 13 固定到 NCCL 2.21.5 的明确 tag/commit，供源码映射使用；Day 23 起才编译并修改该版本。实际 NCCL 修改通过 fork/submodule 或可复查 patch 保存；不能在未跟踪的嵌套 Git 工作区中修改后丢失历史。

## 7. 系统主数据流

```text
E01 Ring 事件 ─┐
               ├─> E03 统一 Event v1 ─> E04 Trigger/RCA
E02 三进度轨迹 ┘                         ^
                                         |
NCCL 2.21.5 tracepoint ─> E05 本 Pod shm ─> reader ─> rank JSONL
                                                        |
                                                        v
                                              Crater 共享文件系统
```

论文完整实现使用本机共享内存、异步 agent 和 Python 后端。本项目只复现这条关键数据路径；共享内存是 Pod/主机本地资源，不跨机器共享。每个 rank/Pod 写独立 JSONL，最后由 E04 汇总。

## 8. 26 个开发日总览

| 开发日 | 阶段 | 当日可提交结果 |
|---:|---|---|
| 01 | 仓库基线 | 公开仓库骨架、正式笔记迁移、统一测试入口 |
| 02—04 | Crater 基础 | 页面概念、CPU 双 Pod、单 GPU 基线 |
| 05—07 | E01 | 独立实现的 4-rank Ring 依赖模拟器 |
| 08—10 | E02 | GPU/Proxy/Network 三进度状态机 |
| 11—13 | E03 | Event v1、乱序恢复、NCCL 2.21.5 映射 |
| 14—17 | E04 | Trigger、MinOp/MinData、RCA、L1 验收 |
| 18—20 | E05 | 共享内存循环缓冲区和 reader |
| 21—22 | Crater 多机 | 双物理节点 Socket/RDMA 基线 |
| 23—26 | E06 | 修改版 NCCL、原生 workload、真实插桩、L2 验收 |

## 9. 每日任务卡

### Day 01：创建公开仓库和可运行基线

**实际问题**：为后续每日提交建立干净、可复现、不会泄露本机信息的项目边界。

**Codex 搭建**：

- 创建第 6 节目录骨架；
- 提供 `.gitignore`、根 `CMakeLists.txt`、`pyproject.toml` 和测试入口；
- 迁移冻结计划、逐日计划、`NCCL_源码学习日志.md` 和 `notes/` 下正式 Markdown；
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

**实际问题**：不用 GPU 验证 Master/Worker、环境变量、服务发现、日志和共享文件系统。

**文件框架**：`cluster/crater/probes/env_probe.py`、`scripts/run_cpu_probe.sh`、C01 报告。

**Codex 搭建**：探测接口、pytest、本次页面填写表和预期日志模板。

**用户核心逻辑与操作**：实现 hostname/env 输出、Gloo process group 和 CPU all-reduce；在页面配置 Master 1/Worker 1/GPU 0；使用 `/crater-start.sh`；提交、查看两个 Pod、日志和共享文件。

**验收与预期现象**：`WORLD_SIZE=2`，rank 为 0/1，AllReduce 结果一致；两个 Pod 都能写各自文件并读取共享目录；用户会停止/清理作业。

**建议 commit**：`feat(crater): validate two-pod CPU control plane`

### Day 04：Crater 单 Pod 单 GPU 基线

**实际问题**：独立确认 GPU、驱动、CUDA 和 PyTorch 镜像可用，为后期多机 NCCL 排除基础环境问题。

**最小补充知识**：GPU request、GPU model、`CUDA_VISIBLE_DEVICES`、宿主驱动与容器 CUDA toolkit 的区别。

**文件框架**：`cluster/crater/probes/gpu_probe.py`、C02 报告。

**Codex 搭建**：检查项、页面配置表和输出模板。

**用户核心逻辑与操作**：检查 `nvidia-smi`、PyTorch CUDA、device count、GPU tensor 运算和同步；选择一张 V100，资源不足时选择 A100。

**验收与预期现象**：Pod 只看到分配的 GPU；PyTorch tensor 运算正确；记录 GPU、driver、CUDA、PyTorch 和镜像版本；作业数分钟内完成并正常释放资源。

**建议 commit**：`feat(crater): verify single-GPU runtime baseline`

**Gate Crater Basics**：用户能独立填写基础作业字段、提交作业、查看 Pod/日志、停止作业，并解释本地代码如何通过镜像、启动命令和挂载进入 Crater。

### Day 05：E01 数据模型与 ReduceScatter

**实际问题**：用代码证明一个 chunk 怎样沿固定 Ring 被逐步规约。

**最小补充知识**：4-rank 单 channel Ring、chunk owner、step、同步轮次；结构体、结构体指针和数组的生命周期。

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

**用户核心逻辑**：初始化 chunk、计算发送/接收 rank、实现三个 ReduceScatter step，并保存每个 step 的状态。

**验收与预期现象**：

- ReduceScatter 后每个 rank 恰好持有一个完成规约的 chunk；
- chunk owner 和直接求和结果一致；
- AddressSanitizer/基本内存检查无越界。

**建议 commit**：`feat(e01): implement ring reduce-scatter steps`

### Day 06：E01 AllGather、最终结果与 JSONL

**实际问题**：让规约结果传播到所有 rank，并把依赖过程变成机器可读事件。

**Codex 搭建**：JSONL writer 接口、schema 断言、最终结果测试和 README 命令模板。

**用户核心逻辑**：实现 AllGather 转发、保存完整结果；在每个 step 发出包含 `op_seq/rank/channel/phase/step/chunk/action/peer/timestamp` 的事件。

**验收与预期现象**：

- 四个 rank 得到相同 AllReduce 结果；
- JSONL 每行可独立解析；
- 正常轨迹中每个 chunk 的 step 单调递增；
- 删除或交换任一必要事件会触发依赖一致性测试失败。

**建议 commit**：`feat(e01): add all-gather and JSONL trace output`

### Day 07：E01 因果依赖、延迟注入与阶段验收

**实际问题**：证明最晚完成的 rank 不一定是主动异常 rank。

**文件扩展**：`src/delay.c`、`tests/test_delay_propagation.c`、`results/samples/e01/`。

**Codex 搭建**：延迟 CLI 参数、根事件 fixture、因果图测试和验收报告模板。

**用户核心逻辑**：实现直接依赖、同 rank 顺序依赖、延迟传播和根事件标记；不能给所有受影响事件直接增加人为延迟。

**验收与预期现象**：

- 只有一个事件被主动注入；
- 跨 rank 后继事件的开始时间被推迟；
- 数值结果不变；
- 输出能明确区分 injected root 与 affected ranks；
- 用户能依据事件解释“最慢 rank 为什么可能只是受害者”。

**建议 commit**：`feat(e01): trace causal delay propagation`

**Gate E01**：全部通过后，阶段 0—4 从“已讲授”变为“已验证”。

### Day 08：E02 正常三执行者状态机

**实际问题**：把 GPU producer、CPU Proxy 和 Network completion 变成有先后约束的可运行模型。

**最小补充知识**：NCCL Proxy 是进程内 CPU 线程；send 侧的准备、提交、完成含义；只学习理解状态机所需的 QP/CQE 概念。

**文件框架**：

```text
experiments/e02_progress_state_machine/
├── README.md
├── progress_sim.py
├── cli.py
└── tests/test_normal_progress.py
```

**Codex 搭建**：三个 actor 接口、tick/scheduler 骨架、正常 fixture 和单调性断言。

**用户核心逻辑**：实现 `GPU_ready >= RDMA_transmitted >= RDMA_done` 的正常推进和周期性状态记录。

**验收与预期现象**：正常轨迹最终三者相等，任何时刻均不违反单调性和先后不变量。

**建议 commit**：`feat(e02): model normal GPU proxy network progress`

### Day 09：E02 三层故障注入

**实际问题**：让相同的“collective 变慢”表象产生不同的内部状态证据。

**Codex 搭建**：`FaultSpec`、三类参数化测试和固定输出目录。

**用户核心逻辑**：实现 GPU 未准备、Proxy 未发送、Network 未完成三类延迟/停滞，并保证故障只作用于指定执行者。

**验收与预期现象**：

- GPU 故障：三个进度量共同停止或整体滞后；
- Proxy 故障：`GPU_ready > RDMA_transmitted`；
- 网络完成故障：`RDMA_transmitted > RDMA_done`；
- 每类轨迹仍满足物理先后约束。

**建议 commit**：`feat(e02): inject component-specific progress faults`

### Day 10：E02 状态分类与阶段验收

**实际问题**：从状态轨迹给出“知道什么、还不能断言什么”，避免把相关性当根因。

**文件扩展**：`classify.py`、`tests/test_state_classification.py`、E02 报告。

**Codex 搭建**：论文四类状态的期望表和模糊边界测试。

**用户核心逻辑**：实现未开始、未发送、未送达、GPU 未继续准备的分类，并输出本地原因、远端可能原因和证据不足项。

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

**用户核心逻辑**：确定必填字段、身份字段、operation 字段、progress 字段和依赖字段的验证规则；编写 E01/E02 adapter。

**验收与预期现象**：两类实验输出均能通过同一 validator；缺 rank、op identity 或非法进度关系时给出明确错误。

**建议 commit**：`feat(schema): define versioned unified trace events`

### Day 12：E03 乱序、多 communicator、多 operation 恢复

**实际问题**：不能依赖文件到达顺序判断事件属于哪次 collective。

**文件框架**：`src/mycroft/trace/group.py`、`timeline.py`、`tests/trace/`。

**Codex 搭建**：两组 communicator、多个 operation/channel 的打乱 fixture 和期望分组。

**用户核心逻辑**：定义 operation/flow key，完成乱序分组、去重、缺口报告和时间线恢复。

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

**用户核心逻辑**：添加官方 NCCL 源码引用并固定到 2.21.5 的明确 tag/commit（不沿用当前未固定的 master 工作区）；逐项确认 `IP/comm_id/Gid/GPU_id/channel_id/QP_id/op_seq/msg_size` 的创建者、结构、生命周期、跨 rank 一致性和唯一性；记录三个进度量的候选位置及“已确认/待确认”。

**验收与预期现象**：仓库记录的 tag/commit 可复查且工作区干净；每项结论都指向该版本的文件、函数、结构体成员或明确的数据流；所有猜测均标为待确认。

**建议 commit**：`docs(e03): map trace identity to NCCL 2.21.5`

**Gate E03**：Event v1 冻结。后续若更改字段，必须提升 schema version 或提供兼容转换。

### Day 14：E04 时间窗口与 Trigger

**实际问题**：从连续轨迹中找出值得进入根因分析的时间窗口。

**文件框架**：

```text
src/mycroft/analysis/
├── window.py
├── trigger.py
└── result.py
tests/analysis/test_trigger.py
```

**Codex 搭建**：正常、停滞、吞吐下降、间隔增大 fixture 和参数接口。

**用户核心逻辑**：实现完成日志缺失的停滞触发，以及默认“吞吐减半/operation 间隔翻倍”的可配置慢速触发。

**验收与预期现象**：正常短时波动不触发；固定停滞和慢速用例在预期窗口触发；输出只标记异常时间和触发类型，不提前声称根因。

**建议 commit**：`feat(e04): implement configurable anomaly triggers`

### Day 15：E04 MinOp 与 MinData

**实际问题**：先定位落后 operation，再在同一 operation 内比较数据进度。

**Codex 搭建**：多 rank 最后状态 fixture、并列最小值、缺失 rank 和 operation rollover 测试。

**用户核心逻辑**：实现 `CheckMinOp` 和 `CheckMinData`，保留并列候选和输入证据。

**验收与预期现象**：operation 落后时优先输出 MinOp；operation 一致时才比较 MinData；并列时不任意挑选唯一 rank。

**建议 commit**：`feat(e04): locate lagging operations and data progress`

### Day 16：E04 RCA 状态表、依赖链和置信边界

**实际问题**：把候选 rank 的三进度状态与上下游证据组合成可解释结论。

**文件框架**：`rca.py`、`evidence.py`、`tests/analysis/test_rca.py`。

**Codex 搭建**：论文状态条件、发送端/接收端对照 fixture 和结构化 `RcaResult`。

**用户核心逻辑**：实现未开始、未发送、未送达、GPU 未准备规则；沿依赖边区分 root candidate 和 affected rank；输出 local/remote cause 和 evidence gap。

**验收与预期现象**：受阻 rank 不会被误报为唯一根因；证据不充分时输出候选集合和置信边界。

**建议 commit**：`feat(e04): produce dependency-backed RCA evidence`

### Day 17：E04 固定故障套件与 L1 验收

**实际问题**：证明整个本机最小复现可演示、可回归，而不是只在一个样例上工作。

**故障集合**：GPU producer 延迟、Proxy 发送延迟、网络完成延迟、rank 整体停止、正常负载不均。

**Codex 搭建**：端到端测试矩阵、expected JSON、CLI 报告模板和 L1 验收清单。

**用户核心逻辑**：连接 Event v1、Trigger、MinOp/MinData 和 RCA；生成文本/JSON 报告与依赖时间线。

**验收与预期现象**：

- 每类故障均有固定输入和期望输出；
- 正常负载不均不会被简单判故障；
- 所有异常结果包含候选、受影响 rank、证据和边界；
- 从干净 clone 可一条命令重现演示。

**建议 commit**：`feat(e04): complete Mycroft L1 end-to-end analyzer`

**Gate L1**：用户演示并解释结果后，L1 才能标为已验证。

### Day 18：E05 固定事件 ABI 与共享内存生命周期

**实际问题**：定义能够被 NCCL 写入、被独立 reader 稳定读取的二进制事件格式。

**最小补充知识**：POD/standard-layout、字节对齐、版本号、POSIX shared memory、进程生命周期。

**文件框架**：

```text
runtime/
├── CMakeLists.txt
├── include/mycroft_trace/event_abi.h
├── include/mycroft_trace/shm_region.h
├── src/shm_region.cpp
└── tests/test_event_abi.cpp
```

**Codex 搭建**：ABI 结构、静态尺寸断言、create/open/close/unlink 接口和失败测试。

**用户核心逻辑**：实现共享内存创建、映射、只读/读写打开、清理和版本校验。

**验收与预期现象**：两个独立进程能映射同一段内存；版本或容量不匹配时拒绝读取；异常退出后有明确清理办法。

**建议分支/commit**：`feature/e05-shm-abi`；`feat(e05): define trace ABI and shared memory lifecycle`

### Day 19：E05 单写单读循环缓冲区

**实际问题**：让业务写入方不因 reader 变慢而阻塞。

**最小补充知识**：SPSC ring、write/read sequence、原子变量、acquire/release 的最小含义；不展开通用无锁算法课程。

**文件框架**：`ring_buffer.h/.cpp`、`tests/test_ring_buffer.cpp`。

**Codex 搭建**：接口、容量 4 的确定性测试、慢 reader 和 wrap-around fixture。

**用户核心逻辑**：实现 reserve/publish/read、序号判断、覆盖策略和 dropped counter。

**验收与预期现象**：writer 不等待；wrap-around 后 reader 只读到有效完整记录；覆盖数量与 dropped counter 一致；ThreadSanitizer 在环境允许时无数据竞争。

**建议 commit**：`feat(e05): implement non-blocking SPSC trace ring`

### Day 20：E05 reader、JSONL 导出和压力验收

**实际问题**：把二进制事件稳定转换成 E04 能消费的 Event v1。

**文件框架**：`runtime/tools/trace_reader.cpp`、`runtime/tests/test_reader_e2e.cpp`、E05 报告。

**Codex 搭建**：reader CLI、writer workload、慢 reader 参数、E04 validator 接口和压力脚本。

**用户核心逻辑**：实现批量读取、二进制到 JSONL 转换、丢事件标记、优雅停止和 rank 文件命名。

**验收与预期现象**：高频 writer 不被 reader 阻塞；正常负载零丢失；故意溢出时丢失可观测；导出文件通过 Event v1 validator 和 E04 parser。

**建议 commit/PR**：`feat(e05): export shared-memory traces to Event v1`，通过 PR 合并 E05 分支。

### Day 21：Crater 双物理节点 Socket NCCL

**实际问题**：先验证多节点调度和 NCCL TCP 路径，把 RDMA 变量留到下一天。

**最小补充知识**：一进程一 Pod、rank/world size、NCCL bootstrap 与 data transport、Socket 路径。

**文件框架**：`cluster/crater/probes/ddp_smoke.py`、`run_ddp_socket.sh`、C03 报告。

**Codex 搭建**：DDP 接口、正确性测试、页面配置表和 NCCL 日志检查项。

**用户核心逻辑与操作**：实现 NCCL process group 和 GPU AllReduce；Master/Worker 各 1 GPU；Allow List 选择两台同型号节点；设置诊断用 `NCCL_IB_DISABLE=1` 和 `NCCL_DEBUG=INFO`。

**验收与预期现象**：作业详情确认两个 Pod 位于不同物理节点；AllReduce 正确；日志出现 `NET/Socket`；若落到同一节点则本日不通过并重新调度。

**建议 commit**：`feat(crater): establish two-node NCCL socket baseline`

### Day 22：Crater 双物理节点 RDMA NCCL

**实际问题**：在唯一变量为网络路径的条件下，从 Socket 切换到 RDMA。

**最小补充知识**：InfiniBand、RDMA、HCA、CQE、GPU Direct 的最小角色；`ibstat`、`ibv_devices`、`ulimit -l` 分别说明什么。

**文件框架**：`run_ddp_rdma.sh`、`docs/crater/RDMA检查清单.md`、C04 报告。

**Codex 搭建**：RDMA 页面填写表、命令检查清单和 Socket/RDMA 对照报告模板。

**用户核心逻辑与操作**：为所有角色启用一致 RDMA 配置；确认 IB 设备和 memlock；移除 Socket 强制开关；设置 `NCCL_DEBUG=INFO` 与 `NCCL_DEBUG_SUBSYS=INIT,NET`。

**验收与预期现象**：两个物理节点、同型号 GPU、AllReduce 正确；日志明确出现 `NET/IB` 而非回退 `NET/Socket`；无法获取 RDMA 时标为环境阻塞，不伪造通过。

**建议 commit**：`feat(crater): validate two-node NCCL RDMA transport`

**Gate Crater**：用户能独立完成创建、提交、查看 Pod/Node、读取日志、停止作业和保存脱敏证据。

### Day 23：E06 可复现镜像与 NCCL 2.21.5 构建

**实际问题**：建立能稳定编译修改版 NCCL 的环境，并证明版本确实为 2.21.5。

**文件框架**：

```text
cluster/crater/images/Dockerfile
cluster/crater/scripts/build_nccl.sh
third_party/nccl/
instrumentation/nccl-2.21.5/README.md
```

**Codex 搭建**：Dockerfile/build script 骨架、版本检查和 smoke link test。

**用户核心逻辑与操作**：选择 CUDA devel 基础镜像；安装编译/IB 工具；固定 NCCL 2.21.5；完成未插桩版本编译；记录实际版本矩阵。

**验收与预期现象**：`nvcc`、G++、CMake 可用；NCCL 2.21.5 编译成功；最小链接测试运行；构建过程不依赖交互式手工修补。

**建议分支/commit**：`feature/e06-nccl-instrumentation`；`build(e06): reproduce NCCL 2.21.5 toolchain`

### Day 24：E06 原生双 rank 最小 AllReduce

**实际问题**：绕开 PyTorch 自带 NCCL，直接运行和验证我们编译的 `libnccl.so`。

**文件框架**：

```text
workloads/minimal_allreduce/
├── CMakeLists.txt
├── include/bootstrap.h
├── src/bootstrap.cpp
├── src/main.cpp
└── tests/
```

**Codex 搭建**：CLI、socket/bootstrap 接口、错误处理框架、单进程单元测试和 Crater 启动脚本。

**用户核心逻辑**：rank 0 获取 `ncclUniqueId`，通过最小 TCP bootstrap 分发；各 rank 设置 GPU、初始化 communicator、执行 AllReduce、同步并验证结果。

**验收与预期现象**：两个 Crater Pod 上各一个原生进程；结果正确；动态链接检查明确指向项目构建的 NCCL 2.21.5；不依赖 MPI 或 PyTorch NCCL。

**建议 commit**：`feat(e06): run native two-rank NCCL all-reduce`

### Day 25：E06 接入 E05 与真实 tracepoints

**实际问题**：把模拟阶段确认的 Event v1 和三进度观测接入 NCCL 2.21.5 真实关键路径。

**Codex 搭建**：NCCL 内部 tracing API 接口、构建开关、patch 生成脚本和真实轨迹集成测试骨架。

**用户核心逻辑**：依据 Day 13 映射确认最终插桩位置；初始化共享内存；写 completion/state log；更新 operation identity 和三个进度量；在同 Pod 启动 reader。

**验收与预期现象**：

- 未开启 tracing 时行为与基线一致；
- 开启后每个 rank 产生独立 JSONL；
- 真实事件通过 Event v1 validator；
- operation/channel/rank 能恢复成完整时间线；
- 所有 patch 可从干净 NCCL 2.21.5 重放。

**建议 commit**：`feat(e06): instrument NCCL progress into shared memory`

### Day 26：E06 软件延迟、开销和 L2 最终验收

**实际问题**：证明真实 NCCL 事件能支持检测和定位，而不是只完成日志采集。

**Codex 搭建**：基线/插桩/延迟三组运行矩阵、结果报告模板和端到端断言。

**用户核心逻辑与操作**：加入可控、默认关闭的软件延迟；运行正常与异常 workload；收集 rank JSONL；运行 E04；比较正确性、运行时间、事件数和 dropped counter。

**验收与预期现象**：

1. 正常与插桩版本 AllReduce 数值均正确；
2. 正常真实轨迹不触发故障；
3. 延迟用例在预期窗口触发；
4. RCA 输出主动异常 rank、受影响 rank、证据和边界；
5. 记录插桩前后运行时间，不把小样本结果夸大为生产开销结论；
6. 丢事件为零，或在报告中明确说明数量与影响；
7. README 能从干净环境复现实验。

**建议 commit/PR**：`feat(e06): complete real NCCL tracing and L2 validation`，通过 PR 合并 E06 分支。

**Gate L2**：用户完成演示、核对两个物理节点和 RDMA/NCCL 日志，并明确确认所有验收项后，项目才标为 L2 已验证。

## 10. 阶段验收矩阵

| Gate | 必须回答的问题 | 必须存在的证据 |
|---|---|---|
| Crater Basics | 本地代码如何通过镜像、启动命令和挂载进入作业？ | 脱敏配置模板、CPU 双 Pod 日志、单 GPU 探测报告 |
| E01 | 为什么最慢 rank 不一定是根因？ | 正常/延迟 JSONL、因果链、数值测试 |
| E02 | 哪个执行者没有推进，还缺什么证据？ | 三类故障轨迹、状态分类测试 |
| E03 | 事件为何属于同一次 op/flow？ | 乱序恢复测试、2.21.5 字段来源表 |
| L1 | 分析器为什么输出该根因候选？ | 五类 fixture、expected RCA、时间线 |
| E05 | reader 慢时 writer 会发生什么？ | 压力结果、dropped counter、Event v1 输出 |
| Crater Multi-node | 两个 Pod 是否真在两台机器，NCCL 走哪条网络？ | Node 字段、Socket/IB 日志、正确性结果 |
| L2 | 真实插桩是否支持定位且不破坏 NCCL？ | 动态链接、真实 JSONL、延迟 RCA、开销报告 |

## 11. GitHub 工作流

### 11.1 Day 01—Day 17

- 每天至少一个通过测试的 commit 直接提交 `main`；
- commit 前运行统一检查脚本；
- 不提交失败测试、WIP、二进制或大型输出；
- 推荐格式：`type(scope): subject`。

### 11.2 Day 18—Day 26

- E05、Crater 集成和 E06 使用短分支；
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
5. 不把本机模拟成功写成 L2 完成；
6. 等待用户或管理员处理后继续同一任务。

任何范围、验收或顺序变化都必须按冻结计划提交变更记录，得到用户确认后执行。

## 13. Day 01 允许迁移的现有文件

计划确认后，只迁移以下正式 Markdown：

- `Mycroft_复现冻结计划.md`；
- `Mycroft_逐日开发计划.md`；
- `NCCL_源码学习日志.md`；
- `notes/阶段01_Communicator与AllReduce任务提交.md`；
- `notes/阶段01_综合测验_从应用调用到GPU网络执行.md`；
- `notes/阶段02_从Host任务到GPU_Ring_AllReduce执行.md`。
- `notes/NCCL_专有名词_英文简称与常见函数速查.md`。

不迁移：

- `参考答案/`；
- `experiments/e01_ring_dependency/` 中的旧 C 程序和二进制；
- `notes/lab1`、`notes/lab1.cpp`、`notes/learn`；
- 当前 master 版 `nccl/` 工作树；
- 本机绝对路径、截图、临时日志和构建产物。

## 14. 依据

- [Mycroft SOSP 2025 论文](https://minlanyu.seas.harvard.edu/writeup/sosp25.pdf)
- [Mycroft arXiv 页面](https://arxiv.org/abs/2509.03018)
- [NVIDIA NCCL 官方仓库](https://github.com/NVIDIA/nccl)
- [NVIDIA nccl-tests 官方仓库](https://github.com/NVIDIA/nccl-tests)
- [Crater 官方仓库](https://github.com/raids-lab/crater)
- [Crater Volcano 集成说明](https://raids-lab.github.io/crater/zh/docs/admin/deployment/volcano/)
- [Crater RDMA 说明](https://raids-lab.github.io/crater/zh/docs/admin/more/rdma/)
- [Volcano PyTorch 插件说明](https://volcano.sh/docs/userguide/user_guide_how_to_use_pytorch_plugin/)

## 15. v0.2 变更记录与当前执行边界

用户已确认原 v0.1 计划并启动 Day 01。随后用户提出先熟悉开发与运行工具，再进入 E01 知识和核心实现；本次 v0.2 调整已经用户明确确认。

v0.2 只改变顺序，不改变 E01—E06 的范围、总开发日数量或最终 L2 验收：

- 原 Day 18—20 的 Crater 页面、CPU 双 Pod、单 GPU 基线前移为 Day 02—04；
- 原 E01—E05 顺延为 Day 05—20；
- 双物理节点 Socket/RDMA 保持在 Day 21—22；
- E06 保持在 Day 23—26；
- Day 01—17 直接提交 `main`，Day 18 起对 E05/E06 底层代码使用分支和 PR。

当前只允许继续完成 Day 01 的文档审查、本地验收、首次 commit 和公开发布。只有用户明确说“Day 01 验收通过，进入 Day 02”后，才开始创建 Crater 教学材料或提交平台作业。

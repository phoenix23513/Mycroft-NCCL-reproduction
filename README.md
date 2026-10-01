# Mycroft NCCL Reproduction

一个以真实 NCCL 2.21.5 运行与插桩为后续实验主线的 Mycroft 复现项目。

本项目关注一个实际问题：当分布式训练中的某个 rank、GPU、Proxy 或网络环节变慢时，许多其他 rank 也会因为依赖关系表现为停滞。我们希望通过记录底层通信进度和恢复事件因果关系，区分最早的异常来源与随后被阻塞的受影响 rank。

项目已用 E01/E02 建立确定性回归夹具，并用 E03 建立版本化事件契约。从 Day 14 起不再先扩展离线模型：先构建并运行真实 NCCL 2.21.5，动态确认 completion/state 字段，再用真实轨迹建设共享内存通道和分析器，最终在 Crater 集群验证。

## 当前进度

- **E01 已完成**：4-rank、8-chunk Ring ReduceScatter/AllGather、JSONL 事件、单点延迟注入和因果传播。
- **E02 已完成**：发送侧归一化分析 fixture、三类合成停滞和论文四类快照状态分类均已通过验收；它不作为真实 NCCL 执行或插桩正确性的证据。
- **E03 已完成并完成契约纠错**：Event v1 保持可读，Event v2 补充真实单调时间与 operation completion；NCCL 2.21.5 字段目前仍是候选，Day 16—17 将用真实运行动态确认。
- **Day 14 真实构建与加载自检通过，待用户验收**：Crater 已从固定源码构建 NCCL 2.21.5，运行版本与实际加载路径核对通过，脱敏证据已保存。执行入口与结果见 [`instrumentation/nccl-2.21.5/README.md`](instrumentation/nccl-2.21.5/README.md)。
- Crater 上已经完成 CPU/Gloo 双进程 AllReduce 与单 GPU PyTorch/CUDA 环境验证。

详细任务与每日进度见 [`docs/plans/Mycroft_26日开发路线图.md`](docs/plans/Mycroft_26日开发路线图.md)。

## 已完成的历史回归演示

E01 是当前已完成、可直接运行的历史演示。它在 CPU 上模拟 Ring AllReduce，不需要 GPU、NCCL 或集群；它只用于理解和回归，不代表新路线的真实 NCCL 验收。

环境要求：

- Python 3.10+；
- 支持 C++17 的编译器；
- CMake 3.16+。

构建并运行测试：

```bash
cmake -S . -B .build
cmake --build .build
ctest --test-dir .build --output-on-failure
```

运行 Ring 模拟器：

```bash
./.build/experiments/e01_ring_dependency/e01_ring_sim
```

观察正常执行产生的 96 条事件：

```bash
./.build/experiments/e01_ring_dependency/e01_ring_sim --jsonl
```

在 ReduceScatter 的一个 send 事件上注入 100 个逻辑时间单位的延迟：

```bash
./.build/experiments/e01_ring_dependency/e01_ring_sim \
  --jsonl \
  --inject-delay reduce_scatter 0 1 0 send 100
```

延迟只在根事件上主动增加一次，之后沿消息依赖和同一 rank 的执行顺序传播；通信数值和 contributor mask 不会改变。示例输出位于 [`results/samples/e01/`](results/samples/e01/)。

还可以直接用浏览器打开 [`experiments/e01_ring_dependency/visualizer.html`](experiments/e01_ring_dependency/visualizer.html)，逐步观察 chunk 在 ReduceScatter 和 AllGather 中的移动与聚合。

## 复现路线

| 阶段 | 要解决的问题 | 主要产物 |
|---|---|---|
| E01 | Ring 通信中的数据流和因果依赖如何形成 | Ring 模拟器、事件轨迹、延迟传播 |
| E02 | 如何用统一累计量构造确定性的分析输入 | 发送侧归一化进度 fixture |
| E03 | 如何跨 communicator、operation 和 channel 唯一识别事件 | 兼容 Event v1 的 Event v2 与 NCCL 源码字段映射 |
| E04 | 如何分析真实 NCCL 轨迹中的停滞和根因候选 | Trigger、MinOp/MinData 和 RCA |
| E05 | 如何低开销传递真实 NCCL 运行时事件 | 共享内存循环缓冲区与 reader |
| E06 | 真实 NCCL 能否产生可分析事件并复现确定性异常 | NCCL 2.21.5 构建、插桩与双节点验证 |

阶段编号表示能力边界，不再表示“先分析、后接入”的执行顺序。当前顺序是：E06 真实 NCCL 构建与 tracepoint → E05 共享内存与 reader → E04 基于真实轨迹的 Trigger/RCA → 双节点 Socket/RDMA。E01/E02 只保留为单元回归，不能作为 Day 14 之后的实验验收证据。

## 仓库中有什么

- [`experiments/`](experiments/)：E01、E02 确定性回归夹具；
- [`docs/plans/`](docs/plans/)：当前 26 日开发路线及其变更记录；
- [`notes/nccl/`](notes/nccl/)：与实现相关的 NCCL 源码学习笔记；
- [`cluster/crater/`](cluster/crater/)：Crater 探针、作业配置和平台实验资料；
- [`src/mycroft/`](src/mycroft/)：后续统一事件和分析代码；
- [`runtime/`](runtime/)：后续共享内存事件通道；
- [`instrumentation/`](instrumentation/)：后续 NCCL 2.21.5 插桩记录；
- [`results/samples/`](results/samples/)：可公开、可复查的小型结果样例。

## 复现范围

这是对 Mycroft 核心思路的独立、小规模复现，不是原系统的源码移植，也不是生产级监控平台。项目不会复刻 Kafka、云数据库、Web 后端或论文中的完整大规模硬件实验；实验结果只用于验证实现和分析思路。

## English summary

This repository independently reproduces the core path of Mycroft-style NCCL stall diagnosis. The remaining work now starts from project-built NCCL 2.21.5, validates real completion and state traces, and only then builds the shared-memory and analysis path.

E01 provides a deterministic Ring AllReduce simulator, JSONL traces, and causal delay propagation. E02 provides a deterministic normalized send-side fixture for analysis tests, not an execution model of NCCL. E03 keeps Event v1 readable while Event v2 adds process-local monotonic time and operation-completion records. Starting with Day 14, every experimental acceptance must use the project-built NCCL 2.21.5: build and tracepoint validation now precede the shared-memory transport and E04 analysis. E01 and E02 remain deterministic unit-test fixtures only. See the [26-day development roadmap](docs/plans/Mycroft_26日开发路线图.md) for the current plan.

## License

Unless stated otherwise for third-party material, this project is licensed under the [Apache License 2.0](LICENSE). Third-party components retain their original licenses and notices.

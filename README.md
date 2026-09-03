# Mycroft NCCL Reproduction

[中文](#中文) | [English](#english)

## 中文

### 项目定位

这是一个独立实现的系统研究复现项目，目标是逐步构建 Mycroft 风格的 NCCL 通信停滞检测与根因分析原型。开发主线是完成可运行的 E01—E06；NCCL、Docker、PyTorch、Crater 和分布式系统知识按开发需要补充。

当前状态：**Day 01 本地验收通过，等待首次发布到 GitHub**。

### 复现边界

- 不复制 Mycroft 原项目实现或本地“参考答案”；
- E01—E04 先在可控模型和轨迹上建立因果分析；
- E05 建立独立 reader 使用的共享内存事件通道；
- E06 接入固定版本 NCCL 2.21.5，并在 Crater 上完成双物理节点验证；
- 小规模实验结果只用于复现验证，不外推为生产环境性能结论。

### 快速检查

需要 Python 3.10+、支持 C++17 的编译器；推荐安装 CMake 3.16+。

```bash
./scripts/check.sh
```

脚本运行 Python smoke test，并优先使用 CMake/CTest 构建 C++ smoke test。未安装 CMake 时，会临时使用 `c++` 完成同等的 Day 01 检查。

### 仓库导航

- `docs/plans/`：冻结计划与逐日开发计划；
- `notes/nccl/`：已确认可迁移的 NCCL 正式笔记；
- `experiments/`：E01/E02 等逐阶段实验；
- `src/mycroft/`：Python schema、trace 和 analysis；
- `runtime/`：E05 C++ 共享内存运行时；
- `instrumentation/`：NCCL 2.21.5 插桩说明与 patch；
- `cluster/crater/`：脱敏后的 Crater 探测、镜像和作业配置；
- `tests/`：跨模块测试、fixture 与 expected 输出；
- `results/samples/`：可公开的小型结果样例。

### 安全边界

不得提交 GitHub token、Harbor 凭据、集群 cookie、SSH key、内部域名、真实节点名、个人绝对路径、大型轨迹或构建产物。

## English

### Project scope

This repository is an independent systems reproduction project. Its development goal is to build an E01–E06 prototype for Mycroft-style NCCL communication-stall detection and root-cause analysis. NCCL and cluster knowledge is learned just in time as required by implementation.

Current status: **Day 01 locally validated, pending the first GitHub publication**.

### Reproduction boundaries

- Do not copy an existing Mycroft implementation or local reference solutions.
- Build causal analysis on controlled models and traces in E01–E04.
- Build a shared-memory event path with an independent reader in E05.
- Instrument pinned NCCL 2.21.5 and validate on two physical Crater nodes in E06.
- Treat small experiments as reproduction evidence, not production performance claims.

### Quick check

Python 3.10+ and a C++17 compiler are required. CMake 3.16+ is recommended.

```bash
./scripts/check.sh
```

The script runs Python and C++ smoke tests. It uses CMake/CTest when available and falls back to the C++ compiler for the Day 01 smoke build.

### License

Unless stated otherwise for third-party material, this project is licensed under the Apache License 2.0. Third-party components retain their original licenses and notices.

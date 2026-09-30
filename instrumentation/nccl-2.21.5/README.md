# NCCL 2.21.5 Instrumentation

Day 13 使用官方 NCCL submodule 做源码映射；Day 23—26 才添加可复查的插桩 patch。

固定基线：

```text
repository: https://github.com/NVIDIA/nccl.git
tag:        v2.21.5-1
commit:     ab2b89c4c339bd7f816fbc114a4b05d386b66290
```

克隆主仓库后初始化源码：

```bash
git submodule update --init --recursive third_party/nccl
git -C third_party/nccl describe --tags --exact-match
git -C third_party/nccl rev-parse HEAD
```

预期分别输出 `v2.21.5-1` 和上述完整 commit。NCCL 保留其上游许可证，源码不采用本项目 Apache-2.0 重新许可。

文档入口：

- [`tracepoints.md`](tracepoints.md)：候选插桩点、能观察的状态和 E06 动态验证清单；
- [`../../docs/architecture/event-field-sources.md`](../../docs/architecture/event-field-sources.md)：论文字段、Event v1 与 NCCL 成员之间的映射及限制。

Day 13 不包含 NCCL 修改或编译结果；候选点只有经过 E06 实机验证后才能升级为已确认插桩语义。

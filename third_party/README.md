# Third-party Sources

`third_party/nccl/` 是官方 NVIDIA NCCL submodule，固定在 tag `v2.21.5-1`、commit `ab2b89c4c339bd7f816fbc114a4b05d386b66290`，用于 Day 13 源码映射和后续 E06 插桩。

初始化方式：

```bash
git submodule update --init --recursive third_party/nccl
```

该目录保留 NCCL 自身的 `LICENSE.txt` 和上游版权；本项目的 Apache-2.0 许可证不覆盖第三方源码。

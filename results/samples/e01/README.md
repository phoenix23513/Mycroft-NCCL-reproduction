# E01 Day 07 验收记录

完成核心逻辑后，在本目录保存两份公开安全的样例：

- normal.jsonl：无延迟的 96 条事件；
- rank1-delay.jsonl：只向固定根事件注入 100 ticks 后的事件。

验收记录：

- [x] Day 07 定向测试及原有 Day 01—07 回归测试通过；
- [x] 两次运行的 AllReduce 数值和 contributor mask 相同；
- [x] rank1-delay.jsonl 中恰好一个 injected_root；
- [x] rank 2 至少有一个 affected 事件；
- [x] 能指出一条 root → recv → 后续 send 的传播链；
- [x] 能解释最晚完成的 rank 为什么可能只是受害者。

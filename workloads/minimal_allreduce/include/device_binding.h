#pragma once

#include <stdexcept>

namespace day15 {
// 保留 Day15 的同 Pod 双 GPU 模式；M1 两个 Pod 各只暴露一张 GPU。
inline int select_device(int rank, int visible_devices, bool single_gpu_node) {
  if (rank < 0 || rank > 1) throw std::invalid_argument("rank must be 0 or 1");
  if (single_gpu_node) {
    if (visible_devices != 1)
      throw std::runtime_error("M1 requires exactly one visible GPU per Pod");
    return 0;
  }
  if (visible_devices < 2)
    throw std::runtime_error("Day15 requires two visible GPUs in one Pod");
  return rank;
}
}  // namespace day15

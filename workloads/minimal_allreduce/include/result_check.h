#pragma once

#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace day15 {
// 当前输入及两 rank 的和都是 float 可精确表示的小整数，使用精确比较。
// NaN != expected 为 true；漏写、无穷大和上一 operation 的结果也会被拒绝。
inline void verify_result(const std::vector<float>& output, float expected) {
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (output[index] != expected) {
      std::ostringstream message;
      message << std::setprecision(std::numeric_limits<float>::max_digits10)
              << "AllReduce result mismatch: index=" << index
              << " actual=" << output[index] << " expected=" << expected;
      throw std::runtime_error(message.str());
    }
  }
}
}  // namespace day15

#pragma once

#include <chrono>
#include <cstddef>
#include <string>

namespace day15 {
// 在调用方指定的新目录传递原始 ID 字节，不负责传输 collective 数据。
// Day15 使用同 Pod 私有临时目录；M1 使用已验证跨 Pod 共享的本次运行目录。
// 发布完成前目标文件不可见；已有目标文件会被拒绝，不覆盖旧运行的 ID。
void publish_id(const std::string& path, const void* bytes, std::size_t size);
void read_id(const std::string& path, void* bytes, std::size_t size,
             std::chrono::milliseconds timeout = std::chrono::seconds(30));
}  // namespace day15

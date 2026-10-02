#include "bootstrap.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace day15 {
namespace {
struct File {
  int fd;
  ~File() { if (fd >= 0) ::close(fd); }
};

void fail(const char* operation) {
  throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}
}  // namespace

void publish_id(const std::string& path, const void* bytes, std::size_t size) {
  if (!bytes || size == 0) throw std::invalid_argument("empty bootstrap ID");
  const std::string staging = path + ".pending." + std::to_string(::getpid());
  File file{::open(staging.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600)};
  if (file.fd < 0) fail("create bootstrap staging file");
  try {
    auto* data = static_cast<const char*>(bytes);
    std::size_t written = 0;
    while (written < size) {
      const auto n = ::write(file.fd, data + written, size - written);
      if (n < 0 && errno == EINTR) continue;
      if (n < 0) fail("write bootstrap ID");
      if (n == 0) throw std::runtime_error("bootstrap write made no progress");
      written += static_cast<std::size_t>(n);
    }
    // link 是原子发布，且目标已存在时失败；读者不会看到半写入的 ID。
    if (::link(staging.c_str(), path.c_str()) != 0) fail("publish bootstrap ID");
  } catch (...) {
    ::unlink(staging.c_str());
    throw;
  }
  ::unlink(staging.c_str());
}

void read_id(const std::string& path, void* bytes, std::size_t size,
             std::chrono::milliseconds timeout) {
  if (!bytes || size == 0 || timeout.count() <= 0)
    throw std::invalid_argument("invalid bootstrap read parameters");
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  File file{-1};
  while (file.fd < 0) {
    file.fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK);
    if (file.fd >= 0) break;
    if (errno != ENOENT && errno != EINTR) fail("open bootstrap ID");
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error("bootstrap timeout: rank 0 has not published an ID");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  struct stat info{};
  if (::fstat(file.fd, &info) != 0) fail("stat bootstrap ID");
  if (!S_ISREG(info.st_mode) || info.st_size != static_cast<off_t>(size))
    throw std::runtime_error("bootstrap ID must be a regular file of the expected size");
  auto* data = static_cast<char*>(bytes);
  std::size_t received = 0;
  while (received < size) {
    const auto n = ::read(file.fd, data + received, size - received);
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) fail("read bootstrap ID");
    if (n == 0) throw std::runtime_error("truncated bootstrap ID");
    received += static_cast<std::size_t>(n);
  }
}
}  // namespace day15

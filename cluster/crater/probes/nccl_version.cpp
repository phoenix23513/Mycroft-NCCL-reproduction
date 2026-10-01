// Day 14: load a specific NCCL shared library without initializing CUDA or a GPU.
#include <dlfcn.h>
#include <stdlib.h>

#include <iostream>
#include <memory>
#include <string>

namespace {
constexpr int kExpectedVersion = 22105;

std::string canonical_path(const char* path) {
  std::unique_ptr<char, decltype(&free)> resolved(realpath(path, nullptr), free);
  return resolved ? std::string(resolved.get()) : std::string();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: nccl_version /path/to/project/libnccl.so\n";
    return 2;
  }
  const auto expected_library = canonical_path(argv[1]);
  if (expected_library.empty()) {
    std::cerr << "Cannot resolve requested library: " << argv[1] << '\n';
    return 1;
  }
  void* library = dlopen(expected_library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    std::cerr << "dlopen failed: " << dlerror() << '\n';
    return 1;
  }
  const auto close_library = [](void* value) { dlclose(value); };
  std::unique_ptr<void, decltype(close_library)> handle(library, close_library);
  dlerror();
  void* symbol = dlsym(handle.get(), "ncclGetVersion");
  const char* error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    std::cerr << "ncclGetVersion lookup failed: "
              << (error ? error : "null symbol") << '\n';
    return 1;
  }
  Dl_info info{};
  if (dladdr(symbol, &info) == 0 || info.dli_fname == nullptr) {
    std::cerr << "Cannot identify library providing ncclGetVersion\n";
    return 1;
  }
  const auto loaded_library = canonical_path(info.dli_fname);
  std::cout << "requested_library=" << expected_library << '\n'
            << "loaded_library=" << loaded_library << '\n';
  if (loaded_library != expected_library) {
    std::cerr << "ncclGetVersion came from a different library\n";
    return 1;
  }
  // NCCL's ncclResult_t is an int-sized enum on the supported Linux ABI.
  // POSIX permits converting the dlsym result to the function pointer type.
  using GetVersion = int (*)(int*);
  const auto get_version = reinterpret_cast<GetVersion>(symbol);
  int version = 0;
  const int result = get_version(&version);
  if (result != 0) {  // ncclSuccess
    std::cerr << "ncclGetVersion failed: result=" << result << '\n';
    return 1;
  }
  std::cout << "nccl_version_code=" << version << '\n'
            << "nccl_version=" << version / 10000 << '.'
            << (version % 10000) / 100 << '.' << version % 100 << '\n';
  if (version != kExpectedVersion) {
    std::cerr << "Expected NCCL 2.21.5 (22105)\n";
    return 1;
  }
  std::cout << "status=PASS\n";
  return 0;
}

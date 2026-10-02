#include "bootstrap.h"
#include "result_check.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include <dlfcn.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(NCCL_VERSION_CODE == 22105, "Day15 requires NCCL 2.21.5 headers");

namespace {
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void nccl_check(ncclResult_t status) {
  if (status != ncclSuccess) throw std::runtime_error(ncclGetErrorString(status));
}

int integer(const char* text, int minimum, int maximum) {
  int result = 0;
  const std::string value(text);
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
      result < minimum || result > maximum)
    throw std::invalid_argument("invalid integer argument: " + value);
  return result;
}

// 这些是应用侧阶段标记；operation 是应用序号，尚未映射 NCCL CollOp 身份。
// time_ns 只用于本进程的阶段差值，不是 GPU 的精确执行时间。
void mark(int rank, int operation, std::size_t count, const char* stage) {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  std::cout << "rank=" << rank << " operation=" << operation << " count=" << count
            << " stage=" << stage << " time_ns="
            << std::chrono::duration_cast<std::chrono::nanoseconds>(now).count() << '\n';
}

void verify_library() {
  int version = 0;
  nccl_check(ncclGetVersion(&version));
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&ncclGetVersion), &info) || !info.dli_fname)
    throw std::runtime_error("cannot identify the loaded NCCL library");
  const auto loaded = std::filesystem::canonical(info.dli_fname);
  const auto expected = std::filesystem::canonical(DAY15_EXPECTED_NCCL_LIBRARY);
  std::cout << "requested_library=" << expected.string() << '\n'
            << "loaded_library=" << loaded.string() << '\n'
            << "nccl_version_code=" << version << '\n';
  if (version != 22105 || loaded != expected)
    throw std::runtime_error("loaded NCCL is not the specified Day14 library");
}

// 提交 Sum AllReduce；成功返回表示提交成功，完成由下一函数等待。
void submit_allreduce(float* send, float* recv, std::size_t count,
                      ncclComm_t comm, cudaStream_t stream) {
  nccl_check(ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream));
}

// 等待本地 stream 上此前的输入准备和通信工作完成。
void wait_for_collective(cudaStream_t stream) {
  cuda_check(cudaStreamSynchronize(stream));
}

struct Runtime {
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
  float* send = nullptr;
  float* recv = nullptr;
  bool finished = false;
  ~Runtime() {
    if (comm) {
      if (finished) ncclCommDestroy(comm);
      else ncclCommAbort(comm);
    }
    if (send) cudaFree(send);
    if (recv) cudaFree(recv);
    if (stream) cudaStreamDestroy(stream);
  }
};
}  // namespace

int main(int argc, char** argv) {
  std::cout << std::unitbuf;
  std::cerr << std::unitbuf;
  if (argc != 3 && argc != 4) {
    std::cerr << "Usage: day15_allreduce RANK ID_FILE [ITERATIONS>=3]\n";
    return 2;
  }
  int rank = -1;
  try {
    rank = integer(argv[1], 0, 1);
    const int iterations = argc == 4 ? integer(argv[3], 3, 1000) : 3;
    verify_library();

    int devices = 0;
    cuda_check(cudaGetDeviceCount(&devices));
    if (devices < 2) throw std::runtime_error("Day15 requires two visible GPUs in one Pod");
    cuda_check(cudaSetDevice(rank));
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, rank));
    std::cout << "rank=" << rank << " device=" << rank
              << " gpu_name=" << properties.name << '\n';

    ncclUniqueId id{};
    mark(rank, -1, 0, "bootstrap_begin");
    if (rank == 0) {
      nccl_check(ncclGetUniqueId(&id));
      day15::publish_id(argv[2], &id, sizeof(id));
    } else {
      day15::read_id(argv[2], &id, sizeof(id));
    }
    mark(rank, -1, 0, "bootstrap_return");

    Runtime runtime;
    mark(rank, -1, 0, "comm_init_begin");
    nccl_check(ncclCommInitRank(&runtime.comm, 2, id, rank));
    mark(rank, -1, 0, "comm_init_return");
    cuda_check(cudaStreamCreateWithFlags(&runtime.stream, cudaStreamNonBlocking));

    int operation = 0;
    for (const std::size_t count : {4u, 4096u, 1048576u}) {
      const auto bytes = count * sizeof(float);
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&runtime.send), bytes));
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&runtime.recv), bytes));
      std::vector<float> input(count), output(count);
      for (int iteration = 0; iteration < iterations; ++iteration, ++operation) {
        const float input_value = static_cast<float>(rank + 1 + operation);
        const float expected = static_cast<float>(3 + 2 * operation);
        std::fill(input.begin(), input.end(), input_value);
        cuda_check(cudaMemcpyAsync(runtime.send, input.data(), bytes,
                                   cudaMemcpyHostToDevice, runtime.stream));
        // 为未写完的输出填充 NaN，便于发现漏写；这是初始状态，不是完成证据。
        cuda_check(cudaMemsetAsync(runtime.recv, 0xff, bytes, runtime.stream));

        mark(rank, operation, count, "api_begin");
        submit_allreduce(runtime.send, runtime.recv, count, runtime.comm, runtime.stream);
        mark(rank, operation, count, "api_return");
        wait_for_collective(runtime.stream);
        mark(rank, operation, count, "stream_sync_return");
        cuda_check(cudaMemcpy(output.data(), runtime.recv, bytes, cudaMemcpyDeviceToHost));
        day15::verify_result(output, expected);
        std::cout << "rank=" << rank << " operation=" << operation
                  << " count=" << count << " expected=" << expected << " result=PASS\n";
      }
      cuda_check(cudaFree(runtime.send));
      runtime.send = nullptr;
      cuda_check(cudaFree(runtime.recv));
      runtime.recv = nullptr;
    }
    runtime.finished = true;
    std::cout << "rank=" << rank << " operations=" << operation << " status=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rank=" << rank << " status=FAILED error=" << error.what() << '\n';
    return 1;
  }
}

#include "bootstrap.h"
#include "device_binding.h"
#include "result_check.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include <dlfcn.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

static_assert(NCCL_VERSION_CODE == 22105, "Day15 requires NCCL 2.21.5 headers");

namespace {
// NCCL debug.cc 同样以一次 fwrite 输出一条记录；共用 stdout 的 FILE 锁，
// 避免 unitbuf 下多次 operator<< 写入被 Proxy TRACE 插在字段之间。
void print_line(std::string record) {
  record += '\n';
  if (std::fwrite(record.data(), 1, record.size(), stdout) != record.size() ||
      std::fflush(stdout) != 0)
    throw std::runtime_error("cannot write application log");
}

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
  timespec now{};
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    throw std::runtime_error("cannot read application monotonic clock");
  std::ostringstream record;
  record << "rank=" << rank << " operation=" << operation << " count=" << count
         << " stage=" << stage << " time_ns="
         << (static_cast<std::uint64_t>(now.tv_sec)*1000000000ull + now.tv_nsec);
  print_line(record.str());
}

void verify_library() {
  int version = 0;
  nccl_check(ncclGetVersion(&version));
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&ncclGetVersion), &info) || !info.dli_fname)
    throw std::runtime_error("cannot identify the loaded NCCL library");
  const auto loaded = std::filesystem::canonical(info.dli_fname);
  const auto expected = std::filesystem::canonical(DAY15_EXPECTED_NCCL_LIBRARY);
  print_line("requested_library=" + expected.string());
  print_line("loaded_library=" + loaded.string());
  print_line("nccl_version_code=" + std::to_string(version));
  if (version != 22105 || loaded != expected)
    throw std::runtime_error("loaded NCCL is not the specified project-built library");
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

struct M2Capture {
  using Start = int (*)(void*, void*, std::uint64_t, std::uint64_t);
  using Complete = int (*)(void*);
  using Finish = int (*)(const char*, const char*);
  void* handle = nullptr;
  Complete complete_function = nullptr;
  Finish finish_function = nullptr;
  using DelayConfigure = int (*)(int, int, std::uint64_t, int, std::uint64_t);
  using DelayFinish = int (*)(const char*);
  DelayFinish delay_finish = nullptr;
  bool started = false;
  std::string capture_id, output;
  ~M2Capture() { if (handle) dlclose(handle); }
  void start(ncclComm_t comm, cudaStream_t stream, int rank) {
    const char* enabled = std::getenv("MYCROFT_M2_ENABLE");
    if (!enabled || std::string(enabled) != "1") return;
    const char* id = std::getenv("MYCROFT_M2_CAPTURE_ID");
    const char* destination = std::getenv("MYCROFT_M2_OUTPUT");
    if (!id || !destination) throw std::runtime_error("M2 capture ID/output missing");
    capture_id = id; output = destination;
    handle = dlopen(DAY15_EXPECTED_NCCL_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error("cannot open specified M2 NCCL library");
    auto start_function = reinterpret_cast<Start>(dlsym(handle, "mycroftM2Start"));
    complete_function = reinterpret_cast<Complete>(dlsym(handle, "mycroftM2Complete"));
    finish_function = reinterpret_cast<Finish>(dlsym(handle, "mycroftM2Finish"));
    if (!start_function || !complete_function || !finish_function)
      throw std::runtime_error("specified NCCL lacks M2 instrumentation symbols");
    if (const char* delay = std::getenv("MYCROFT_M3_DELAY_NS")) {
      const char* target_rank = std::getenv("MYCROFT_M3_RANK");
      const char* target_op = std::getenv("MYCROFT_M3_OPERATION");
      const char* target_channel = std::getenv("MYCROFT_M3_CHANNEL");
      if (!target_rank || !target_op || !target_channel)
        throw std::runtime_error("M3 target parameters missing");
      auto configure = reinterpret_cast<DelayConfigure>(dlsym(handle, "mycroftM3Configure"));
      delay_finish = reinterpret_cast<DelayFinish>(dlsym(handle, "mycroftM3Finish"));
      if (!configure || !delay_finish || configure(rank, integer(target_rank, 0, 1),
          integer(target_op, 0, 8), integer(target_channel, 0, 1), integer(delay, 0, 2000000000)) != 0)
        throw std::runtime_error("M3 delay configuration/library rejected");
    }
    if (start_function(comm, stream, 262144, 10000) != 0)
      throw std::runtime_error("M2 recorder initialization/scope rejected");
    started = true;
    print_line("m2_capture=STARTED sample_period_ns=10000 capacity=262144");
  }
  void complete(ncclComm_t comm) {
    if (started && complete_function(comm) != 0)
      throw std::runtime_error("M2 actual-stream completion evidence rejected");
  }
  int finish() noexcept {
    if (!started) return 0;
    started = false;
    const int result = finish_function(capture_id.c_str(), output.c_str());
    // Successful M2 export proves producer join before reading the injection record.
    return result == 0 && delay_finish ? delay_finish(output.c_str()) : result;
  }
};

struct Runtime {
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
  float* send = nullptr;
  float* recv = nullptr;
  bool finished = false;
  bool producer_cleanup_failed = false;
  M2Capture capture;
  ncclResult_t close_comm() noexcept {
    if (!comm) return producer_cleanup_failed ? ncclSystemError : ncclSuccess;
    // Do not retry a failed teardown on a possibly already released communicator.
    const auto old_comm = comm;
    comm = nullptr;
    const auto result = finished ? ncclCommDestroy(old_comm) : ncclCommAbort(old_comm);
    producer_cleanup_failed = result != ncclSuccess;
    return result;
  }
  ~Runtime() {
    const auto cleanup = close_comm();
    // Only freeze storage after NCCL has stopped its producers.
    if (cleanup != ncclSuccess)
      std::fputs("m2_capture_finalize=SKIPPED reason=communicator_cleanup_failed\n", stderr);
    else if (capture.finish() != 0)
      std::fputs("m2_capture_finalize=FAILED\n", stderr);
    if (send) cudaFree(send);
    if (recv) cudaFree(recv);
    if (stream) cudaStreamDestroy(stream);
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4 && argc != 5) {
    std::fputs("Usage: day15_allreduce RANK ID_FILE [ITERATIONS>=3 [--single-gpu-node]]\n", stderr);
    return 2;
  }
  int rank = -1;
  try {
    rank = integer(argv[1], 0, 1);
    const int iterations = argc >= 4 ? integer(argv[3], 3, 1000) : 3;
    const bool single_gpu_node = argc == 5;
    if (single_gpu_node && std::string(argv[4]) != "--single-gpu-node")
      throw std::invalid_argument("unknown execution mode");
    verify_library();
    print_line("application_clock=CLOCK_MONOTONIC");

    int devices = 0;
    cuda_check(cudaGetDeviceCount(&devices));
    const int device = day15::select_device(rank, devices, single_gpu_node);
    cuda_check(cudaSetDevice(device));
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device));
    print_line("rank=" + std::to_string(rank) + " device=" + std::to_string(device) +
               " gpu_name=" + properties.name);

    ncclUniqueId id{};
    mark(rank, -1, 0, "bootstrap_begin");
    if (rank == 0) {
      nccl_check(ncclGetUniqueId(&id));
      day15::publish_id(argv[2], &id, sizeof(id));
    } else {
      day15::read_id(argv[2], &id, sizeof(id), single_gpu_node ?
                     std::chrono::seconds(120) : std::chrono::seconds(30));
    }
    mark(rank, -1, 0, "bootstrap_return");

    Runtime runtime;
    mark(rank, -1, 0, "comm_init_begin");
    nccl_check(ncclCommInitRank(&runtime.comm, 2, id, rank));
    mark(rank, -1, 0, "comm_init_return");
    cuda_check(cudaStreamCreateWithFlags(&runtime.stream, cudaStreamNonBlocking));
    runtime.capture.start(runtime.comm, runtime.stream, rank);

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
        runtime.capture.complete(runtime.comm);
        mark(rank, operation, count, "stream_sync_return");
        cuda_check(cudaMemcpy(output.data(), runtime.recv, bytes, cudaMemcpyDeviceToHost));
        day15::verify_result(output, expected);
        std::ostringstream record;
        record << "rank=" << rank << " operation=" << operation
               << " count=" << count << " expected=" << expected << " result=PASS";
        print_line(record.str());
      }
      cuda_check(cudaFree(runtime.send));
      runtime.send = nullptr;
      cuda_check(cudaFree(runtime.recv));
      runtime.recv = nullptr;
    }
    runtime.finished = true;
    // Successful capture exports only after blocking communicator destruction stops Proxy producers.
    nccl_check(runtime.close_comm());
    if (runtime.capture.finish() != 0) throw std::runtime_error("M2 capture export failed");
    print_line("rank=" + std::to_string(rank) + " operations=" + std::to_string(operation) + " status=PASS");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "rank=%d status=FAILED error=%s\n", rank, error.what());
    return 1;
  }
}

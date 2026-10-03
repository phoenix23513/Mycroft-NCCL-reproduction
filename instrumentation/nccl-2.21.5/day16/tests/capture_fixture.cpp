// CPU recorder fixture only. Times/progress/completions below are handwritten, not NCCL evidence.
#include "operation_trace.h"
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace mycroft_trace::day16;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void saved(CaptureStatus status) { require(status == CaptureStatus::Recorded, "record not saved"); }
OperationKey operation{0xabc, 2, 0};
ConnectionRef send_ref{operation, 1, 10}, recv_ref{operation, 1, 11};
SendProgress send_at(std::uint64_t ns) {
  return {send_ref, ns, 64, 16, 2, false, 8, 6, 4};
}
ReceiveProgress recv_at(std::uint64_t ns) {
  return {recv_ref, ns, 64, 16, 2, false, 8, 6, 4, 2};
}
void context(const std::string& scenario) {
  saved(record_operation_begin({operation, 100, 4096}));
  if (scenario == "duplicate_begin") saved(record_operation_begin({operation, 101, 2048}));
  if (scenario != "missing_channel") saved(record_channel_membership({operation, 1}));
  if (scenario != "orphan_peer")
    saved(record_peer_connection({send_ref, 1, scenario == "wrong_direction" ? Direction::Receive : Direction::Send,
                                  Transport::NetIb}));
  saved(record_peer_connection({recv_ref, 1, Direction::Receive, Transport::NetIb}));
  if (scenario == "ambiguous_send")
    saved(record_peer_connection({{operation, 1, 12}, 1, Direction::Send, Transport::NetIb}));
  if (scenario != "missing_plan") saved(record_plan_binding({operation, 7, 4, 1}));
}
int main(int argc, char** argv) {
  try {
    require(argc == 3, "scenario output_directory required");
    const std::string scenario = argv[1];
    const char* output = argv[2];
    CaptureStats stats;
    stats.recorded = 123;
    require(read_capture_stats(stats) == CaptureStatus::InvalidState && stats.recorded == 123,
            "uninitialized stats must not overwrite output");
    require(record_operation_begin({operation, 100, 4096}) == CaptureStatus::InvalidState, "uninitialized record");
    require(initialize_capture({0, 10}) == CaptureStatus::InvalidArgument, "zero capacity");
    require(initialize_capture({10, 0}) == CaptureStatus::InvalidArgument, "zero period");
    saved(initialize_capture({scenario == "overflow" ? 6u : scenario == "concurrent" ? 2000u : 64u, 10}));
    require(initialize_capture({64, 10}) == CaptureStatus::InvalidState, "running reinitialize");
    require(export_capture("cpu_fixture", output) == CaptureStatus::InvalidState, "running export");
    if (scenario == "overflow" || scenario == "concurrent") {
      std::vector<std::thread> workers;
      std::atomic<bool> failure{false};
      for (std::uint64_t t = 0; t < 4; ++t) workers.emplace_back([t, &failure] {
        for (std::uint64_t i = 0; i < 500; ++i) {
          const auto status = record_operation_begin({{0xabc, t*500+i, 0}, 100+i, 4096});
          if (status != CaptureStatus::Recorded && status != CaptureStatus::Dropped) failure.store(true);
        }
      });
      for (auto& worker : workers) worker.join();
      require(!failure.load(), "concurrent writes failed");
    } else if (scenario == "multi_rank") {
      saved(record_operation_begin({operation, 100, 4096}));
      saved(record_operation_begin({{0xabc, 2, 1}, 100, 4096}));
    } else if (scenario != "empty") {
      context(scenario);
      SamplingState sending, receiving;
      auto first = send_at(120);
      saved(sample_send_progress(first, sending));
      auto early = send_at(129);
      require(sample_send_progress(early, sending) == CaptureStatus::Skipped, "period not enforced");
      auto repeated = send_at(130);
      if (scenario == "counter_regression") {
        repeated.gpu_ready_steps = 4; repeated.transmitted_steps = 2; repeated.done_steps = 0;
      }
      saved(sample_send_progress(repeated, sending));
      saved(sample_receive_progress(recv_at(120), receiving));
      saved(sample_receive_progress(recv_at(130), receiving));
      if (scenario == "normal") {
        auto progressed = send_at(140);
        progressed.gpu_ready_steps = 16; progressed.transmitted_steps = 12; progressed.done_steps = 8;
        saved(sample_send_progress(progressed, sending));
        progressed.observed_at_ns = 150;
        progressed.transmitted_steps = progressed.done_steps = 16;
        saved(sample_send_progress(progressed, sending));
      }
      if (scenario == "invalid") {
        auto bad = send_at(140); bad.done_steps = 8;
        require(record_send_progress(bad) == CaptureStatus::InvalidArgument, "counter order");
        bad = send_at(140); bad.nsteps = 15;
        require(record_send_progress(bad) == CaptureStatus::InvalidArgument, "fractional slice");
        require(sample_send_progress(send_at(128), sending) == CaptureStatus::InvalidArgument, "clock regression");
        require(sample_receive_progress(recv_at(125), receiving) == CaptureStatus::InvalidArgument, "recv clock regression");
        require(record_peer_connection({send_ref, 0, Direction::Send, Transport::NetIb}) ==
                CaptureStatus::InvalidArgument, "self peer");
        bad = send_at(140); bad.registered_buffer = true;
        require(record_send_progress(bad) == CaptureStatus::Unsupported, "registered send");
        auto bad_recv = recv_at(140); bad_recv.registered_buffer = true;
        require(record_receive_progress(bad_recv) == CaptureStatus::Unsupported, "registered receive");
        require(record_plan_binding({operation, 8, 4, 2}) == CaptureStatus::Unsupported, "multi-op plan");
        require(record_peer_connection({send_ref, 1, Direction::Send, Transport::Unknown}) ==
                CaptureStatus::Unsupported, "unverified transport");
      }
      saved(record_operation_completion({operation, 180, 7, scenario == "wrong_stream" ? 99u : 4u}));
      if (scenario == "duplicate_completion") saved(record_operation_completion({operation, 181, 7, 4}));
    }
    saved(stop_capture());
    require(stop_capture() == CaptureStatus::InvalidState, "double stop");
    require(record_operation_begin({operation, 200, 4096}) == CaptureStatus::InvalidState, "record after stop");
    saved(read_capture_stats(stats));
    if (scenario == "overflow") require(stats.recorded == 6 && stats.dropped == 1994, "drops accounting");
    if (scenario == "concurrent") require(stats.recorded == 2000 && stats.dropped == 0, "concurrent accounting");
    if (scenario == "invalid") require(stats.invalid == 5 && stats.unsupported == 4, "invalid accounting");
    require(export_capture("bad\"id", output) == CaptureStatus::InvalidArgument, "unsafe capture id");
    if (scenario == "multi_rank") {
      require(export_capture("cpu_fixture", output) == CaptureStatus::Unsupported, "mixed-rank scope");
    } else {
      saved(export_capture("cpu_fixture", output));
      require(export_capture("cpu_fixture", output) == CaptureStatus::InvalidArgument, "must not overwrite");
    }
    std::cout << "fixture=CPU_RECORDER_ONLY recorded=" << stats.recorded << " dropped=" << stats.dropped << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}

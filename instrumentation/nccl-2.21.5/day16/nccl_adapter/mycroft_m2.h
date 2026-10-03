#pragma once
// Project-owned host hooks for pinned NCCL 2.21.5; C++11-compatible (also included by .cu).
#include <stdint.h>
#include <cuda_runtime.h>
struct ncclComm;
struct ncclKernelPlan;
struct ncclInfo;
struct ncclChannel;
struct ncclConnector;
struct ncclProxyOp;
struct ncclProxyArgs;
struct ncclProxySubArgs;
struct ncclProxyState;
struct MycroftM2Tag {
  int enabled;
  uint64_t comm_hash;
  uint64_t op_seq;
  int rank;
  int peer;
  uint64_t connection_id;
};
struct MycroftM2Sampling {
  bool initialized;
  uint64_t last_observed;
  uint64_t last_sampled;
};
void mycroftM2Collective(ncclComm*, ncclKernelPlan*, ncclInfo*);
void mycroftM2Launched(ncclComm*, ncclKernelPlan*, cudaStream_t);
void mycroftM2Identity(ncclComm*, ncclKernelPlan*, uint64_t);
void mycroftM2Attach(ncclComm*, ncclChannel*, int, int, ncclConnector*, ncclProxyOp*);
void mycroftM2Ready(ncclProxyArgs*, ncclProxySubArgs*);
void mycroftM2SendSample(ncclProxyArgs*, ncclProxySubArgs*, bool);
void mycroftM2RecvSample(ncclProxyArgs*, ncclProxySubArgs*, bool);
void mycroftM2ProxyStopped(ncclProxyState*);
// Start after comm/stream initialization; Complete after local stream synchronization.
// Finish only after ncclCommDestroy/Abort has stopped the Proxy producers.
extern "C" __attribute__((visibility("default"))) int mycroftM2Start(void*, void*, uint64_t, uint64_t);
extern "C" __attribute__((visibility("default"))) int mycroftM2Complete(void*);
extern "C" __attribute__((visibility("default"))) int mycroftM2Finish(const char*, const char*);

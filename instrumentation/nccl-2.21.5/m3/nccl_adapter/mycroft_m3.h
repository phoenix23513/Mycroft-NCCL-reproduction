#pragma once
#include <stdint.h>
struct ncclProxyArgs;
struct ncclProxySubArgs;
bool mycroftM3AllowSend(ncclProxyArgs*, ncclProxySubArgs*);
// Configure once before M2Start; export only after successful M2Finish (producer join).
extern "C" __attribute__((visibility("default"))) int mycroftM3Configure(int, int, uint64_t, int, uint64_t);
extern "C" __attribute__((visibility("default"))) int mycroftM3Finish(const char*);

#ifndef MYCROFT_E01_TRACE_EVENT_H
#define MYCROFT_E01_TRACE_EVENT_H

#include <stdint.h>

typedef enum {
    TRACE_PHASE_REDUCE_SCATTER = 0,
    TRACE_PHASE_ALL_GATHER = 1,
} TracePhase;

typedef enum {
    TRACE_ACTION_SEND = 0,
    TRACE_ACTION_RECV = 1,
} TraceAction;

typedef enum {
    TRACE_DELAY_NONE = 0,
    TRACE_DELAY_INJECTED_ROOT = 1,
    TRACE_DELAY_AFFECTED = 2,
} TraceDelayRole;

/* One deterministic simulator event. */
typedef struct {
    uint64_t op_seq;
    int rank;
    int channel;
    TracePhase phase;
    int step;
    int chunk_id;
    TraceAction action;
    int peer;
    uint64_t baseline_timestamp;
    uint64_t timestamp;
    TraceDelayRole delay_role;
    int value;
    uint32_t contributor_mask;
} TraceEvent;

#endif

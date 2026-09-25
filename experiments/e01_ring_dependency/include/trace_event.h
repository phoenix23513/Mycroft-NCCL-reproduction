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
    uint64_t timestamp;
    int value;
    uint32_t contributor_mask;
} TraceEvent;

#endif

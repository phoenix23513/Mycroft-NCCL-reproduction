#ifndef MYCROFT_E01_TRACE_EVENT_H
#define MYCROFT_E01_TRACE_EVENT_H

#include <stdint.h>

/* One post-reduction receive record. JSONL serialization starts on Day 06. */
typedef struct {
    int step;
    int src_rank;
    int dst_rank;
    int chunk_id;
    int value;
    uint32_t contributor_mask;
} TraceEvent;

#endif

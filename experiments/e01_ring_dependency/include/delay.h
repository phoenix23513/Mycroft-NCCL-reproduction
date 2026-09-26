#ifndef MYCROFT_E01_DELAY_H
#define MYCROFT_E01_DELAY_H

#include <stddef.h>
#include <stdint.h>

#include "ring_sim.h"

typedef struct {
    TracePhase phase;
    int step;
    int rank;
    int chunk_id;
    TraceAction action;
    uint64_t delay_ticks;
} DelayConfig;

typedef struct {
    int root_event_index;
    int affected_event_count;
    uint32_t affected_rank_mask;
} DelayReport;

/*
 * Return the matching send event for one receive event, or -1 when the
 * current event has no message predecessor.
 */
int delay_find_message_predecessor(
    const TraceEvent *events,
    size_t event_count,
    size_t event_index
);

/* Return the previous event on the same rank, or -1 when none exists. */
int delay_find_rank_predecessor(
    const TraceEvent *events,
    size_t event_count,
    size_t event_index
);

/*
 * Inject one delay into the selected root event and propagate it only through
 * causal predecessors. Chunk values and contributor masks must not change.
 */
RingStatus ring_apply_causal_delay(
    RingSim *sim,
    const DelayConfig *config,
    DelayReport *report
);

#endif

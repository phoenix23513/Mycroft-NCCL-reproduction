#ifndef MYCROFT_E01_RING_SIM_H
#define MYCROFT_E01_RING_SIM_H

#include <stdint.h>

#include "trace_event.h"

enum {
    RING_RANKS = 4,
    RING_CHUNKS = 8,
    RING_CHUNKS_PER_RANK = RING_CHUNKS / RING_RANKS,
    RING_STEPS = RING_RANKS - 1,
    RING_PHASES = 2,
    RING_ACTIONS_PER_MESSAGE = 2,
    RING_MAX_TRACE_EVENTS = RING_PHASES * RING_STEPS * RING_RANKS *
                            RING_CHUNKS_PER_RANK * RING_ACTIONS_PER_MESSAGE,
};

typedef enum {
    RING_OK = 0,
    RING_ERROR_INVALID_ARGUMENT = -1,
    RING_ERROR_NOT_IMPLEMENTED = -2,
} RingStatus;

typedef struct {
    int value;
    uint32_t contributor_mask;
} ChunkState;

typedef struct {
    int src_rank;
    int dst_rank;
    int chunk_id;
    ChunkState payload;
} RingMessage;

typedef struct {
    ChunkState chunks[RING_RANKS][RING_CHUNKS];
    TraceEvent history[RING_STEPS][RING_RANKS][RING_CHUNKS_PER_RANK];
    TraceEvent events[RING_MAX_TRACE_EVENTS];
    int event_count;
    uint64_t next_timestamp;
    int completed_steps;
    int completed_all_gather_steps;
} RingSim;

int ring_next_rank(int rank);
int ring_prev_rank(int rank);
int ring_send_chunk(int rank, int step, int lane);
int ring_recv_chunk(int rank, int step, int lane);
int ring_all_gather_send_chunk(int rank, int step, int lane);
int ring_all_gather_recv_chunk(int rank, int step, int lane);

RingStatus ring_sim_init(RingSim *sim);
RingStatus ring_reduce_scatter_step(RingSim *sim, int step);
RingStatus ring_run_reduce_scatter(RingSim *sim);
RingStatus ring_all_gather_step(RingSim *sim, int step);
RingStatus ring_run_all_gather(RingSim *sim);
RingStatus ring_run_all_reduce(RingSim *sim);

#endif

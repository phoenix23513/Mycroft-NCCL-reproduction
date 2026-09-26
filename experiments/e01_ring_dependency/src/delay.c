#include "delay.h"

#include <limits.h>
#include <stdbool.h>

int delay_find_message_predecessor(
    const TraceEvent *events,
    size_t event_count,
    size_t event_index
) {
    if (events == NULL || event_index >= event_count) {
        return -1;
    }

    const TraceEvent *current = &events[event_index];
    if (current->action != TRACE_ACTION_RECV) {
        return -1;
    }

    for (size_t cursor = event_index; cursor > 0; --cursor) {
        const size_t candidate_index = cursor - 1;
        const TraceEvent *candidate = &events[candidate_index];

        if (candidate->action == TRACE_ACTION_SEND &&
            candidate->op_seq == current->op_seq &&
            candidate->channel == current->channel &&
            candidate->phase == current->phase &&
            candidate->step == current->step &&
            candidate->chunk_id == current->chunk_id &&
            candidate->rank == current->peer &&
            candidate->peer == current->rank) {
            if (candidate_index > (size_t)INT_MAX) {
                return -1;
            }
            return (int)candidate_index;
        }
    }

    return -1;
}

int delay_find_rank_predecessor(
    const TraceEvent *events,
    size_t event_count,
    size_t event_index
) {
    if (events == NULL || event_index >= event_count) {
        return -1;
    }

    const int current_rank = events[event_index].rank;

    for (size_t cursor = event_index; cursor > 0; --cursor) {
        const size_t candidate_index = cursor - 1;

        if (events[candidate_index].rank == current_rank) {
            if (candidate_index > (size_t)INT_MAX) {
                return -1;
            }
            return (int)candidate_index;
        }
    }

    return -1;
}

RingStatus ring_apply_causal_delay(
    RingSim *sim,
    const DelayConfig *config,
    DelayReport *report
) {
    if (sim == NULL || config == NULL || report == NULL ||
        sim->event_count <= 0 || config->delay_ticks == 0 ||
        sim->event_count > RING_MAX_TRACE_EVENTS) {
        return RING_ERROR_INVALID_ARGUMENT;
    }

    report->root_event_index = -1;
    report->affected_event_count = 0;
    report->affected_rank_mask = UINT32_C(0);

    for (int index = 0; index < sim->event_count; ++index) {
        TraceEvent *event = &sim->events[index];
        event->timestamp = event->baseline_timestamp;
        event->delay_role = TRACE_DELAY_NONE;

        if (event->phase == config->phase && event->step == config->step &&
            event->rank == config->rank &&
            event->chunk_id == config->chunk_id &&
            event->action == config->action) {
            if (report->root_event_index != -1) {
                return RING_ERROR_INVALID_ARGUMENT;
            }
            report->root_event_index = index;
        }
    }

    if (report->root_event_index == -1) {
        return RING_ERROR_EVENT_NOT_FOUND;
    }

    for (int index = 0; index < sim->event_count; ++index) {
        TraceEvent *event = &sim->events[index];
        uint64_t inherited_delay = UINT64_C(0);

        const int message_predecessor = delay_find_message_predecessor(
            sim->events,
            (size_t)sim->event_count,
            (size_t)index
        );
        if (message_predecessor >= 0) {
            const TraceEvent *predecessor =
                &sim->events[message_predecessor];
            if (predecessor->timestamp < predecessor->baseline_timestamp) {
                return RING_ERROR_INVALID_ARGUMENT;
            }

            inherited_delay =
                predecessor->timestamp - predecessor->baseline_timestamp;
        }

        const int rank_predecessor = delay_find_rank_predecessor(
            sim->events,
            (size_t)sim->event_count,
            (size_t)index
        );
        if (rank_predecessor >= 0) {
            const TraceEvent *predecessor = &sim->events[rank_predecessor];
            if (predecessor->timestamp < predecessor->baseline_timestamp) {
                return RING_ERROR_INVALID_ARGUMENT;
            }

            const uint64_t rank_delay =
                predecessor->timestamp - predecessor->baseline_timestamp;
            if (rank_delay > inherited_delay) {
                inherited_delay = rank_delay;
            }
        }

        uint64_t event_delay = inherited_delay;
        const bool is_root = index == report->root_event_index;
        if (is_root) {
            if (UINT64_MAX - event_delay < config->delay_ticks) {
                return RING_ERROR_INVALID_ARGUMENT;
            }
            event_delay += config->delay_ticks;
        }

        if (UINT64_MAX - event->baseline_timestamp < event_delay) {
            return RING_ERROR_INVALID_ARGUMENT;
        }
        event->timestamp = event->baseline_timestamp + event_delay;

        if (is_root) {
            event->delay_role = TRACE_DELAY_INJECTED_ROOT;
            continue;
        }

        if (event_delay == 0) {
            event->delay_role = TRACE_DELAY_NONE;
            continue;
        }

        if (event->rank < 0 || event->rank >= 32) {
            return RING_ERROR_INVALID_ARGUMENT;
        }
        event->delay_role = TRACE_DELAY_AFFECTED;
        report->affected_event_count += 1;
        report->affected_rank_mask |= UINT32_C(1) << event->rank;
    }

    return RING_OK;
}

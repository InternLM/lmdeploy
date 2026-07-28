// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <memory>

namespace turbomind {
class TurboMind;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tm_engine_handle tm_engine_handle;
typedef struct tm_request_handle tm_request_handle;

typedef void (*tm_notify_fn)(void* context);

enum tm_result_code {
    TM_RESULT_OK               = 0,
    TM_RESULT_INVALID_ARGUMENT = 1,
    TM_RESULT_INVALID_STATE    = 2,
    TM_RESULT_NOT_FOUND        = 3,
    TM_RESULT_BUFFER_TOO_SMALL = 4,
    TM_RESULT_INTERNAL_ERROR   = 255,
};

typedef struct tm_error {
    int32_t code;
    char    message[512];
} tm_error;

typedef struct tm_int_slice {
    const int32_t* data;
    size_t         len;
} tm_int_slice;

typedef struct tm_generation_config {
    int32_t  max_new_tokens;
    int32_t  min_new_tokens;
    int32_t  top_k;
    float    top_p;
    float    min_p;
    float    temperature;
    float    repetition_penalty;
    uint64_t random_seed;
    int32_t  output_logprobs;
    uint8_t  return_ppl;
    uint8_t  reserved[7];
    tm_int_slice eos_ids;
    tm_int_slice stop_ids;
    tm_int_slice bad_ids;
} tm_generation_config;

typedef struct tm_submit_params {
    tm_int_slice         input_ids;
    uint64_t             session_id;
    int32_t              session_step;
    uint8_t              stream_output;
    uint8_t              enable_metrics;
    uint8_t              reserved[6];
    tm_generation_config generation;
} tm_submit_params;

typedef struct tm_request_state {
    uint8_t available;
    uint8_t reserved[3];
    int32_t status;
    int32_t sequence_length;
} tm_request_state;

typedef struct tm_logprob_entry {
    int32_t token_id;
    float   logprob;
} tm_logprob_entry;

typedef struct tm_request_metrics {
    int64_t enqueue_time_us;
    int64_t scheduled_time_us;
} tm_request_metrics;

typedef struct tm_schedule_metrics {
    int32_t total_sequences;
    int32_t active_sequences;
    int32_t waiting_sequences;
    int32_t total_blocks;
    int32_t active_blocks;
    int32_t cached_blocks;
    int32_t free_blocks;
    int64_t scheduler_tick;
} tm_schedule_metrics;

uint32_t tm_api_version(void);
void tm_error_clear(tm_error* error);

void tm_engine_retain(tm_engine_handle* engine);
void tm_engine_release(tm_engine_handle* engine);

int32_t tm_engine_create_request(tm_engine_handle* engine, tm_request_handle** request, tm_error* error);
int32_t tm_engine_get_schedule_metrics(tm_engine_handle*   engine,
                                       int32_t             device_index,
                                       tm_schedule_metrics* metrics,
                                       uint8_t*            available,
                                       tm_error*           error);

int32_t tm_request_submit(tm_request_handle*     request,
                          const tm_submit_params* params,
                          tm_notify_fn            notify,
                          void*                   notify_context,
                          tm_error*               error);
int32_t tm_request_consume_state(tm_request_handle* request, tm_request_state* state, tm_error* error);
int32_t tm_request_copy_output_ids(tm_request_handle* request,
                                   int32_t            begin,
                                   int32_t            end,
                                   int32_t*            output,
                                   size_t              output_capacity,
                                   tm_error*           error);
int32_t tm_request_get_logprob_count(tm_request_handle* request,
                                     int32_t            generated_index,
                                     int32_t*            count,
                                     tm_error*           error);
int32_t tm_request_copy_logprobs(tm_request_handle* request,
                                 int32_t            generated_index,
                                 tm_logprob_entry*  output,
                                 size_t             output_capacity,
                                 size_t*            written,
                                 tm_error*           error);
int32_t tm_request_get_ce_loss(tm_request_handle* request, float* ce_loss, tm_error* error);
int32_t tm_request_get_metrics(tm_request_handle* request,
                               tm_request_metrics* metrics,
                               uint8_t*            available,
                               tm_error*           error);
void tm_request_cancel(tm_request_handle* request);
void tm_request_release(tm_request_handle* request);

#ifdef __cplusplus
}  // extern "C"

namespace turbomind::rust {

// Retain a pybind-owned engine without exposing std::shared_ptr through C ABI.
tm_engine_handle* RetainEngine(std::shared_ptr<TurboMind> engine);

}  // namespace turbomind::rust
#endif

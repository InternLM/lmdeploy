// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/rust/c_api.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/model_request.h"
#include "src/turbomind/turbomind.h"

namespace {

constexpr uint32_t kApiVersion = 1;

class ApiError: public std::runtime_error {
public:
    ApiError(int32_t code, std::string message): std::runtime_error{std::move(message)}, code_{code} {}

    int32_t code() const noexcept
    {
        return code_;
    }

private:
    int32_t code_;
};

struct CallbackState {
    std::mutex   mutex;
    tm_notify_fn notify{};
    void*        context{};
    bool         armed{};

    void Invoke()
    {
        std::lock_guard<std::mutex> lock{mutex};
        if (armed && notify) {
            notify(context);
        }
    }

    void Disarm()
    {
        std::lock_guard<std::mutex> lock{mutex};
        armed   = false;
        notify  = nullptr;
        context = nullptr;
    }
};

void SetError(tm_error* error, int32_t code, const std::string& message)
{
    if (!error) {
        return;
    }
    error->code = code;
    std::strncpy(error->message, message.c_str(), sizeof(error->message) - 1);
    error->message[sizeof(error->message) - 1] = '\0';
}

template<class F>
int32_t Guard(tm_error* error, F&& fn)
{
    tm_error_clear(error);
    try {
        fn();
        return TM_RESULT_OK;
    }
    catch (const ApiError& e) {
        SetError(error, e.code(), e.what());
        return e.code();
    }
    catch (const std::invalid_argument& e) {
        SetError(error, TM_RESULT_INVALID_ARGUMENT, e.what());
        return TM_RESULT_INVALID_ARGUMENT;
    }
    catch (const std::out_of_range& e) {
        SetError(error, TM_RESULT_NOT_FOUND, e.what());
        return TM_RESULT_NOT_FOUND;
    }
    catch (const std::logic_error& e) {
        SetError(error, TM_RESULT_INVALID_STATE, e.what());
        return TM_RESULT_INVALID_STATE;
    }
    catch (const std::exception& e) {
        SetError(error, TM_RESULT_INTERNAL_ERROR, e.what());
        return TM_RESULT_INTERNAL_ERROR;
    }
    catch (...) {
        SetError(error, TM_RESULT_INTERNAL_ERROR, "unknown TurboMind C API error");
        return TM_RESULT_INTERNAL_ERROR;
    }
}

void ValidateSlice(tm_int_slice slice, const char* name)
{
    if (slice.len && !slice.data) {
        throw std::invalid_argument(std::string{name} + " data is null");
    }
}

std::vector<int> CopyInts(tm_int_slice slice, const char* name)
{
    static_assert(sizeof(int) == sizeof(int32_t));
    ValidateSlice(slice, name);
    if (!slice.len) {
        return {};
    }
    return {slice.data, slice.data + slice.len};
}

void SetStopBadWords(std::array<std::vector<int>, 2>& output, tm_int_slice input, const char* name)
{
    output[0] = CopyInts(input, name);
    output[1].resize(output[0].size());
    std::iota(output[1].begin(), output[1].end(), 1);
}

turbomind::GenerationConfig ConvertGenerationConfig(const tm_generation_config& input)
{
    turbomind::GenerationConfig output{};
    output.max_new_tokens     = input.max_new_tokens;
    output.min_new_tokens     = input.min_new_tokens;
    output.top_k              = input.top_k;
    output.top_p              = input.top_p;
    output.min_p              = input.min_p;
    output.temperature        = input.temperature;
    output.repetition_penalty = input.repetition_penalty;
    output.random_seed        = input.random_seed;
    output.output_logprobs    = input.output_logprobs;
    output.return_ppl         = input.return_ppl != 0;
    output.eos_ids            = CopyInts(input.eos_ids, "eos_ids");
    SetStopBadWords(output.stop_ids, input.stop_ids, "stop_ids");
    SetStopBadWords(output.bad_ids, input.bad_ids, "bad_ids");
    return output;
}

turbomind::Tensor MakeInputIds(tm_int_slice input)
{
    ValidateSlice(input, "input_ids");
    if (!input.len) {
        throw std::invalid_argument("input_ids must not be empty");
    }
    turbomind::Tensor tensor{turbomind::Layout{static_cast<turbomind::ssize_t>(input.len)},
                             turbomind::data_type_v<int>,
                             turbomind::kCPU};
    std::memcpy(tensor.data<int>(), input.data, input.len * sizeof(int32_t));
    return tensor;
}

}  // namespace

struct tm_engine_handle {
    std::atomic<size_t>                references{1};
    std::shared_ptr<turbomind::TurboMind> engine;
};

struct tm_request_handle {
    std::shared_ptr<turbomind::TurboMind>    engine;
    std::unique_ptr<turbomind::ModelRequest> request;
    turbomind::ModelRequest::OutputParam     output;
    std::shared_ptr<CallbackState>            callback;
    bool                                      submitted{};
};

extern "C" uint32_t tm_api_version(void)
{
    return kApiVersion;
}

extern "C" void tm_error_clear(tm_error* error)
{
    if (error) {
        error->code       = TM_RESULT_OK;
        error->message[0] = '\0';
    }
}

extern "C" void tm_engine_retain(tm_engine_handle* engine)
{
    if (engine) {
        engine->references.fetch_add(1, std::memory_order_relaxed);
    }
}

extern "C" void tm_engine_release(tm_engine_handle* engine)
{
    if (engine && engine->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete engine;
    }
}

extern "C" int32_t tm_engine_create_request(tm_engine_handle*   engine,
                                              tm_request_handle** request,
                                              tm_error*           error)
{
    return Guard(error, [&] {
        if (!engine || !engine->engine) {
            throw std::invalid_argument("engine handle is null");
        }
        if (!request) {
            throw std::invalid_argument("request output is null");
        }
        auto output      = std::make_unique<tm_request_handle>();
        output->engine   = engine->engine;
        output->request  = output->engine->CreateRequest();
        output->callback = std::make_shared<CallbackState>();
        *request         = output.release();
    });
}

extern "C" int32_t tm_engine_get_schedule_metrics(tm_engine_handle*    engine,
                                                    int32_t              device_index,
                                                    tm_schedule_metrics* metrics,
                                                    uint8_t*             available,
                                                    tm_error*            error)
{
    return Guard(error, [&] {
        if (!engine || !engine->engine || !metrics || !available) {
            throw std::invalid_argument("invalid schedule metrics arguments");
        }
        auto source = engine->engine->GetScheduleMetrics(device_index);
        *available  = source ? 1 : 0;
        if (!source) {
            *metrics = {};
            return;
        }
        metrics->total_sequences   = source->total_seqs;
        metrics->active_sequences  = source->active_seqs;
        metrics->waiting_sequences = source->waiting_seqs;
        metrics->total_blocks      = source->total_blocks;
        metrics->active_blocks     = source->active_blocks;
        metrics->cached_blocks     = source->cached_blocks;
        metrics->free_blocks       = source->free_blocks;
        metrics->scheduler_tick    = source->scheduler_tick;
    });
}

extern "C" int32_t tm_request_submit(tm_request_handle*      request,
                                       const tm_submit_params* params,
                                       tm_notify_fn             notify,
                                       void*                    notify_context,
                                       tm_error*                error)
{
    return Guard(error, [&] {
        if (!request || !request->request || !params) {
            throw std::invalid_argument("invalid submit arguments");
        }
        if (request->submitted) {
            throw std::logic_error("request has already been submitted");
        }

        auto tensors = std::make_shared<turbomind::TensorMap>();
        tensors->emplace("input_ids", MakeInputIds(params->input_ids));

        turbomind::ModelRequest::InputParam input{};
        input.tensors        = std::move(tensors);
        input.session        = {params->session_id, params->session_step};
        input.gen_cfg        = ConvertGenerationConfig(params->generation);
        input.stream_output  = params->stream_output != 0;
        input.enable_metrics = params->enable_metrics != 0;

        {
            std::lock_guard<std::mutex> lock{request->callback->mutex};
            request->callback->notify  = notify;
            request->callback->context = notify_context;
            request->callback->armed   = true;
        }
        auto callback = request->callback;
        try {
            request->output = request->request->Forward(std::move(input), [callback = std::move(callback)] {
                callback->Invoke();
            });
        }
        catch (...) {
            request->callback->Disarm();
            throw;
        }
        request->submitted = true;
    });
}

extern "C" int32_t tm_request_consume_state(tm_request_handle* request,
                                              tm_request_state*  state,
                                              tm_error*          error)
{
    return Guard(error, [&] {
        if (!request || !state) {
            throw std::invalid_argument("invalid consume state arguments");
        }
        if (!request->submitted || !request->output.state) {
            throw std::logic_error("request has not been submitted");
        }
        auto source = request->output.state->exchange(nullptr);
        *state      = {};
        if (source) {
            state->available       = 1;
            state->status          = source->status;
            state->sequence_length = source->seq_len;
        }
    });
}

extern "C" int32_t tm_request_copy_output_ids(tm_request_handle* request,
                                                int32_t            begin,
                                                int32_t            end,
                                                int32_t*           output,
                                                size_t             output_capacity,
                                                tm_error*          error)
{
    return Guard(error, [&] {
        if (!request || !request->submitted || !request->output.tensors) {
            throw std::logic_error("request output is not available");
        }
        if (begin < 0 || end < begin) {
            throw std::invalid_argument("invalid output token range");
        }
        const auto count = static_cast<size_t>(end - begin);
        if (count > output_capacity) {
            throw ApiError{TM_RESULT_BUFFER_TOO_SMALL, "output token buffer is too small"};
        }
        if (count && !output) {
            throw std::invalid_argument("output token buffer is null");
        }
        const auto& tensor = request->output.tensors->at("output_ids");
        if (end > tensor.size()) {
            throw std::out_of_range("output token range exceeds the native buffer");
        }
        std::memcpy(output, tensor.data<int>() + begin, count * sizeof(int32_t));
    });
}

extern "C" int32_t tm_request_get_logprob_count(tm_request_handle* request,
                                                  int32_t            generated_index,
                                                  int32_t*           count,
                                                  tm_error*          error)
{
    return Guard(error, [&] {
        if (!request || !count || !request->submitted || !request->output.tensors) {
            throw std::invalid_argument("invalid logprob count arguments");
        }
        const auto& tensor = request->output.tensors->at("logprob_nums");
        if (generated_index < 0 || generated_index >= tensor.size()) {
            throw std::out_of_range("generated logprob index is out of range");
        }
        *count = tensor.data<int>()[generated_index];
    });
}

extern "C" int32_t tm_request_copy_logprobs(tm_request_handle* request,
                                              int32_t            generated_index,
                                              tm_logprob_entry*  output,
                                              size_t             output_capacity,
                                              size_t*            written,
                                              tm_error*          error)
{
    return Guard(error, [&] {
        if (!request || !written || !request->submitted || !request->output.tensors) {
            throw std::invalid_argument("invalid logprob arguments");
        }
        const auto& nums    = request->output.tensors->at("logprob_nums");
        const auto& indexes = request->output.tensors->at("logprob_indexes");
        const auto& values  = request->output.tensors->at("logprob_vals");
        if (generated_index < 0 || generated_index >= nums.size()) {
            throw std::out_of_range("generated logprob index is out of range");
        }
        const auto count = static_cast<size_t>(std::max(0, nums.data<int>()[generated_index]));
        if (count > output_capacity) {
            throw ApiError{TM_RESULT_BUFFER_TOO_SMALL, "logprob output buffer is too small"};
        }
        if (count && !output) {
            throw std::invalid_argument("logprob output buffer is null");
        }
        const auto width = static_cast<size_t>(indexes.shape(1));
        if (count > width) {
            throw std::out_of_range("native logprob count exceeds the tensor width");
        }
        const auto base  = static_cast<size_t>(generated_index) * width;
        for (size_t i = 0; i < count; ++i) {
            output[i] = {indexes.data<int>()[base + i], values.data<float>()[base + i]};
        }
        *written = count;
    });
}

extern "C" int32_t tm_request_get_ce_loss(tm_request_handle* request, float* ce_loss, tm_error* error)
{
    return Guard(error, [&] {
        if (!request || !ce_loss || !request->submitted || !request->output.tensors) {
            throw std::invalid_argument("invalid ce_loss arguments");
        }
        const auto& tensor = request->output.tensors->at("ce_loss");
        *ce_loss           = tensor.data<float>()[0];
    });
}

extern "C" int32_t tm_request_get_metrics(tm_request_handle* request,
                                            tm_request_metrics* metrics,
                                            uint8_t*            available,
                                            tm_error*           error)
{
    return Guard(error, [&] {
        if (!request || !metrics || !available) {
            throw std::invalid_argument("invalid request metrics arguments");
        }
        *available = request->output.metrics ? 1 : 0;
        if (!request->output.metrics) {
            *metrics = {};
            return;
        }
        metrics->enqueue_time_us = request->output.metrics->enqueue_time.load(std::memory_order_relaxed);
        metrics->scheduled_time_us = request->output.metrics->scheduled_time.load(std::memory_order_relaxed);
    });
}

extern "C" void tm_request_cancel(tm_request_handle* request)
{
    if (request && request->request) {
        try {
            request->request->Cancel();
        }
        catch (...) {
        }
    }
}

extern "C" void tm_request_release(tm_request_handle* request)
{
    if (!request) {
        return;
    }
    request->callback->Disarm();
    if (request->submitted) {
        tm_request_cancel(request);
    }
    delete request;
}

namespace turbomind::rust {

tm_engine_handle* RetainEngine(std::shared_ptr<TurboMind> engine)
{
    if (!engine) {
        return nullptr;
    }
    auto* handle   = new tm_engine_handle;
    handle->engine = std::move(engine);
    return handle;
}

}  // namespace turbomind::rust

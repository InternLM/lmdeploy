

#include <algorithm>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "src/turbomind/engine/model_request.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/utils/constant.h"
#include "src/turbomind/utils/metrics.h"

namespace turbomind {

ModelRequest::ModelRequest(Gateway* gateway, DataType data_type, int session_len, int vocab_size, int hidden_dim):
    gateway_{gateway},
    data_type_{data_type},
    session_len_{session_len},
    vocab_size_{vocab_size},
    hidden_dim_{hidden_dim}
{
}

void ModelRequest::Cancel()
{
    // request is finished if lock failed
    if (auto r = request_.lock()) {
        gateway_->cancel(std::move(r));
    }
}

void ModelRequest::End(std::function<void(int)> cb, uint64_t session_id)
{
    auto r = std::make_shared<Request>();

    r->id = r->session.id = session_id;
    r->session.kill_flag  = true;

    r->end_cb = std::move(cb);

    gateway_->kill(std::move(r));
}

auto ModelRequest::Forward(InputParam param, std::function<void()> cb) -> OutputParam
{
    inputs_  = std::make_shared<TensorMap>();
    outputs_ = std::make_shared<TensorMap>();

    auto add = [](auto& dest, auto key, auto dtype, auto where, auto shape, auto&&... dims) {
        Layout shape_;
        if constexpr (std::is_integral_v<decltype(shape)>) {
            shape_ = {shape, dims...};
        }
        else {
            shape_ = {shape.cbegin(), shape.cend()};
        }
        dest->emplace(key, Tensor{shape_, dtype, where});
    };

    auto& inputs = *param.tensors;

    TM_CHECK_EQ(inputs.at("input_ids").ndim(), 1);

    const int input_len  = inputs.at("input_ids").shape(0);
    const int output_len = param.gen_cfg.max_new_tokens;

    // Max possible length of a sequence, this depends on `history_len` which isn't available here, so `session_len`
    // is used instead
    const int max_seq_len = session_len_ + 1;
    const int max_out_len = std::min(output_len, session_len_) + 1;
    // This does not include histroy length in interactive mode
    const int max_in_out_len = std::min(input_len + output_len, session_len_) + 1;

    for (auto& [k, v] : *param.tensors) {
        inputs_->emplace(k, v);
    }

    add(outputs_, "output_ids", data_type_v<int>, kCPU, max_seq_len);
    add(outputs_, "sequence_length", data_type_v<int>, kCPU, 1);

    if (param.gen_cfg.output_logits) {
        const int len = param.gen_cfg.output_logits == GenerationConfig::kAll ? max_in_out_len : max_out_len;
        add(outputs_, "logits", data_type_, kCPU, len, vocab_size_);
    }

    if (param.gen_cfg.output_last_hidden_state) {
        const int len = param.gen_cfg.output_last_hidden_state == GenerationConfig::kAll ? max_in_out_len : max_out_len;
        add(outputs_, "last_hidden_state", data_type_, kCPU, len, hidden_dim_);
    }

    if (param.gen_cfg.output_logprobs) {
        add(outputs_, "logprob_vals", data_type_v<float>, kCPU, max_out_len, kMaxLogProb);
        add(outputs_, "logprob_indexes", data_type_v<int>, kCPU, max_out_len, kMaxLogProb);
        add(outputs_, "logprob_nums", data_type_v<int>, kCPU, max_out_len);
    }

    auto r = std::make_shared<Request>();

    for (const auto& [k, v] : *inputs_) {
        r->inputs.emplace(k, v);
    }
    for (const auto& [k, v] : *outputs_) {
        r->outputs.emplace(k, v);
    }

    auto state = std::make_shared<AtomicRequestState>();

    auto metrics = param.enable_metrics ? std::make_shared<RequestMetrics>() : nullptr;
    if (metrics) {
        metrics->enque_time     = RequestMetrics::timestamp();
        metrics->scheduled_time = 0;  // will be set later
    }

    if (param.session.start_flag) {
        session_id_ = param.session.id;
    }

    r->id            = param.session.id;
    r->session       = param.session;
    r->gen_cfg       = param.gen_cfg;
    r->stream_output = param.stream_output;
    r->forward_cb    = std::move(cb);
    r->state         = state;
    r->metrics       = metrics;

    r->output_ids      = outputs_->at("output_ids");
    r->sequence_length = outputs_->at("sequence_length");

    if (grammar_) {
        r->matcher = std::make_shared<xgrammar::GrammarMatcher>(*grammar_);
    }

    // Keep a weak reference for canceling the request
    request_ = r;

    gateway_->push({std::move(r)});

    return OutputParam{outputs_, state, metrics};
}

void ModelRequest::setGrammar(const xgrammar::CompiledGrammar& grammar)
{
    grammar_ = std::make_shared<xgrammar::CompiledGrammar>(grammar);
}

}  // namespace turbomind

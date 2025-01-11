

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/turbomind/engine/model_request.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/constant.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

static ManagedTensor create(DataType dtype, MemoryType where, const std::vector<int64_t>& size, int64_t& byte_size)
{
    byte_size = std::accumulate(size.begin(), size.end(), Tensor::getTypeSize(dtype), std::multiplies<>{});
    void* data{};

    if (where == MEMORY_GPU) {
        check_cuda_error(cudaMallocAsync(&data, byte_size, nullptr));
    }
    else {
        data = std::malloc(byte_size);
    }

    ManagedTensor ret;
    ret.tensor = Tensor{where, dtype, std::vector<size_t>(size.begin(), size.end()), data};
    ret.data_holder.reset((void*)nullptr, [data, where](auto) {
        // std::cerr << "turbomind tensor deallocate" << std::endl;
        if (where == MEMORY_GPU) {
            /// TODO: guard device id
            check_cuda_error(cudaFreeAsync(data, nullptr));
        }
        else {
            std::free(data);
        }
    });
    return ret;
}

template<class T>
static T get(const std::unordered_map<std::string, ManagedTensor>& m, const std::string& key, T fallback = {})
{
    auto it = m.find(key);
    if (it != m.end()) {
        return it->second->getVal<T>();
    }
    return fallback;
}

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
    inputs_  = std::make_shared<TensorMap_>();
    outputs_ = std::make_shared<TensorMap_>();

    auto add = [](auto& dest, auto key, auto dtype, auto where, auto shape, auto&&... dims) {
        std::vector<int64_t> shape_;
        if constexpr (std::is_integral_v<decltype(shape)>) {
            shape_ = {shape, dims...};
        }
        else {
            shape_ = {shape.cbegin(), shape.cend()};
        }
        int64_t byte_size{};
        auto    it = dest->emplace(key, create(dtype, where, shape_, byte_size)).first;
        return std::make_pair(it->second->data, byte_size);
    };

    auto& inputs = *param.tensors;

    FT_CHECK(inputs.at("input_ids")->shape.size() == 1);

    const int input_len  = inputs.at("input_ids")->shape[0];
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

    add(outputs_, "output_ids", TYPE_INT32, MEMORY_CPU, max_seq_len);
    add(outputs_, "sequence_length", TYPE_INT32, MEMORY_CPU, 1);

    if (param.gen_cfg.output_logits) {
        const int len = param.gen_cfg.output_logits == GenerationConfig::kAll ? max_in_out_len : max_out_len;
        add(outputs_, "logits", TYPE_FP32, MEMORY_CPU, len, vocab_size_);
    }

    if (param.gen_cfg.output_last_hidden_state) {
        const int len = param.gen_cfg.output_last_hidden_state == GenerationConfig::kAll ? max_in_out_len : max_out_len;
        add(outputs_, "last_hidden_state", data_type_, MEMORY_CPU, len, hidden_dim_);
    }

    if (param.gen_cfg.output_logprobs) {
        add(outputs_, "logprob_vals", TYPE_FP32, MEMORY_CPU, max_out_len, kMaxLogProb);
        add(outputs_, "logprob_indexes", TYPE_INT32, MEMORY_CPU, max_out_len, kMaxLogProb);
        add(outputs_, "logprob_nums", TYPE_INT32, MEMORY_CPU, max_out_len);
    }

    auto r = std::make_shared<Request>();

    for (const auto& [k, v] : *inputs_) {
        r->inputs.insert(k, *v);
    }
    for (const auto& [k, v] : *outputs_) {
        r->outputs.insert(k, *v);
    }

    auto state = std::make_shared<AtomicRequestState>();

    if (param.session.start_flag) {
        session_id_ = param.session.id;
    }

    r->id            = param.session.id;
    r->session       = param.session;
    r->gen_cfg       = param.gen_cfg;
    r->stream_output = param.stream_output;
    r->forward_cb    = std::move(cb);
    r->state         = state;

    r->output_ids      = *outputs_->at("output_ids");
    r->sequence_length = *outputs_->at("sequence_length");

    // Keep a weak reference for canceling the request
    request_ = r;

    gateway_->push({std::move(r)});

    return OutputParam{outputs_, state};
}

}  // namespace turbomind

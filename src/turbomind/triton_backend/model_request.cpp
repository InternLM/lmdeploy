

#include <algorithm>
#include <atomic>
#include <functional>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/turbomind/models/llama/Request.h"
#include "src/turbomind/triton_backend/model_request.h"
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
    // FT_CHECK_WITH_INFO(byte_size == ret.tensor.sizeBytes(), fmtstr("%ld vs %ld", byte_size, ret.tensor.sizeBytes()));
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

ModelRequest::ModelRequest(RequestQueue* queue, std::atomic<float>* tok_per_tick, int session_len, int vocab_size):
    queue_{queue}, tok_per_tick_{tok_per_tick}, session_len_{session_len}, vocab_size_{vocab_size}
{
}

void ModelRequest::Cancel(bool end, std::function<void(int)> cb)
{
    auto r = std::make_shared<Request>();

    r->id = session_id_;
    // r->stop_flag = true;

    r->cancel_cb = std::move(cb);

    queue_->enqueue({std::move(r)});
}

void ModelRequest::End(std::function<void(int)> cb)
{
    auto r = std::make_shared<Request>();

    r->id = session_id_;
    // r->end_flag = true;

    r->end_cb = std::move(cb);

    queue_->enqueue({std::move(r)});
}

auto ModelRequest::Forward(InputParam param, std::function<void(RequestState)> cb) -> OutputParam
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

    const int batch_size = 1;
    const int beam_width = 1;

    FT_CHECK(inputs.at("input_ids")->shape.size() == 1);

    const int input_len  = inputs.at("input_ids")->shape[0];
    const int output_len = input_len + param.gen_cfg.max_new_tokens;

    for (auto& [k, v] : *param.tensors) {
        inputs_->emplace(k, v);
    }

    add(outputs_, "output_ids", TYPE_INT32, MEMORY_CPU, session_len_);
    add(outputs_, "sequence_length", TYPE_INT32, MEMORY_CPU, 1);

    if (param.gen_cfg.output_logprobs) {
        const int max_logprob_len = std::min(output_len, session_len_) + 1;
        add(outputs_, "logprob_vals", TYPE_FP32, MEMORY_CPU, max_logprob_len, kMaxLogProb);
        add(outputs_, "logprob_indexes", TYPE_INT32, MEMORY_CPU, max_logprob_len, kMaxLogProb);
        add(outputs_, "logprob_nums", TYPE_INT32, MEMORY_CPU, max_logprob_len);
    }

    if (param.gen_cfg.output_logits) {
        /// TODO: allow output logits on GPU
        add(outputs_, "logits", TYPE_FP32, MEMORY_CPU, output_len, vocab_size_);
    }

    auto r = std::make_shared<Request>();

    for (const auto& [k, v] : *inputs_) {
        r->inputs.insert(k, *v);
    }
    for (const auto& [k, v] : *outputs_) {
        r->outputs.insert(k, *v);
    }

    r->id            = param.session.id;
    r->session       = param.session;
    r->gen_cfg       = param.gen_cfg;
    r->stream_output = param.stream_output;
    r->forward_cb    = std::move(cb);
    r->flag          = &flag_;

    // flag_.clear(std::memory_order_release);
    flag_.store(0);

    queue_->enqueue({std::move(r)});

    return OutputParam{outputs_};
}

void ModelRequest::ReportTokensPerTick(int observed)
{
    // flag_.clear(std::memory_order_release);

    flag_.fetch_sub(1, std::memory_order_relaxed);

#if 0
    constexpr float decay = 0.525;

    float value = (float)observed;
    // value -= std::max(0.f, std::min(decay, value - 1.f));

    float old = tok_per_tick_->load();
    float cur{};
    auto  update = [&]() mutable {
        float alpha = old > value ? 0.001 : 0.002;
        cur         = old * (1 - alpha) + value * alpha;
    };
    update();
    while (!tok_per_tick_->compare_exchange_weak(old, cur)) {
        update();
    }

    static int count = 0;
    if (++count % 100 == 0) {
        std::cerr << cur << std::endl;
    }
#endif
}

}  // namespace turbomind

/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.cc

#include <algorithm>
#include <math.h>
#include <numeric>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/mla_utils.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

namespace attention {

struct ForwardParam {

    explicit ForwardParam(int max_batch_size)
    {
        d_cu_x_len = {2 * (max_batch_size + 1), MEMORY_GPU};
        h_cu_x_len = {2 * (max_batch_size + 1), MEMORY_CPU_PINNED};
        event      = core::Event::create();
    }

    void Init(core::TensorMap& args, const core::Tensor& input_, core::Tensor& output_)
    {
        h_q_len = args.at("h_q_len").buffer();
        h_k_len = args.at("h_k_len").buffer();

        const int bsz = h_q_len.size();

        d_cu_q_len = d_cu_x_len.data();
        h_cu_q_len = h_cu_x_len.data();
        d_cu_k_len = d_cu_q_len + bsz + 1;
        h_cu_k_len = h_cu_q_len + bsz + 1;

        h_cu_q_len[0] = h_cu_k_len[0] = 0;

        std::inclusive_scan(h_q_len.data(), h_q_len.data() + bsz, h_cu_q_len + 1);
        std::inclusive_scan(h_k_len.data(), h_k_len.data() + bsz, h_cu_k_len + 1);

        Copy(h_cu_x_len.slice(0, 2 * bsz + 2), d_cu_x_len.slice(0, 2 * bsz + 2));

        event.Record(core::Context::stream());

        decode_num = *args.at("decode_num").data<int>();
        prefil_num = *args.at("prefil_num").data<int>();

        finished  = args.at("finished").buffer();
        rope_base = args.at("rope_base").buffer();

        cu_block_nums = args.at("cu_block_nums").buffer();
        kv_block_ptrs = args.at("kv_block_ptrs").buffer();

        input  = input_;
        output = output_;
    }

    core::Tensor input;
    core::Tensor output;

    core::Buffer_<int> h_q_len;
    core::Buffer_<int> h_k_len;

    core::Buffer_<int> d_cu_x_len;
    core::Buffer_<int> h_cu_x_len;

    int* d_cu_q_len;
    int* d_cu_k_len;
    int* h_cu_q_len;
    int* h_cu_k_len;

    core::Buffer_<bool>  finished;
    core::Buffer_<float> rope_base;

    core::Buffer_<int>       cu_block_nums;
    core::Buffer_<uintptr_t> kv_block_ptrs;

    const void* weights;

    core::Event event;

    int decode_num;
    int prefil_num;
    int layer_id;
};

void Initialize(ForwardParam& p, core::TensorMap& args, const core::Tensor& input, core::Tensor& output)
{
    p.Init(args, input, output);
}

void SetLayer(ForwardParam& p, const void* weights, int layer_id)
{
    p.weights  = weights;
    p.layer_id = layer_id;
}

void Finalize(ForwardParam& p)
{
    // This is used to prevent data-race on `h_cu_q_len`, otherwise it may be modified by later
    // `Init` calls before the HtoD copy is done
    p.event.Sync();
}

const int* d_cu_q_len(ForwardParam& p)
{
    return p.d_cu_q_len;
}

}  // namespace attention

auto UnifiedAttentionLayer::CreateForwardParam(int max_batch_size) -> std::shared_ptr<ForwardParam>
{
    return std::make_shared<ForwardParam>(max_batch_size);
}

UnifiedAttentionLayer::~UnifiedAttentionLayer()
{
    for (auto& s : streams_) {
        s = {};
    }

    check_cuda_error(cudaEventDestroy(aux_event_));
    check_cuda_error(cudaEventDestroy(qkv_event_));
    check_cuda_error(cudaStreamDestroy(aux_stream_));

    aux_event_ = qkv_event_ = {};
    aux_stream_             = {};
}

UnifiedAttentionLayer::UnifiedAttentionLayer(
    const ModelParam& model, const AttentionParam& attn, const LoraParam& lora, int tp_size, const Context& ctx):
    head_num_(model.head_num),
    kv_head_num_(model.kv_head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    local_head_num_(head_num_ / tp_size),
    local_kv_head_num_(model.kv_head_num / tp_size),
    param_(attn),
    model_param_(model),
    lora_param_(lora),
    context_(ctx),
    stream_(ctx.stream),
    linear_(*ctx.linear),
    arch_(getSMVersion())
{
    TM_CHECK_EQ(head_num_ % tp_size, 0) << head_num_ << " " << tp_size;
    TM_CHECK_EQ(head_num_ % kv_head_num_, 0) << head_num_ << " " << kv_head_num_;

    check_cuda_error(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    check_cuda_error(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

    streams_[0] = stream_;
    streams_[1] = aux_stream_;

    init_rope_kernel_param(param_.rope, rope_param_);

    partial_M_ = core::Tensor_<float>({kMaxWorkspaceTokens, local_head_num_}, MEMORY_GPU);
    partial_L_ = core::Tensor_<float>({kMaxWorkspaceTokens, local_head_num_}, MEMORY_GPU);
    partial_O_ = core::Tensor_<float>({kMaxWorkspaceTokens, local_head_num_, size_per_head_}, MEMORY_GPU);
    split_cnt_ = core::Tensor_<int>({kMaxWorkspaceTokens}, MEMORY_GPU);
    barriers_  = core::Tensor_<int>({kMaxWorkspaceTokens, local_head_num_}, MEMORY_GPU);

    Clear(split_cnt_.buffer());
    Clear(barriers_.buffer());
}

void UnifiedAttentionLayer::forward(ForwardParam& p)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    /**
     * input_tensors:
     *   \param input_query [token_num, hidden_dim]
     *   \param cu_q_len [batch_size+1], int
     *   \param cu_k_len [batch_size+1], int
     *   \param cu_block_counts [batch_size+1], int
     *   \param finished [batch_size], bool
     *   \param rope_theta [batch_size], float
     *   \param h_q_len [batch_size], int on cpu
     *   \param h_k_len [batch_size], int on cpu
     *   \param h_cu_q_len [batch_size+1], int on cpu
     *   \param h_cu_k_len [batch_size+1], int on cpu
     *   \param dc_batch_size [1], int on cpu
     *   \param pf_batch_size [1], int on cpu
     *   \param layer_id [1], int on cpu
     *
     * output_tensors:
     *   \param hidden_features [token_num, hidden_dim], float
     *   \param block_ptrs [total_block_counts], void*
     */

    /////////////////////////////////////////////
    /// parse inputs
    const int token_num = p.input.shape(0);

    if (token_num == 0) {
        return;
    }

    const int layer_id = p.layer_id;

    const auto& weights = *(const WeightType*)p.weights;

    // [L, 2, H, s, D]
    const size_t layer_offset = layer_id * 2 * local_kv_head_num_ * param_.cache_block_seq_len * size_per_head_;

    core::Tensor qkv;

    if (weights.qkv.output_dim) {
        // [token_num, hidden_dim] -> [token_num, local_q_kv_head_num, head_dim]
        qkv = linear_.forward(p.input, weights.qkv, LlamaLinear::kGemm);
        sync_check_cuda_error();

        if (model_param_.qk_norm) {
            qk_norm(qkv, weights);
        }
    }
    else {
        qkv = forward_mla(p.input, weights);
    }

    TM_DEBUG_TENSOR(qkv, Concat("qkv", layer_id), 3);

    auto invoke = [&](auto t) -> core::Tensor {
        using T = decltype(t);
        return core_attention<T>(qkv, p, weights);
    };

    core::Tensor attn;

    switch (qkv.dtype()) {
        case TYPE_FP16:
            attn = invoke(half{});
            break;
        case TYPE_BF16:
            attn = invoke(nv_bfloat16{});
            break;
        default:
            TM_CHECK(0) << "not implemented";
    }

    TM_DEBUG_TENSOR(attn, Concat("attn", layer_id), 3);

    //////////////////////////////////////////////
    /// output gemm <Bs,HD> -> <Bs,HD>
    (void)linear_.forward(attn, weights.output, LlamaLinear::kGemm, p.output);
    sync_check_cuda_error();
}

template<class T>
core::Tensor UnifiedAttentionLayer::core_attention(core::Tensor& qkv, const ForwardParam& p, const WeightType& weights)
{
    const auto device = qkv.device();
    const auto dtype  = qkv.dtype();

    const int batch_size = p.decode_num + p.prefil_num;
    const int q_count    = qkv.shape(0);
    const int k_count    = p.h_cu_k_len[batch_size] - p.h_cu_k_len[p.decode_num];
    const int layer_id   = p.layer_id;

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    core::Tensor attn{{q_count, (int)local_head_num_ * (int)size_per_head_}, dtype, device};
    core::Tensor tmp_kv{{2, (int)local_kv_head_num_, k_count + MAX_CTA_S, (int)size_per_head_}, dtype, device};

    auto stream_ptr = streams_.data();

    auto CreateParams = [&](int offset, int batch_size, int max_kv_splits, cudaStream_t stream) {
        AttentionParams<T> params{};

        // Batch offset for `out` and `q` are computed inside the kernel
        params.out = (T*)attn.raw_data();

        params.q      = (T*)qkv.raw_data();
        params.k      = params.q + local_head_num_ * size_per_head_;
        params.v      = params.k + local_kv_head_num_ * size_per_head_;
        params.stride = (local_head_num_ + 2 * local_kv_head_num_) * size_per_head_;

        if (weights.qkv.bias) {
            params.q_bias = weights.qkv.bias.unsafe_data<T>();
            params.k_bias = params.q_bias + local_head_num_ * size_per_head_;
            params.v_bias = params.k_bias + local_kv_head_num_ * size_per_head_;
        }

        params.token_num  = p.h_cu_q_len[offset + batch_size] - p.h_cu_q_len[offset];
        params.batch_size = batch_size;
        /// TODO: maximum on buffer slice
        params.max_q_len = *std::max_element(p.h_q_len.data() + offset, p.h_q_len.data() + offset + batch_size);
        params.max_k_len = *std::max_element(p.h_k_len.data() + offset, p.h_k_len.data() + offset + batch_size);

        // Decoding use only
        params.block_iter_params = BlockIteratorParams{(char**)p.kv_block_ptrs.data(),  //
                                                       p.cu_block_nums.data() + offset,
                                                       layer_id,
                                                       (int)param_.cache_block_seq_len};

        // Prefilling use only
        const int sum_k_len       = p.h_cu_k_len[offset + p.prefil_num] - p.h_cu_k_len[offset];
        params.linear_iter_params = LinearIteratorParams{tmp_kv.raw_data(),  //
                                                         int(2 * sum_k_len * size_per_head_),
                                                         int(sum_k_len * size_per_head_)};

        params.finished = p.finished.data() + offset;
        params.cu_q_len = p.d_cu_q_len + offset;
        params.cu_k_len = p.d_cu_k_len + offset;

        params.num_heads     = local_head_num_;
        params.num_kv_heads  = local_kv_head_num_;
        params.size_per_head = size_per_head_;

        // MSVC does not have M_LOG2E
        params.inv_sqrt_dh = (float)std::log2(expf(1.));
        if (param_.softmax_scale) {  // model predefined softmax scale
            params.inv_sqrt_dh *= param_.softmax_scale;
        }
        else {  // default value
            params.inv_sqrt_dh /= std::sqrt((float)params.size_per_head);
        }

        // rotary embedding
        if (rope_param_.type == RopeType::kDynamic) {
            rope_param_.base = const_cast<float*>(p.rope_base.data()) + offset;
        }
        params.rope_param = rope_param_;

        // logn attn
        params.use_logn_attn           = param_.use_logn_attn;
        params.max_position_embeddings = param_.max_position_embeddings;

        // Decoding use only for now
        params.split_cnt   = split_cnt_.data();
        params.partial_L   = partial_L_.data();
        params.partial_M   = partial_M_.data();
        params.partial_O   = partial_O_.data();
        params.locks       = barriers_.data();
        params.max_split_k = std::min(std::max(1, kMaxWorkspaceTokens / params.token_num), max_kv_splits);

        params.arch   = arch_;
        params.stream = stream;

        params.quant_policy = model_param_.quant_policy;
        return params;
    };

    cudaStream_t pf_stream = stream_;
    cudaStream_t dc_stream = stream_;

    if (p.decode_num && p.prefil_num) {
        pf_stream = aux_stream_;
        check_cuda_error(cudaEventRecord(qkv_event_, stream_));
        check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));
    }

    if (p.prefil_num && !isTuning()) {
        const int offset    = p.decode_num;
        const int sum_k_len = p.h_cu_k_len[offset + p.prefil_num] - p.h_cu_k_len[offset];
        // We are executing prefill & decoding kernels concurrently, but only have 1 workspace
        // disable split kv for prefill for now
        auto params = CreateParams(offset, p.prefil_num, 1, pf_stream);
        if constexpr (sizeof(T) == 2) {
            invokeProcessKV_v2_(params);
            sync_check_cuda_error();

            /// TODO: skip flattening for `sm_80`
            invokeFlattenKV_v2_(params, sum_k_len);
            sync_check_cuda_error();

            dispatchAttention(params);
            sync_check_cuda_error();
        }
    }

    if (p.decode_num && !isTuning()) {
        auto params = CreateParams(0, p.decode_num, kMaxKVSplits, dc_stream);
        if constexpr (sizeof(T) == 2) {
            dispatchDecoding<T>(params);
            sync_check_cuda_error();
        }
    }

    if (p.decode_num && p.prefil_num) {
        check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
        check_cuda_error(cudaStreamWaitEvent(stream_, aux_event_));
    }

    if (isTuning()) {
        rng_.set_stream(stream_);
        rng_.GenerateUniform(attn.data<T>(), attn.size(), .02f, -.01f);
    }

    return attn;
}

core::Tensor UnifiedAttentionLayer::forward_mla(const core::Tensor& hidden_state, const WeightType& w)
{
    const int q_lora_rank  = w.q_a_proj.output_dim;
    const int kv_lora_rank = w.kv_b_proj.input_dim;
    const int qk_rope_dim  = w.kv_a_proj.output_dim - kv_lora_rank;
    const int qk_nope_dim  = std::max(w.q_b_proj.output_dim, w.q_proj.output_dim) / local_head_num_ - qk_rope_dim;
    const int v_head_dim   = w.kv_b_proj.output_dim / local_head_num_ - qk_nope_dim;

    const auto token_num = hidden_state.shape(0);
    const auto dtype     = hidden_state.dtype();

    core::Tensor q;

    if (w.q_proj.weight) {
        q = linear_.forward(hidden_state, w.q_proj);
        sync_check_cuda_error();
    }
    else {
        core::Tensor q_a = linear_.forward(hidden_state, w.q_a_proj);
        sync_check_cuda_error();

        invokeRMSNorm(q_a, q_a, w.q_a_layernorm, model_param_.norm_eps, stream_);
        sync_check_cuda_error();

        q = linear_.forward(q_a, w.q_b_proj);
        sync_check_cuda_error();
    }

    core::Tensor kv_a = linear_.forward(hidden_state, w.kv_a_proj);
    sync_check_cuda_error();

    invokeRMSNorm(kv_a, kv_a, w.kv_a_layernorm, model_param_.norm_eps, stream_);
    sync_check_cuda_error();

    core::Tensor kv_b = linear_.forward(kv_a.slice({0, 0}, {token_num, kv_lora_rank}), w.kv_b_proj);
    sync_check_cuda_error();

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    core::Tensor qkv{{token_num, local_q_kv_head_num, (int)size_per_head_}, dtype, hidden_state.device()};
    MLACopyQKV(dtype,
               qkv.raw_data(),
               q.raw_data(),
               kv_a.raw_data(),
               kv_b.raw_data(),
               token_num,
               local_head_num_,
               qk_nope_dim,
               qk_rope_dim,
               kv_lora_rank,
               v_head_dim,
               stream_);
    sync_check_cuda_error();

    return qkv;
}

void UnifiedAttentionLayer::qk_norm(core::Tensor& qkv, const WeightType& weights)
{
    check_cuda_error(cudaEventRecord(qkv_event_, stream_));
    check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));

    TM_CHECK(model_param_.attn_bias == false) << "not implemented";

    const auto token_num = qkv.shape(0);

    auto qkv3 = qkv.view({token_num, -1, (int)size_per_head_});

    auto q = qkv3.slice({0, 0, 0}, {-1, (int)local_head_num_, -1});
    invokeRMSNormQK(q, weights.q_a_layernorm, model_param_.norm_eps, stream_);
    sync_check_cuda_error();

    auto k = qkv3.slice({0, (int)local_head_num_, 0}, {-1, (int)local_kv_head_num_, -1});
    invokeRMSNormQK(k, weights.kv_a_layernorm, model_param_.norm_eps, aux_stream_);
    sync_check_cuda_error();

    check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
    check_cuda_error(cudaStreamWaitEvent(stream_, aux_event_));
}

}  // namespace turbomind

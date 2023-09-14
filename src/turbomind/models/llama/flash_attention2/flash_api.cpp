/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
// modify from: https://github.com/Dao-AILab/flash-attention

#include "flash.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "static_switch.h"
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <math.h>

void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream)
{
    FP16_SWITCH(true,
                [&] { FWD_HEADDIM_SWITCH(params.d, [&] { run_mha_fwd_<elem_type, kHeadDim>(params, stream); }); });
}

namespace turbomind {

static constexpr int FMHA_VERSION = 2;

template<typename T>
class FlashAttentionOpImpl<T, FMHA_VERSION> {

public:
    using AttentionLayout = BaseAttentionLayout<T>;
    using Params          = BaseAttentionParams<T>;

public:
    FlashAttentionOpImpl(int batch_size, int head_num, int key_len, int seq_len, int size_per_head);
    ~FlashAttentionOpImpl();

    int get_workspace_size() const;

    void operator()(Params& params, cudaStream_t st) const;

private:
    class impl;
    std::unique_ptr<impl> pimpl;
};

template<typename T>
class FlashAttentionOpImpl<T, FMHA_VERSION>::impl {

private:
    using scalar_t =
        typename std::conditional_t<std::is_same<half, typename std::decay<T>::type>::value, cutlass::half_t, T>;
    using Params = typename FlashAttentionOpImpl<T, FMHA_VERSION>::Params;

    int batch_size_;
    int head_num_;
    int key_len_;
    int seq_len_;
    int size_per_head_;

public:
    impl(int batch_size, int head_num, int key_len, int seq_len, int size_per_head):
        batch_size_(batch_size),
        head_num_(head_num),
        key_len_(key_len),
        seq_len_(seq_len),
        size_per_head_(size_per_head)
    {
    }

    ~impl() {}

    int get_workspace_size() const
    {
        return 0;
    }

    void operator()(Params& params, cudaStream_t st) const
    {
        const float      qk_scale = static_cast<float>(1.f / sqrtf(size_per_head_ * 1.f));
        Flash_fwd_params fwd_params;
        memset(&fwd_params, 0, sizeof(fwd_params));

        fwd_params.q_ptr = reinterpret_cast<void*>(params.query);
        fwd_params.k_ptr = reinterpret_cast<void*>(params.key);
        fwd_params.v_ptr = reinterpret_cast<void*>(params.val);

        fwd_params.k_batched_ptr    = reinterpret_cast<void**>(params.layout_k.batch_seqs);
        fwd_params.v_batched_ptr    = reinterpret_cast<void**>(params.layout_v.batch_seqs);
        fwd_params.k_batched_offset = params.layout_k.batch_seqs_offset;
        fwd_params.v_batched_offset = params.layout_v.batch_seqs_offset;

        fwd_params.q_batch_stride = params.layout_q.stride_batch;
        fwd_params.k_batch_stride = params.layout_k.stride_batch;
        fwd_params.v_batch_stride = params.layout_v.stride_batch;
        fwd_params.q_row_stride   = params.layout_q.stride_seq;
        fwd_params.k_row_stride   = params.layout_k.stride_seq;
        fwd_params.v_row_stride   = params.layout_v.stride_seq;
        fwd_params.q_head_stride  = params.layout_q.stride_head;
        fwd_params.v_head_stride  = params.layout_v.stride_head;
        fwd_params.k_head_stride  = params.layout_k.stride_head;

        fwd_params.h           = head_num_;
        fwd_params.h_k         = head_num_ / params.group_size;
        fwd_params.h_h_k_ratio = params.group_size;

        fwd_params.o_ptr = reinterpret_cast<void*>(params.attn_out);

        fwd_params.o_batch_stride = params.layout_o.stride_batch;
        fwd_params.o_row_stride   = params.layout_o.stride_seq;
        fwd_params.o_head_stride  = params.layout_o.stride_head;

        fwd_params.p_ptr = nullptr;

        fwd_params.b                = batch_size_;
        fwd_params.seqlen_q         = seq_len_;
        fwd_params.seqlen_k         = key_len_;
        fwd_params.d                = size_per_head_;
        fwd_params.seqlen_q_rounded = 0;
        fwd_params.seqlen_k_rounded = 0;

        fwd_params.scale_softmax      = qk_scale;
        fwd_params.scale_softmax_log2 = qk_scale * M_LOG2E;

        fwd_params.cu_seqlens_q = params.cu_seqlens_q;
        fwd_params.cu_seqlens_k = params.cu_seqlens_k;

        fwd_params.actual_seqlen_q = params.actual_seqlen_q;
        fwd_params.actual_seqlen_k = params.actual_seqlen_k;

        fwd_params.blockmask = reinterpret_cast<void*>(params.mask);

        fwd_params.is_bf16   = false;
        fwd_params.is_causal = true;

        fwd_params.q_enable_seqlen = params.layout_q.use_seqlens;
        fwd_params.o_enable_seqlen = params.layout_o.use_seqlens;

        run_mha_fwd(fwd_params, st);
    }
};

template<typename T>
FlashAttentionOpImpl<T, FMHA_VERSION>::FlashAttentionOpImpl(
    int batch_size, int head_num, int key_len, int seq_len, int size_per_head):
    pimpl{std::make_unique<FlashAttentionOpImpl<T, FMHA_VERSION>::impl>(
        batch_size, head_num, key_len, seq_len, size_per_head)}
{
}

template<typename T>
FlashAttentionOpImpl<T, FMHA_VERSION>::~FlashAttentionOpImpl()
{
}

template<typename T>
int FlashAttentionOpImpl<T, FMHA_VERSION>::get_workspace_size() const
{
    return pimpl->get_workspace_size();
}

template<typename T>
void FlashAttentionOpImpl<T, FMHA_VERSION>::operator()(Params& params, cudaStream_t st) const
{
    pimpl->operator()(params, st);
}

template class FlashAttentionOpImpl<float, FMHA_VERSION>;
template class FlashAttentionOpImpl<half, FMHA_VERSION>;

}  // namespace turbomind

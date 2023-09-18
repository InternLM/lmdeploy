// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/cuda_bf16_wrapper.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <numeric>

namespace turbomind {

template<typename T>
void invokeRootMeanSquareNorm(T* out, const T* input, const T* scale, float eps, int m, int n, cudaStream_t stream);

template<typename T>
void invokeAddResidual(T* out, const T* in, int m, int n, cudaStream_t stream);

void invokeFixInputIds(int*         ids,
                       const int*   input_ids,
                       const int*   input_lengths,
                       int          batch_size,
                       int          seq_len,
                       int          max_input_len,
                       cudaStream_t st);

template<typename T>
void invokeSliceCausalMask(T* mask, int seq_len, int key_len, int step, int batch_size, cudaStream_t stream);

template<typename T>
void invokeCreateCausalMasks(
    T* mask, const int* q_lens, const int* k_lens, int max_q_len, int max_k_len, int batch_size, cudaStream_t stream);

template<typename T>
void invokeExtendKVCache(T**          k_dst,
                         T**          v_dst,
                         size_t       layer_offset,
                         const T*     k_src,
                         const T*     v_src,
                         int          batch_size,
                         const int*   query_length,
                         int          max_q_len,
                         const int*   history_length,
                         int          max_seq_len,
                         int          size_per_head,
                         int          local_head_num,
                         cudaStream_t stream,
                         int          quant,
                         const float* kv_scale);

template<typename T>
void invokeTransposeKVCache(T*           key_cache_trans,
                            T*           val_cache_trans,
                            const T**    key_cache,
                            const T**    val_cache,
                            size_t       layer_offset,
                            int          batch_size,
                            const int*   key_length,
                            int          max_kv_len,
                            int          max_seq_len,
                            int          size_per_head,
                            int          head_num,
                            int          head_n_rep,
                            cudaStream_t stream,
                            int          quant_policy,
                            const float* kv_scale);

void invokeGatherOutput(int*         output_ids,
                        const int*   ids,
                        const int*   context_length,
                        int          max_context_len,
                        int          max_gen_step,
                        int          max_output_len,
                        int          batch_size,
                        cudaStream_t stream);

void invokeMyCopyInt(int* dst, const int* src, size_t count, cudaStream_t st);

template<typename T>
struct BaseAttentionLayout {
    int    stride_batch;
    int    stride_seq;
    int    stride_head;
    bool   use_seqlens       = false;
    size_t batch_seqs_offset = 0;
    T**    batch_seqs        = nullptr;
};

template<typename T>
struct BaseAttentionParams {
    T*                     attn_out;
    T*                     query;
    T*                     key;
    T*                     val;
    T*                     mask;
    float*                 out_accum       = nullptr;
    int*                   cu_seqlens_q    = nullptr;
    int*                   cu_seqlens_k    = nullptr;
    int*                   actual_seqlen_q = nullptr;
    int*                   actual_seqlen_k = nullptr;
    size_t                 group_size      = 1;
    BaseAttentionLayout<T> layout_q;
    BaseAttentionLayout<T> layout_k;
    BaseAttentionLayout<T> layout_v;
    BaseAttentionLayout<T> layout_o;
};

template<typename T, int version>
class FlashAttentionOpImpl {
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
class FlashAttentionOp {
public:
    using AttentionLayout = BaseAttentionLayout<T>;
    using Params          = BaseAttentionParams<T>;

public:
    FlashAttentionOp(int batch_size, int head_num, int key_len, int seq_len, int size_per_head);

    int get_workspace_size() const;

    void operator()(Params& params, cudaStream_t st) const;

private:
    int batch_size_;
    int head_num_;
    int key_len_;
    int seq_len_;
    int size_per_head_;
    int op_version_;
};

template<typename T>
inline void dump(const T* x, int size, cudaStream_t st, const char* msg, bool full = false)
{
    std::vector<T> h_x(size);
    cudaMemcpyAsync(h_x.data(), x, sizeof(T) * size, cudaMemcpyDefault, st);
    cudaStreamSynchronize(st);
    fprintf(stderr, "\n%s:\n", msg);
    std::vector<float> h_y(h_x.begin(), h_x.end());
    float              asum = 0.f;
    for (const auto& x : h_y) {
        asum += std::fabs(x);
    }
    if (full) {
        for (int i = 0; i < size; ++i) {
            printf("%d %.8f\n", i, h_y[i]);
        }
    }
    else {
        for (int i = 0; i < 8; ++i) {
            fprintf(stderr, "%.8f\n", h_y[i]);
        }
        for (int i = size - 8; i < size; ++i) {
            fprintf(stderr, "%.8f\n", h_y[i]);
        }
    }
    fprintf(stderr, "\nasum = %f\n", asum);
    // getchar();
}

template<typename T>
struct TempBuffer {
    TempBuffer(size_t size)
    {
        deviceMalloc(&data, size, false);
    }
    T* data;
};

inline void dump_sequence_len(int* d_seq_len, int step, int tp_rank, cudaStream_t st)
{
    int h_seq_len = -1;
    cudaMemcpyAsync(&h_seq_len, d_seq_len, sizeof(int), cudaMemcpyDefault, st);
    cudaStreamSynchronize(st);
    TM_LOG_ERROR("--------> rank = %d, step = %d, seq_len = %d <--------", tp_rank, step, h_seq_len);
}

}  // namespace turbomind

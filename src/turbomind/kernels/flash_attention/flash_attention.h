#pragma once

#include <memory>

namespace turbomind {

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

}  // namespace turbomind

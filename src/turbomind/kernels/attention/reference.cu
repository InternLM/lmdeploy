
#include "array_ops.h"
#include "reference.h"
#include "src/turbomind/kernels/unfused_attention_kernels.h"

namespace turbomind {

template<class T>
__global__ void createCausalMasks(T* mask, const int* q_lens, const int* k_lens, int max_q_len, int max_k_len)
{
    const auto q_len = q_lens ? q_lens[blockIdx.x] : max_q_len;
    const auto k_len = k_lens ? k_lens[blockIdx.x] : max_k_len;
    mask += blockIdx.x * max_q_len * max_k_len;
    for (int i = threadIdx.x; i < max_q_len * max_k_len; i += blockDim.x) {
        const int q        = i / max_k_len;  // [0, max_q_len)
        const int k        = i % max_k_len;  // [0, max_k_len)
        bool      is_valid = q < q_len && k < k_len && k <= q + (k_len - q_len);
        mask[i]            = static_cast<T>(is_valid);
    }
}

// [B, H, S, D]
template<class T>
__global__ void applyRotaryEmbedding(T* k_cache, int max_k_len, int head_num, int head_dim, float rope_base)
{
    const int    ti = blockIdx.x;
    const size_t hi = blockIdx.y;
    const size_t bi = blockIdx.z;

    constexpr int kVecSize = 2;
    const int     history  = 0;

    for (int d = threadIdx.x * kVecSize; d < head_dim; d += blockDim.x * kVecSize) {
        const size_t idx =
            bi * head_num * max_k_len * head_dim + hi * max_k_len * head_dim + (history + ti) * head_dim + d;

        Array<T, kVecSize> vec_K;

        Load(vec_K, &k_cache[idx]);

        RotaryEmbedding<kVecSize> rope(rope_base, head_dim, history + ti, {d, 0});

        rope.apply(vec_K);

        Store(&k_cache[idx], vec_K);
    }
}

template<class T>
void invokeApplyRotaryEmbedding(
    T* k_cache, int max_k_len, int head_num, int head_dim, float rope_base, int batch_size, cudaStream_t stream)
{
    int  threads = 128;
    dim3 blocks(max_k_len, head_num, batch_size);

    applyRotaryEmbedding<<<blocks, threads, 0, stream>>>(k_cache, max_k_len, head_num, head_dim, rope_base);
}

template void invokeApplyRotaryEmbedding(
    half* k_cache, int max_k_len, int head_num, int head_dim, float rope_base, int batch_size, cudaStream_t stream);

template<class T>
__global__ void processQKV(T*       q_out,    // [B, H, s, D]
                           T*       k_cache,  // [B, H, S, D]
                           T*       v_cache,  // [B, H, S, D]
                           const T* qkv,      // [B, s, H, D]
                           int      max_q_len,
                           int      max_k_len,
                           int      head_num,
                           int      head_dim,
                           int      kv_head_num,
                           float    rope_theta)
{
    const int    ti = blockIdx.x;
    const size_t hi = blockIdx.y;
    const size_t bi = blockIdx.z;

    const int history = max_k_len - max_q_len;

    size_t qkv_head_num = head_num + 2 * kv_head_num;

    auto q = qkv + (bi * max_q_len + ti) * qkv_head_num * head_dim;
    auto k = q + head_num * head_dim;
    auto v = k + kv_head_num * head_dim;

    constexpr int kVecSize = 2;

    for (int d = threadIdx.x * kVecSize; d < head_dim; d += blockDim.x * kVecSize) {
        const auto         idx = bi * head_num * max_q_len * head_dim + hi * max_q_len * head_dim + ti * head_dim + d;
        Array<T, kVecSize> vec;
        Ldg(vec, &q[hi * head_dim + d]);
        if (rope_theta) {
            RotaryEmbedding<kVecSize> rope(rope_theta, head_dim, history + ti, {d, 0});
            rope.apply(vec);
        }
        Store(&q_out[idx], vec);
    }

    if (hi >= kv_head_num) {
        return;
    }

    for (int d = threadIdx.x * kVecSize; d < head_dim; d += blockDim.x * kVecSize) {
        const auto idx =
            bi * kv_head_num * max_k_len * head_dim + hi * max_k_len * head_dim + (history + ti) * head_dim + d;
        Array<T, kVecSize> vec_K;
        Array<T, kVecSize> vec_V;
        Ldg(vec_K, &k[hi * head_dim + d]);
        Ldg(vec_V, &v[hi * head_dim + d]);
        if (rope_theta) {
            RotaryEmbedding<kVecSize> rope(rope_theta, head_dim, history + ti, {d, 0});
            rope.apply(vec_K);
        }
        Store(&k_cache[idx], vec_K);
        Store(&v_cache[idx], vec_V);
    }
}

template<class T>
Reference<T>::Reference(Type type, cudaStream_t stream): type_{type}, stream_(stream)
{
    if (type == kUNFUSED) {
        cublasCreate(&cublas_);
        cublasSetStream(cublas_, stream);
    }
}

template<class T>
void Reference<T>::Reshape(
    size_t max_q_len, size_t max_k_len, size_t head_num, size_t head_dim, size_t kv_head_num, size_t batch_size)
{
    std::cout << max_q_len << " " << max_k_len << " " << head_num << " " << head_dim << " " << batch_size << "\n";

    q_.resize(batch_size * head_num * max_q_len * head_dim);
    mask_.resize(batch_size * max_q_len * max_k_len);

    if (type_ == kUNFUSED) {
        std::cout << "size of QK buf: "
                  << ((batch_size * head_num * max_q_len * max_k_len * sizeof(float)) / float(1 << 30)) << " GB\n";
        qk_.resize(batch_size * head_num * max_q_len * max_k_len);
        pr_.resize(batch_size * head_num * max_q_len * max_k_len);
        out_.resize(batch_size * max_q_len * head_num * head_dim);
        cudaStreamSynchronize(0);
    }
    else if (type_ == kFLASH_ATTENTION) {
        key_cache_ptrs_.resize(batch_size);
        val_cache_ptrs_.resize(batch_size);
        cu_q_seqlens_.resize(batch_size + 1);
        for (size_t i = 0; i <= batch_size; ++i) {
            cu_q_seqlens_[i] = i * max_q_len;
        }
        k_seqlens_.resize(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            k_seqlens_[i] = max_k_len;
        }
    }

    createCausalMasks<<<batch_size, 512, 0, stream_>>>(mask_.data().get(), nullptr, nullptr, max_q_len, max_k_len);

    max_q_len_   = max_q_len;
    max_k_len_   = max_k_len;
    head_num_    = head_num;
    head_dim_    = head_dim;
    kv_head_num_ = kv_head_num;
    batch_size_  = batch_size;
}

template<class T>
void Reference<T>::Execute(T* output, T* k_cache, T* v_cache, const T* qkv)
{
    {
        int  threads = 128;
        dim3 blocks(max_q_len_, head_num_, batch_size_);
        cudaDeviceSynchronize();

        processQKV<<<blocks, threads, 0, stream_>>>(q_.data().get(),  //
                                                    k_cache,
                                                    v_cache,
                                                    qkv,
                                                    max_q_len_,
                                                    max_k_len_,
                                                    head_num_,
                                                    head_dim_,
                                                    kv_head_num_,
                                                    10000.f);

        cudaDeviceSynchronize();
    }

    if (type_ == kUNFUSED) {

        float alpha = 1.f;
        float beta  = 0.f;
        cublasGemmStridedBatchedEx(cublas_,
                                   CUBLAS_OP_T,              // trans A
                                   CUBLAS_OP_N,              // trans B
                                   max_k_len_,               // m
                                   max_q_len_,               // n
                                   head_dim_,                // k
                                   &alpha,                   // alpha
                                   k_cache,                  // A
                                   CUDA_R_16F,               // A type
                                   head_dim_,                // lda
                                   max_k_len_ * head_dim_,   // strideA
                                   q_.data().get(),          // B
                                   CUDA_R_16F,               // B type
                                   head_dim_,                // ldb
                                   max_q_len_ * head_dim_,   // stride B
                                   &beta,                    // beta
                                   qk_.data().get(),         // C
                                   CUDA_R_32F,               // C type
                                   max_k_len_,               // ldc
                                   max_q_len_ * max_k_len_,  // stride C
                                   batch_size_ * head_num_,  // batch count
                                   CUBLAS_COMPUTE_32F,       // compute type
                                   CUBLAS_GEMM_DEFAULT);

        MaskedSoftmaxParam<T, float> params{};
        params.attention_score = pr_.data().get();
        params.qk              = qk_.data().get();
        params.attention_mask  = mask_.data().get();
        params.batch_size      = batch_size_;
        params.q_length        = max_q_len_;
        params.k_length        = max_k_len_;
        params.num_heads       = head_num_;
        params.qk_scale        = T(1.f / sqrtf((float)head_dim_));
        invokeMaskedSoftmax(params, stream_);

        cublasGemmStridedBatchedEx(cublas_,
                                   CUBLAS_OP_N,              // trans A
                                   CUBLAS_OP_N,              // trans B
                                   head_dim_,                // m
                                   max_q_len_,               // n
                                   max_k_len_,               // k
                                   &alpha,                   // alpha
                                   v_cache,                  // A
                                   CUDA_R_16F,               // A type
                                   head_dim_,                // lda
                                   max_k_len_ * head_dim_,   // strideA
                                   pr_.data().get(),         // B
                                   CUDA_R_16F,               // B type
                                   max_k_len_,               // ldb
                                   max_q_len_ * max_k_len_,  // stride B
                                   &beta,                    // beta
                                   out_.data().get(),        // C [b, h, q, d]
                                   CUDA_R_16F,               // C type
                                   head_dim_,                // ldc
                                   max_q_len_ * head_dim_,   // stride C
                                   batch_size_ * head_num_,  // batch count
                                   CUBLAS_COMPUTE_32F,       // compute type
                                   CUBLAS_GEMM_DEFAULT);
        // [B, H, Q, D] -> [B, Q, H, D]
        invokeTransposeAttentionOutRemovePadding(out_.data().get(),
                                                 output,
                                                 batch_size_ * max_q_len_,
                                                 batch_size_,
                                                 max_q_len_,
                                                 head_num_,
                                                 head_dim_,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 stream_);
    }
    else if (type_ == kFLASH_ATTENTION) {

        for (int i = 0; i < batch_size_; ++i) {
            key_cache_ptrs_[i] = k_cache + i * kv_head_num_ * max_k_len_ * head_dim_;
            val_cache_ptrs_[i] = v_cache + i * kv_head_num_ * max_k_len_ * head_dim_;
        }

        using AttentionOp = FlashAttentionOp<T>;
        using Layout      = typename AttentionOp::AttentionLayout;
        Layout layout_q{int(head_num_ * max_q_len_ * head_dim_), int(head_dim_), int(max_q_len_ * head_dim_)};
        Layout layout_k{int(head_num_ * max_k_len_ * head_dim_),
                        int(head_dim_),
                        int(max_k_len_ * head_dim_),
                        false,
                        0,
                        key_cache_ptrs_.data().get()};
        Layout layout_v{int(head_num_ * max_k_len_ * head_dim_),
                        int(head_dim_),
                        int(max_k_len_ * head_dim_),
                        false,
                        0,
                        val_cache_ptrs_.data().get()};
        Layout layout_o{
            int(head_num_ * max_q_len_ * head_dim_),
            int(head_num_ * head_dim_),
            int(head_dim_),
            true,
        };
        size_t                       group_size = size_t(head_num_ / kv_head_num_);
        AttentionOp                  flash_attention(batch_size_, head_num_, max_k_len_, max_q_len_, head_dim_);
        typename AttentionOp::Params attn_params{output,
                                                 q_.data().get(),
                                                 nullptr,             // k ptr
                                                 nullptr,             // v ptr
                                                 mask_.data().get(),  // attention mask
                                                 nullptr,             // qk buf float
                                                 cu_q_seqlens_.data().get(),
                                                 nullptr,
                                                 nullptr,
                                                 k_seqlens_.data().get(),
                                                 group_size,
                                                 layout_q,
                                                 layout_k,
                                                 layout_v,
                                                 layout_o};

        //
        flash_attention(attn_params, stream_);
    }
    else {
        std::abort();
    }
}

template class Reference<half>;

}  // namespace turbomind
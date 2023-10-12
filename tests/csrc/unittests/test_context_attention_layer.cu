// Copyright (c) OpenMMLab. All rights reserved.

#include <assert.h>
#include <cstdlib>
#include <math.h>
#include <numeric>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#undef TORCH_CUDA

#include "src/turbomind/kernels/bert_preprocess_kernels.h"
#include "src/turbomind/kernels/unfused_attention_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"
#include "unittest_utils.h"

using namespace turbomind;

template<typename scalar_t>
__global__ void pad_query_kernel(
    scalar_t* query_ptr, const int* cu_seqlens, int batch_size, int batch_stride, int seq_stride, int max_seq_length)
{
    int batch_id = blockIdx.x;
    int seqlen   = cu_seqlens[batch_id + 1] - cu_seqlens[batch_id];

    query_ptr += batch_id * batch_stride;
    for (int tid = threadIdx.x; tid < batch_stride; tid += blockDim.x) {
        int seq_id = (tid / seq_stride) % max_seq_length;
        if (seq_id >= seqlen) {
            query_ptr[tid] = scalar_t(0.0f);
        }
    }
}

template<typename scalar_t>
void pad_query(scalar_t*    query_ptr,
               const int*   cu_seqlens,
               int          batch_size,
               int          batch_stride,
               int          seq_stride,
               int          max_seq_length,
               cudaStream_t stream)
{
    pad_query_kernel<<<batch_size, 512, 0, stream>>>(
        query_ptr, cu_seqlens, batch_size, batch_stride, seq_stride, max_seq_length);
}

template<typename scalar_t>
__global__ void
pad_out_kernel(scalar_t* out_ptr, const int* cu_seqlens, int batch_size, int batch_stride, int seq_stride)
{
    int seqlen = cu_seqlens[batch_size];

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < batch_size * batch_stride;
         tid += blockDim.x * gridDim.x) {
        int seq_id = (tid / seq_stride);
        if (seq_id >= seqlen) {
            out_ptr[tid] = scalar_t(0.0f);
        }
    }
}

template<typename scalar_t>
void pad_out(
    scalar_t* out_ptr, const int* cu_seqlens, int batch_size, int batch_stride, int seq_stride, cudaStream_t stream)
{
    pad_out_kernel<<<batch_size, 512, 0, stream>>>(out_ptr, cu_seqlens, batch_size, batch_stride, seq_stride);
}

template<typename scalar_t>
void naive_mha(scalar_t*        out_ptr,
               scalar_t*        query_ptr,
               scalar_t*        key_ptr,
               scalar_t*        val_ptr,
               scalar_t*        mask_ptr,
               scalar_t*        q_buf_ptr,
               scalar_t*        k_buf_ptr,
               scalar_t*        v_buf_ptr,
               scalar_t*        qk_buf_ptr,
               scalar_t*        out_buf_ptr,
               int*             padding_offset,
               int*             cu_seqlens,
               int              batch_size,
               int              head_num,
               int              key_len,
               int              seq_len,
               int              size_per_head,
               cudaStream_t     stream,
               cublasMMWrapper* cublas_wrapper_)
{
    const scalar_t qk_scale = static_cast<scalar_t>(1.f / sqrtf(size_per_head * 1.f));
    // create

    //////////////////////////////////////////////
    /// Q,K,V
    /// transpose <B,s,h,D> -> <B,h,s,D>
    /// TODO: remove padding
    // invokeTransposeQKV(q_buf_ptr,
    //                    query_ptr,
    //                    batch_size,
    //                    head_num,
    //                    seq_len,
    //                    size_per_head,
    //                    nullptr,  // scale, only used in int8 mode
    //                    0,        // int8_mode
    //                    stream);
    // invokeTransposeQKV(k_buf_ptr,
    //                    key_ptr,
    //                    batch_size,
    //                    head_num,
    //                    key_len,
    //                    size_per_head,
    //                    nullptr,  // scale, only used in int8 mode
    //                    0,        // int8_mode
    //                    stream);
    // invokeTransposeQKV(v_buf_ptr,
    //                    val_ptr,
    //                    batch_size,
    //                    head_num,
    //                    key_len,
    //                    size_per_head,
    //                    nullptr,  // scale, only used in int8 mode
    //                    0,        // int8_mode
    //                    stream);

    q_buf_ptr = query_ptr;
    k_buf_ptr = key_ptr;
    v_buf_ptr = val_ptr;

    //////////////////////////////////////////////
    /// Q*K batch gemm
    /// -> [B, H, s, t + s]
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        key_len,                  // m
                                        seq_len,                  // n
                                        size_per_head,            // k
                                        k_buf_ptr,                // A
                                        size_per_head,            // lda
                                        key_len * size_per_head,  // strideA
                                        q_buf_ptr,                // B
                                        size_per_head,            // ldb
                                        seq_len * size_per_head,  // strideB
                                        qk_buf_ptr,               // C
                                        key_len,                  // ldc
                                        seq_len * key_len,        // strideC
                                        batch_size * head_num);   // batchCount

    //////////////////////////////////////////////
    /// ! masked softmax (kernel asserts k_length <= 4096)
    MaskedSoftmaxParam<scalar_t, scalar_t> param{};
    param.attention_score    = qk_buf_ptr;
    param.qk                 = qk_buf_ptr;
    param.attention_mask     = mask_ptr;
    param.batch_size         = batch_size;
    param.q_length           = seq_len;
    param.k_length           = key_len;
    param.num_heads          = head_num;
    param.qk_scale           = qk_scale;
    param.linear_bias_slopes = nullptr;
    invokeMaskedSoftmax(param, stream);

    //////////////////////////////////////////////
    /// softmax(QK)*V batch gemm
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        size_per_head,            // m
                                        seq_len,                  // n
                                        key_len,                  // k
                                        v_buf_ptr,                // A
                                        size_per_head,            // lda
                                        key_len * size_per_head,  // strideA,
                                        qk_buf_ptr,               // B
                                        key_len,                  // ldb
                                        key_len * seq_len,        // strideB
                                        out_buf_ptr,              // C
                                        size_per_head,            // ldc,
                                        seq_len * size_per_head,  // strideC
                                        batch_size * head_num);   // batchCount

    //////////////////////////////////////////////
    /// transpose <B,h,s,D> -> <B,s,h,D>
    int num_token = batch_size * seq_len;
    invokeTransposeAttentionOutRemovePadding(out_buf_ptr,
                                             out_ptr,
                                             num_token,
                                             batch_size,
                                             seq_len,
                                             head_num,
                                             size_per_head,
                                             padding_offset,
                                             nullptr,
                                             0,
                                             stream);

    pad_out(out_ptr, cu_seqlens, batch_size, head_num * seq_len * size_per_head, head_num * size_per_head, stream);
}

static const char* usage = "Usage: %s <batch-size> <num-heads> <key-len> <query-len> <size-per-head>\n"
                           "Example: $test_context_attention_layer 2, 8, 1024, 512, 128\n";

int main(int argc, const char* argv[])
{
    using namespace turbomind;
    using scalar_t                            = half;
    static const cudaDataType_t kCudaDataType = std::is_same<scalar_t, half>::value ? CUDA_R_16F : CUDA_R_32F;

    Logger::getLogger().setLevel(Logger::INFO);

    if (argc != 6) {
        printf(usage, argv[0]);
        return EXIT_FAILURE;
    }

    // First create an instance of an engine.
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{rnd_device()};  // Generates random integers

    int batch_size    = atoi(argv[1]);
    int num_heads     = atoi(argv[2]);
    int key_len       = atoi(argv[3]);
    int seq_len       = atoi(argv[4]);
    int size_per_head = atoi(argv[5]);

    // Create stream and handle
    cudaStream_t     stream;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);

    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in");

    Allocator<AllocatorType::CUDA> allocator(getDevice());
    allocator.setStream(stream);
    std::mutex*     cublas_wrapper_mutex = new std::mutex();
    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
    cublas_wrapper.setGemmConfig(kCudaDataType, kCudaDataType, kCudaDataType, kCudaDataType);

    // initialize device
    scalar_t* query_ptr =
        (scalar_t*)allocator.malloc(batch_size * num_heads * seq_len * size_per_head * sizeof(scalar_t));
    scalar_t* key_ptr =
        (scalar_t*)allocator.malloc(batch_size * num_heads * key_len * size_per_head * sizeof(scalar_t));
    scalar_t* val_ptr =
        (scalar_t*)allocator.malloc(batch_size * num_heads * key_len * size_per_head * sizeof(scalar_t));
    scalar_t* mask_ptr = (scalar_t*)allocator.malloc(batch_size * seq_len * key_len * sizeof(scalar_t));
    scalar_t* expect_out_ptr =
        (scalar_t*)allocator.malloc(batch_size * num_heads * seq_len * size_per_head * sizeof(scalar_t), true);
    scalar_t* actual_out_ptr =
        (scalar_t*)allocator.malloc(batch_size * num_heads * seq_len * size_per_head * sizeof(scalar_t), true);
    scalar_t* q_buf_ptr =
        (scalar_t*)allocator.malloc(batch_size * num_heads * seq_len * size_per_head * sizeof(scalar_t), true);
    scalar_t* k_buf_ptr =
        (scalar_t*)allocator.malloc(batch_size * num_heads * key_len * size_per_head * sizeof(scalar_t), true);
    scalar_t* v_buf_ptr =
        (scalar_t*)allocator.malloc(batch_size * num_heads * key_len * size_per_head * sizeof(scalar_t), true);
    scalar_t* qk_buf_ptr =
        (scalar_t*)allocator.malloc(batch_size * num_heads * seq_len * key_len * sizeof(scalar_t), true);
    scalar_t* out_buf_ptr =
        (scalar_t*)allocator.malloc(batch_size * num_heads * seq_len * size_per_head * sizeof(scalar_t), true);

    auto* h_pinned_token_num_ptr = (size_t*)allocator.malloc(sizeof(size_t), true);
    auto* padding_offset_ptr     = (int*)allocator.malloc(sizeof(int) * batch_size * seq_len, false);
    auto* cu_seqlens_ptr         = (int*)allocator.malloc(sizeof(int) * (batch_size + 1), false);
    // auto* input_lengths  = (int*)allocator.malloc(sizeof(int) * batch_size, false);
    thrust::device_vector<int> input_lengths(batch_size);
    thrust::host_vector<int>   input_lengths_host(batch_size);
    thrust::device_vector<int> kv_lengths(batch_size);
    thrust::host_vector<int>   kv_lengths_host(batch_size);

    cudaRandomUniform<scalar_t>(query_ptr, batch_size * num_heads * seq_len * size_per_head);
    cudaRandomUniform<scalar_t>(key_ptr, batch_size * num_heads * key_len * size_per_head);
    cudaRandomUniform<scalar_t>(val_ptr, batch_size * num_heads * key_len * size_per_head);
    cudaRandomUniform<scalar_t>(mask_ptr, batch_size * seq_len * key_len);

    // create random length for batch
    {
        std::uniform_int_distribution<int> dist{seq_len / 2, seq_len};
        auto                               gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
        std::generate(begin(input_lengths_host), end(input_lengths_host), gen);
        thrust::copy(input_lengths_host.begin(), input_lengths_host.end(), input_lengths.begin());
    }
    size_t  h_token_num = 0;
    size_t* h_pinned_token_num;
    auto    input_lengths_ptr = thrust::raw_pointer_cast(input_lengths.data());
    cudaMallocHost((void**)&h_pinned_token_num, sizeof(size_t));
    invokeGetPaddingOffsetAndCuSeqLens(h_pinned_token_num,
                                       &h_token_num,
                                       padding_offset_ptr,
                                       cu_seqlens_ptr,
                                       input_lengths_ptr,
                                       batch_size,
                                       seq_len,
                                       stream);
    cudaFreeHost((void*)h_pinned_token_num);

    {
        std::uniform_int_distribution<int> dist{seq_len, key_len};
        auto                               gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
        std::generate(begin(kv_lengths_host), end(kv_lengths_host), gen);
        thrust::copy(kv_lengths_host.begin(), kv_lengths_host.end(), kv_lengths.begin());
    }
    auto kv_lengths_ptr = thrust::raw_pointer_cast(kv_lengths.data());
    // deviceFill(kv_lengths_ptr, batch_size, key_len, stream);

    invokeCreateCausalMasks(mask_ptr, input_lengths_ptr, kv_lengths_ptr, seq_len, key_len, batch_size, stream);
    // deviceFill(mask_ptr, batch_size*key_len*seq_len, scalar_t(1), stream);

    // compute gt
    naive_mha<scalar_t>(expect_out_ptr,
                        query_ptr,
                        key_ptr,
                        val_ptr,
                        mask_ptr,
                        q_buf_ptr,
                        k_buf_ptr,
                        v_buf_ptr,
                        qk_buf_ptr,
                        out_buf_ptr,
                        padding_offset_ptr,
                        cu_seqlens_ptr,
                        batch_size,
                        num_heads,
                        key_len,
                        seq_len,
                        size_per_head,
                        stream,
                        &cublas_wrapper);

    // compute actual
#ifdef _MSC_VER
    static constexpr int FMHA_VERSION = 1;
#else
    static constexpr int FMHA_VERSION = 2;
#endif
    using AttentionOp = FlashAttentionOpImpl<scalar_t, FMHA_VERSION>;
    using Layout      = typename AttentionOp::AttentionLayout;
    Layout      layout_q{num_heads * seq_len * size_per_head, size_per_head, seq_len * size_per_head};
    Layout      layout_k{num_heads * key_len * size_per_head, size_per_head, key_len * size_per_head};
    Layout      layout_v{num_heads * key_len * size_per_head, size_per_head, key_len * size_per_head};
    Layout      layout_o{num_heads * seq_len * size_per_head, num_heads * size_per_head, size_per_head, true};
    AttentionOp flash_attention(batch_size, num_heads, key_len, seq_len, size_per_head);
    float*      accum_buf_ptr = (float*)allocator.malloc(flash_attention.get_workspace_size(), true);

    typename AttentionOp::Params attn_params{actual_out_ptr,
                                             query_ptr,
                                             key_ptr,
                                             val_ptr,
                                             mask_ptr,
                                             accum_buf_ptr,
                                             cu_seqlens_ptr,
                                             nullptr,
                                             nullptr,
                                             kv_lengths_ptr,
                                             1,
                                             layout_q,
                                             layout_k,
                                             layout_v,
                                             layout_o};
    flash_attention(attn_params, stream);
    sync_check_cuda_error();

    int num_rows = 8;
    // printf("query:\n");
    // printMatrix(query_ptr, num_rows, 8, size_per_head, true);
    // printf("expect:\n");
    // printMatrix(expect_out_ptr, num_rows, 8, size_per_head, true);
    // printf("actual:\n");
    // printMatrix(actual_out_ptr, num_rows, 8, size_per_head, true);
    checkResult(
        "all close:", actual_out_ptr, expect_out_ptr, batch_size * num_heads * seq_len * size_per_head, true, true);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
}

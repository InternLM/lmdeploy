// Copyright (c) OpenMMLab. All rights reserved.

#include "attention.h"
#include "block.h"
#include "decoding.h"
#include "kv_cache_utils_v2.h"
#include "src/turbomind/kernels/attention/attention_params.h"
#include "src/turbomind/kernels/attention/reference.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <thrust/universal_vector.h>
#include <utility>

using namespace turbomind;

// [b, h, s, d] : current -> stride_h=s, stride_s=1, stride_b=hs
// [cu_q, h, d] : qkvgemm -> stride_h=1, stride_s=h, stride_b=0
// [h, cu_s, d] : prefill -> stride_h=s, stride_s=1, stride_b=0

template<class T, class Tkv>
struct Config {
    int head_dim_;
    int head_num_;
    int block_len_;

    TM_HOST_DEVICE constexpr int t_bits() const
    {
        if constexpr (std::is_same_v<T, Tkv>) {
            return 0;
        }
        else {
            return bitsof<T>;
        }
    }

    TM_HOST_DEVICE constexpr int q_bits() const
    {
        return bitsof<Tkv>;
    }

    TM_HOST_DEVICE constexpr int head_dim() const
    {
        return head_dim_;
    }

    TM_HOST_DEVICE int head_num() const
    {
        return head_num_;
    }

    TM_HOST_DEVICE constexpr int block_len() const
    {
        return block_len_;
    }
};

// [S/S, H, S, D] <-> [S/b, H, b, D]
template<class Tkv, class T>
void TestBlocks(const thrust::universal_vector<T>& k_cache,        // [B, H, S, D]
                const thrust::universal_vector<T>& v_cache,        // [B, H, S, D]
                thrust::universal_vector<char>&    blocks,         // block data
                thrust::universal_vector<char*>&   k_ptrs,         // block ptrs
                thrust::universal_vector<int>&     cu_block_cnts,  // cumulative block counts
                const size_t                       head_num,
                const size_t                       head_dim,
                const size_t                       block_seq_len,
                const size_t                       batch_size,
                int                                quant_policy)
{
    const size_t seq_len  = k_cache.size() / (head_dim * head_num * batch_size);
    const size_t n_blocks = (seq_len + block_seq_len - 1) / block_seq_len;

    Config<T, Tkv> config{(int)head_dim, (int)head_num, (int)block_seq_len};
    block::Layout  layout{config};

    dump(layout);

    const size_t kHSD = head_num * seq_len * head_dim;

    std::cout << "batch_size = " << batch_size << ", seq_len = " << seq_len << ", block_size = " << block_seq_len
              << ", block_num = " << n_blocks << "\n";

    thrust::universal_vector<T> kv_cache(k_cache.size() * 2);  // [B, 2, H, S, D]

    {  // interleave K/V
        auto k_src = k_cache.begin();
        auto v_src = v_cache.begin();
        auto dst   = kv_cache.begin();
        for (size_t i = 0; i < batch_size; ++i) {
            dst = thrust::copy_n(k_src, kHSD, dst);
            dst = thrust::copy_n(v_src, kHSD, dst);
            k_src += kHSD;
            v_src += kHSD;
        }
    }

    // const int kHsD = head_num * block_seq_len * head_dim;

    // [B, S/s, 2, H, s, D]
    // blocks.resize(batch_size * n_blocks * 2 * kHsD);
    blocks.resize(batch_size * n_blocks * layout.block_size(1));
    thrust::fill(blocks.begin(), blocks.end(), NAN);
    k_ptrs.resize(batch_size * n_blocks + 1);  // +1 padding

    std::vector<size_t> idxs(batch_size * n_blocks);
    std::iota(idxs.begin(), idxs.end(), 0);

    std::random_device rd;
    std::mt19937       g(rd());
    std::shuffle(idxs.begin(), idxs.end(), g);

    for (size_t i = 0; i < idxs.size(); ++i) {
        // k_ptrs[i] = blocks.data().get() + idxs[i] * 2 * kHsD;
        k_ptrs[i] = blocks.data().get() + idxs[i] * layout.block_size(1);
    }

    thrust::universal_vector<int> seq_lens(batch_size);
    thrust::universal_vector<int> cu_seq_lens(batch_size + 1);
    thrust::fill(seq_lens.begin(), seq_lens.end(), seq_len);
    for (size_t i = 0; i <= batch_size; ++i) {
        cu_seq_lens[i] = i * seq_len;
    }

    std::vector<int> n_blocks_vec(batch_size + 1, n_blocks);
    cu_block_cnts.resize(batch_size + 1);
    std::exclusive_scan(n_blocks_vec.begin(), n_blocks_vec.end(), cu_block_cnts.begin(), 0);

    cudaDeviceSynchronize();

    // [B, 2H, S, D] -> [B, S/s] x [2H, s, D]
    for (int i = 0; i < 1; ++i) {
        // (B, 2, H, S, D) -> blocks
        invokeProcessKV_v2(k_ptrs.data().get(),
                           kv_cache.data().get(),
                           kv_cache.data().get() + head_num * seq_len * head_dim,
                           (T*)nullptr,
                           (T*)nullptr,
                           cu_seq_lens.data().get(),
                           cu_seq_lens.data().get(),
                           cu_block_cnts.data().get(),
                           nullptr,
                           1.,
                           2 * head_num * seq_len,
                           0,
                           seq_len,
                           1,
                           block_seq_len,
                           0,
                           seq_len,
                           head_num,
                           head_dim,
                           batch_size,
                           quant_policy);
    }

    thrust::universal_vector<T> kv_cache_2(kv_cache.size());

    // round trip test
    for (int i = 0; i < 1; ++i) {
        // kv_cache_2 is [B, 2, H, S, D]
        invokeFlattenKV_v2(kv_cache_2.data().get(),
                           kv_cache_2.data().get() + head_num * seq_len * head_dim,
                           k_ptrs.data().get(),
                           cu_seq_lens.data().get(),
                           cu_block_cnts.data().get(),
                           nullptr,
                           1.,
                           2 * head_num * seq_len,
                           0,
                           seq_len,
                           1,
                           block_seq_len,
                           0,
                           seq_len,
                           head_num,
                           head_dim,
                           batch_size,
                           quant_policy);
    }

    cudaDeviceSynchronize();

    if (0) {
        std::cout << ">>> Compare\n";
        Compare(
            kv_cache_2.data().get(), kv_cache.data().get(), head_dim, head_dim, batch_size * 2 * head_num * seq_len, 0);
        std::cout << "<<< Compare\n";
    }
}

#define KV_INT8 0

#define KV_INT4 0

#define DECODING 0

template<class T>
int test_attention()
{
    AttentionParams<T> params{};

    constexpr size_t kHeadDim = 128;

#if DECODING
    // constexpr size_t kHeadNum   = 32;
    // constexpr size_t kBatchSize = 64;
    constexpr size_t kHeadNum   = 80;
    constexpr size_t KvHeadNum  = kHeadNum / 8;
    constexpr size_t kBatchSize = 256;
    constexpr size_t kInputLen  = 1;
    // constexpr size_t kSequenceLen = 63;
    // constexpr size_t kSequenceLen = 4095;
    // constexpr size_t kSequenceLen = 511;
    constexpr size_t kSequenceLen = 2047;
    // constexpr size_t kSequenceLen = 32767;
    // constexpr size_t kSequenceLen = 65535;
    // constexpr size_t kSequenceLen = 131071;
    // constexpr size_t kSequenceLen = 262143;
    // constexpr size_t kSequenceLen = (1 << 20) - 1;  // 1M
    // constexpr size_t kSequenceLen = (1 << 22) - 1;  // 4M
    // constexpr size_t kSequenceLen = (1 << 24) - 1;  // 16M
    // constexpr int kSequenceLen = 2047;
    constexpr int kBlockSz   = 128;
    constexpr int kMaxSplitK = 1;
#else

    // append
    // constexpr size_t kHeadNum     = 32;
    // constexpr size_t KvHeadNum    = kHeadNum;
    // constexpr size_t kBatchSize   = 1;
    // constexpr size_t kInputLen    = 128;
    // constexpr size_t kSequenceLen = 65536;
    // constexpr int    kMaxSplitK   = 128;

    // constexpr size_t kHeadNum     = 1;
    // constexpr size_t KvHeadNum    = kHeadNum;
    // constexpr size_t kBatchSize   = 1;
    // constexpr size_t kInputLen    = 64;
    // constexpr size_t kSequenceLen = 65536;
    // constexpr int    kMaxSplitK   = 1;

    // prefill
    constexpr size_t kHeadNum     = 32;
    constexpr size_t KvHeadNum    = kHeadNum;
    constexpr size_t kBatchSize   = 1;
    constexpr size_t kInputLen    = 16384;
    constexpr size_t kSequenceLen = 0;
    constexpr int    kMaxSplitK   = 1;

    constexpr int kBlockSz     = 128;

#endif

#if KV_INT8
    using Tkv                  = uint8_t;
    constexpr int kQuantPolicy = QuantPolicy::kCacheKVInt8;
#elif KV_INT4
    using Tkv                  = uint4_t;
    constexpr int kQuantPolicy = QuantPolicy::kCacheKVInt4;
#else
    using Tkv                  = T;
    constexpr int kQuantPolicy = 0;
#endif

    static_assert(KvHeadNum > 0);

    constexpr size_t kContextLen = kSequenceLen + kInputLen;
    constexpr size_t kTokenNum   = kBatchSize * kInputLen;
    constexpr int    kTestIter   = 10;

    constexpr float kRoPEBase = 10000.f;
    constexpr int   kDump     = 0;

    RNG rng{};

    thrust::universal_vector<T> k_cache(kBatchSize * KvHeadNum * kContextLen * kHeadDim);
    thrust::universal_vector<T> v_cache(kBatchSize * KvHeadNum * kContextLen * kHeadDim);

    thrust::universal_vector<T> kv_cache(KvHeadNum * 2 * kBatchSize * kContextLen * kHeadDim);

    thrust::universal_vector<T> qkv(kBatchSize * kInputLen * (kHeadNum + KvHeadNum * 2) * kHeadDim);
    thrust::universal_vector<T> output(kBatchSize * kInputLen * kHeadNum * kHeadDim);

    thrust::universal_vector<bool>  finished(kBatchSize);
    thrust::universal_vector<int>   sequence_length(kBatchSize);
    thrust::universal_vector<int>   input_length(kBatchSize);
    thrust::universal_vector<int>   context_length(kBatchSize);
    thrust::universal_vector<float> rope_base(kBatchSize);
    thrust::universal_vector<int>   cu_seqlens(kBatchSize + 1);
    thrust::universal_vector<int>   cu_kv_lens(kBatchSize + 1);

    thrust::universal_vector<float> partial_M(kTokenNum * kHeadNum * kMaxSplitK);
    thrust::universal_vector<float> partial_L(kTokenNum * kHeadNum * kMaxSplitK);
    thrust::universal_vector<float> partial_O(kTokenNum * kHeadNum * kMaxSplitK * kHeadDim);
    thrust::universal_vector<int>   split_cnt(kTokenNum);
    thrust::universal_vector<int>   semaphores(kTokenNum * kHeadNum * kMaxSplitK);

    thrust::universal_vector<T> kv_cache_quant_data(kBatchSize * KvHeadNum * 2 * kContextLen * 2);
    thrust::fill(kv_cache_quant_data.begin(), kv_cache_quant_data.end(), T{0.});

    thrust::universal_vector<float> qk_buf((size_t)kDump * kBatchSize * kHeadNum * kInputLen * kContextLen);
    thrust::universal_vector<T>     pr_buf((size_t)kDump * kBatchSize * kHeadNum * kInputLen * kContextLen);

    std::fill(semaphores.begin(), semaphores.end(), 0);

    rng.GenerateNormal(qkv.data().get(), qkv.size(), 1.f, 0.f);

    rng.GenerateNormal(k_cache.data().get(), kBatchSize * KvHeadNum * kContextLen * kHeadDim);
    rng.GenerateNormal(v_cache.data().get(), kBatchSize * KvHeadNum * kContextLen * kHeadDim);

    if (0) {
        // Set input range to zero
        // (BH, SD)
        cudaMemset2DAsync(k_cache.data().get() + kSequenceLen * kHeadDim,
                          sizeof(T) * kContextLen * kHeadDim,
                          0,
                          sizeof(T) * kInputLen * kHeadDim,
                          kBatchSize * KvHeadNum);
        cudaMemset2DAsync(v_cache.data().get() + kSequenceLen * kHeadDim,
                          sizeof(T) * kContextLen * kHeadDim,
                          0,
                          sizeof(T) * kInputLen * kHeadDim,
                          kBatchSize * KvHeadNum);
    }

    invokeApplyRotaryEmbedding(k_cache.data().get(), kContextLen, KvHeadNum, kHeadDim, kRoPEBase, kBatchSize);

    thrust::universal_vector<T> k_cache_ref = k_cache;
    thrust::universal_vector<T> v_cache_ref = v_cache;

    thrust::universal_vector<char>  blocks;
    thrust::universal_vector<char*> k_ptrs;
    thrust::universal_vector<int>   cu_block_cnts;

    TestBlocks<Tkv>(
        k_cache, v_cache, blocks, k_ptrs, cu_block_cnts, KvHeadNum, kHeadDim, kBlockSz, kBatchSize, kQuantPolicy);

    thrust::universal_vector<T>     output_ref = output;
    thrust::universal_vector<void*> k_cache_ref_ptrs(kBatchSize);
    thrust::universal_vector<void*> v_cache_ref_ptrs(kBatchSize);

    cudaDeviceSynchronize();

    for (size_t i = 0; i <= kBatchSize; ++i) {
        cu_seqlens[i] = i * kInputLen;
        cu_kv_lens[i] = i * kContextLen;
    }

    for (size_t i = 0; i < kBatchSize; ++i) {
        input_length[i]     = kInputLen;
        sequence_length[i]  = kSequenceLen;
        context_length[i]   = kContextLen;
        k_cache_ref_ptrs[i] = k_cache_ref.data().get() + i * k_cache_ref.size() / kBatchSize;
        v_cache_ref_ptrs[i] = v_cache_ref.data().get() + i * v_cache_ref.size() / kBatchSize;
        rope_base[i]        = kRoPEBase;
    }

    // getchar();

    params.out = output_ref.data().get();
    params.q   = qkv.data().get();
    params.k   = params.q + kHeadNum * kHeadDim;
    params.v   = params.k + KvHeadNum * kHeadDim;

    params.stride = (kHeadNum + 2 * KvHeadNum) * kHeadDim;

    params.token_num  = kTokenNum;
    params.batch_size = kBatchSize;
    params.max_q_len  = kInputLen;
    params.max_k_len  = kContextLen;

    params.block_iter_params = BlockIteratorParams{k_ptrs.data().get(),  //
                                                   cu_block_cnts.data().get(),
                                                   0,
                                                   kBlockSz};

    params.linear_iter_params = LinearIteratorParams{kv_cache.data().get(),  //
                                                     int(2 * kBatchSize * kContextLen * kHeadDim),
                                                     int(kBatchSize * kContextLen * kHeadDim)};

    params.quant_policy = kQuantPolicy;

    // std::copy_n(quant_params_kv.data(), 4, &params.kv_quant_params[0]);

    params.finished   = finished.data().get();
    params.rope_theta = rope_base.data().get();
    params.cu_q_len   = cu_seqlens.data().get();
    params.cu_k_len   = cu_kv_lens.data().get();

    params.num_heads     = kHeadNum;
    params.num_kv_heads  = KvHeadNum;
    params.size_per_head = kHeadDim;
    params.inv_sqrt_dh   = (float)std::log2(expf(1.)) / std::sqrt((float)params.size_per_head);

    params.rotary_embedding_dim  = kHeadDim;
    params.rotary_embedding_base = kRoPEBase;
    params.rope_ti_scale         = 1.;

    params.split_cnt = split_cnt.data().get();
    params.partial_L = partial_L.data().get();
    params.partial_M = partial_M.data().get();
    params.partial_O = partial_O.data().get();
    params.locks     = semaphores.data().get();

    params.max_split_k = kMaxSplitK;
    params.arch        = getSMVersion();

    params.qk = qk_buf.data().get();
    params.pr = pr_buf.data().get();

    Reference<T> reference(kDump ? Reference<T>::kUNFUSED : Reference<T>::kFLASH_ATTENTION, {});
    // Reference<T> reference(Reference<T>::kUNFUSED, {});
    reference.Reshape(kInputLen, kContextLen, kHeadNum, kHeadDim, KvHeadNum, kBatchSize);

    for (int i = 0; i < 1; ++i) {
        reference.Execute(params.out, k_cache_ref.data().get(), v_cache_ref.data().get(), qkv.data().get());
    }

    cudaDeviceSynchronize();

    if constexpr (kDump) {
        for (size_t b = 0; b < kBatchSize; ++b) {
            for (size_t h = 0; h < kHeadNum; ++h) {
                for (size_t q = 0; q < kInputLen; ++q) {
                    auto qk = reference.qk() + b * kHeadNum * kInputLen * kContextLen + h * kInputLen * kContextLen
                              + q * kContextLen;
                    for (size_t k = 0; k < kContextLen; ++k) {
                        std::cout << qk[k] * params.inv_sqrt_dh << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "\n";
        return -1;
    }
    std::cout << "---------------------------------------------------\n";

    params.out = output.data().get();

    std::vector<thrust::universal_vector<T>> outputs;

    for (int i = 0; i < std::max(kTestIter, 1); ++i) {

#if DECODING
        dispatchDecoding<T>(params);
#else
        // input -> blocked
        invokeProcessKV_v2_(params);
        // blocked -> linear
        invokeFlattenKV_v2_(params, cu_kv_lens[kBatchSize]);

        // auto tmp = std::exchange(params.linear_iter_params.kv_cache, nullptr);
        dispatchAttention(params);
        // params.linear_iter_params.kv_cache = std::exchange(tmp, nullptr);
#endif
        if (auto err = cudaGetLastError(); err != cudaSuccess) {
            std::cout << cudaGetErrorString(err) << "\n";
            return -1;
        }
        if (1) {
            outputs.push_back(output);
        }
    }

    if (kDump) {
        cudaDeviceSynchronize();
        for (size_t b = 0; b < kBatchSize; ++b) {
            for (size_t h = 0; h < kHeadNum; ++h) {
                for (size_t q = 0; q < kInputLen; ++q) {
                    auto ref = reference.qk() + b * kHeadNum * kInputLen * kContextLen + h * kInputLen * kContextLen
                               + q * kContextLen;
                    auto data = qk_buf.data().get() + b * kHeadNum * kInputLen * kContextLen
                                + h * kInputLen * kContextLen + q * kContextLen;
                    for (size_t k = 0; k < kContextLen; ++k) {
                        // std::cout << std::max(0.f, std::abs(data[k] - (float)ref[k]) - 1e-5f) << " ";
                        std::cout << data[k] * params.inv_sqrt_dh << " ";
                        // std::cout << (float)data[k] << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    invokeFlattenKV_v2(k_cache.data().get(),  // [B, H, S, D]
                       v_cache.data().get(),
                       k_ptrs.data().get(),
                       cu_kv_lens.data().get(),
                       cu_block_cnts.data().get(),
                       nullptr,  // DECODING ? nullptr : params.rope_theta,
                       1.,
                       KvHeadNum * kContextLen,
                       0,
                       kContextLen,
                       1,
                       kBlockSz,
                       0,
                       kContextLen,
                       KvHeadNum,
                       kHeadDim,
                       kBatchSize,
                       kQuantPolicy);
    cudaDeviceSynchronize();

    if (outputs.size() > 1) {
        std::cout << "Evaluating consistency..." << std::endl;
        for (size_t i = 1; i < outputs.size(); ++i) {
            Compare(outputs[i].data().get(), outputs[i - 1].data().get(), kHeadDim, kHeadDim, kHeadNum, 0, 0, 0);
        }
    }

    std::cout << "---------------------------------------------------\n";

    // [B, S, H, D]
    Compare(output.data().get(),  //
            output_ref.data().get(),
            kHeadNum * kHeadDim,
            kHeadNum * kHeadDim,
            kBatchSize * kInputLen,
            0);

    // [BH, SD]
    Compare(k_cache.data().get() + kSequenceLen * kHeadDim,
            k_cache_ref.data().get() + kSequenceLen * kHeadDim,
            kContextLen * kHeadDim,
            kInputLen * kHeadDim,
            kBatchSize * KvHeadNum,
            0);
    Compare(v_cache.data().get() + kSequenceLen * kHeadDim,
            v_cache_ref.data().get() + kSequenceLen * kHeadDim,
            kContextLen * kHeadDim,
            kInputLen * kHeadDim,
            kBatchSize * KvHeadNum);

    return 0;
}

int main(int argc, char* argv[])
{
    test_attention<half>();

    // test_attention<nv_bfloat16>();
}

// Copyright (c) OpenMMLab. All rights reserved.

#include "decoder_multihead_attention.h"
#include "kv_cache.h"
#include "test_utils.h"
#include <cmath>
#include <ios>
#include <iostream>
#include <thrust/universal_vector.h>

#include <iomanip>
#include <numeric>

using namespace turbomind;

template<typename T>
T* align(T* ptr, size_t alignment)
{
    size_t misalign = (uintptr_t)ptr % alignment;
    std::cout << "misalignment: " << misalign << "\n";
    if (misalign) {
        return (T*)((uint8_t*)ptr + alignment - misalign);
    }
    return ptr;
}

// [S/S, H, S, D] <-> [S/b, H, b, D]

void TestBlocks(thrust::universal_vector<half>&  linear,          // linear data
                thrust::universal_vector<half>&  _blocks,         // block data
                thrust::universal_vector<half*>& _ptrs,           // block ptrs
                thrust::universal_vector<int>&   _cu_block_cnts,  // cumulative block counts
                int                              head_num,
                int                              head_dim,
                int                              block_size,
                int                              batch_size)
{
    int seq_len  = linear.size() / (head_dim * head_num * batch_size);
    int n_blocks = (seq_len + block_size - 1) / block_size;

    std::cout << "batch_size = " << batch_size << ", seq_len = " << seq_len << ", block_num = " << n_blocks
              << ", block_size = " << block_size << "\n";

    thrust::universal_vector<half>  blocks(batch_size * n_blocks * head_num * block_size * head_dim);
    thrust::universal_vector<half*> ptrs(batch_size * n_blocks + 1);  // +1 padding

    std::vector<size_t> idxs(batch_size * n_blocks);
    std::iota(idxs.begin(), idxs.end(), 0);

    std::random_shuffle(idxs.begin(), idxs.end());

    for (int i = 0; i < idxs.size(); ++i) {
        ptrs[i] = blocks.data().get() + idxs[i] * head_num * block_size * head_dim;
    }

    thrust::universal_vector<int> seq_lens(batch_size);
    thrust::fill(seq_lens.begin(), seq_lens.end(), seq_len);

    std::vector<int>              n_blocks_vec(batch_size + 1, n_blocks);
    thrust::universal_vector<int> cu_block_cnts(batch_size + 1);
    std::exclusive_scan(n_blocks_vec.begin(), n_blocks_vec.end(), cu_block_cnts.begin(), 0);

    for (int i = 0; i < 10; ++i) {
        ConvertLinearToBlocks((const half*)linear.data().get(),
                              ptrs.data().get(),
                              cu_block_cnts.data().get(),
                              seq_lens.data().get(),
                              0,
                              seq_len,
                              block_size,
                              head_num,
                              head_dim,
                              batch_size,
                              0);
    }
    thrust::universal_vector<half> _linear(linear.size());

    for (int i = 0; i < 10; ++i) {
        ConvertBlocksToLinear((const half**)ptrs.data().get(),
                              _linear.data().get(),
                              cu_block_cnts.data().get(),
                              seq_lens.data().get(),
                              0,
                              block_size,
                              seq_len,
                              head_num,
                              head_dim,
                              batch_size,
                              0);
    }
    cudaDeviceSynchronize();

    if (0) {
        std::cout << ">>> Compare\n";
        Compare(_linear.data().get(), linear.data().get(), head_dim, head_dim, batch_size * head_num * seq_len);
        std::cout << "<<< Compare\n";
    }

    _blocks.swap(blocks);
    _ptrs.swap(ptrs);
    _cu_block_cnts.swap(cu_block_cnts);
}

int main(int argc, char* argv[])
{

    DecoderMultiHeadAttentionParams<half> params{};

    constexpr int kHeadNum    = 32;
    constexpr int kHeadDim    = 128;
    constexpr int KvHeadNum   = 32;
    constexpr int kBatchSize  = 1;
    constexpr int kContextLen = 1024;
    // constexpr int kContextLen  = 1024;
    constexpr int kSequenceLen = kContextLen + 1;
    constexpr int kBlockSz     = 128;
    constexpr int kTestIter    = 1;
    constexpr int kMaxSplitK   = 4;

    RNG rng{};

    thrust::universal_vector<half>  output(kBatchSize * kHeadNum * kHeadDim);
    thrust::universal_vector<half>  qkv(kBatchSize * (kHeadNum + KvHeadNum * 2) * kHeadDim);
    thrust::universal_vector<bool>  finished(kBatchSize);
    thrust::universal_vector<half>  k_cache(kBatchSize * kSequenceLen * KvHeadNum * kHeadDim);
    thrust::universal_vector<half>  v_cache(kBatchSize * kSequenceLen * KvHeadNum * kHeadDim);
    thrust::universal_vector<int>   sequence_lengths(kBatchSize);
    thrust::universal_vector<void*> k_cache_ptrs(kBatchSize);
    thrust::universal_vector<void*> v_cache_ptrs(kBatchSize);

    thrust::universal_vector<float> partial_M(kBatchSize * kHeadNum * kMaxSplitK);
    thrust::universal_vector<float> partial_L(kBatchSize * kHeadNum * kMaxSplitK);
    thrust::universal_vector<float> partial_O(kBatchSize * kHeadNum * kMaxSplitK * kHeadDim);

    rng.GenerateNormal(qkv.data().get(), qkv.size(), 1.f, 0.f);

    if (kContextLen) {
        rng.GenerateNormal(k_cache.data().get(), kBatchSize * KvHeadNum * kSequenceLen * kHeadDim);
        rng.GenerateNormal(v_cache.data().get(), kBatchSize * KvHeadNum * kSequenceLen * kHeadDim);

        cudaMemset2DAsync(k_cache.data().get() + kContextLen * kHeadDim,
                          sizeof(half) * kSequenceLen * kHeadDim,
                          0,
                          sizeof(half) * kHeadDim,
                          kBatchSize * KvHeadNum);
        if constexpr (0) {
            for (int b = 0; b < kBatchSize; ++b) {
                for (int h = 0; h < KvHeadNum; ++h) {
                    for (int s = 0; s < kSequenceLen; ++s) {
                        for (int d = 0; d < kHeadDim; ++d) {
                            std::cout << std::setw(7) << std::setprecision(4) << std::fixed
                                      << (float)k_cache[b * KvHeadNum * kSequenceLen * kHeadDim
                                                        + h * kSequenceLen * kHeadDim + s * kHeadDim + d]
                                      << " ";
                        }
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::exit(0);
        }

        cudaMemset2DAsync(v_cache.data().get() + kContextLen * kHeadDim,
                          sizeof(half) * kSequenceLen * kHeadDim,
                          0,
                          sizeof(half) * kHeadDim,
                          kBatchSize * KvHeadNum);
    }

    thrust::universal_vector<half>  k_blocks;
    thrust::universal_vector<half*> k_ptrs;
    thrust::universal_vector<int>   cu_block_cnts;

    TestBlocks(k_cache, k_blocks, k_ptrs, cu_block_cnts, KvHeadNum, kHeadDim, kBlockSz, kBatchSize);

    thrust::universal_vector<half>  v_blocks;
    thrust::universal_vector<half*> v_ptrs;

    TestBlocks(v_cache, v_blocks, v_ptrs, cu_block_cnts, KvHeadNum, kHeadDim, kBlockSz, kBatchSize);

    thrust::universal_vector<half>  k_cache_ref = k_cache;
    thrust::universal_vector<half>  v_cache_ref = v_cache;
    thrust::universal_vector<half>  output_ref  = output;
    thrust::universal_vector<void*> k_cache_ref_ptrs(kBatchSize);
    thrust::universal_vector<void*> v_cache_ref_ptrs(kBatchSize);

    cudaDeviceSynchronize();

    for (int i = 0; i < kBatchSize; ++i) {
        sequence_lengths[i] = kContextLen;
        k_cache_ptrs[i]     = k_cache.data().get() + i * k_cache.size() / kBatchSize;
        v_cache_ptrs[i]     = v_cache.data().get() + i * v_cache.size() / kBatchSize;
        k_cache_ref_ptrs[i] = k_cache_ref.data().get() + i * k_cache_ref.size() / kBatchSize;
        v_cache_ref_ptrs[i] = v_cache_ref.data().get() + i * v_cache_ref.size() / kBatchSize;

        // align(k_cache_ptrs[i], 256);
        // align(v_cache_ptrs[i], 256);
    }

    // getchar();

    params.out    = output_ref.data().get();
    params.q      = qkv.data().get();
    params.k      = params.q + kHeadNum * kHeadDim;
    params.v      = params.k + KvHeadNum * kHeadDim;
    params.stride = (kHeadNum + 2 * KvHeadNum) * kHeadDim;

    params.batch_size    = kBatchSize;
    params.max_seq_len   = kContextLen + 1;
    params.cu_block_cnts = cu_block_cnts.data().get();

    printf("%d %d\n", (int)k_ptrs.size(), (int)v_ptrs.size());
    params.k_cache_block_ptrs  = (void**)k_ptrs.data().get();
    params.v_cache_block_ptrs  = (void**)v_ptrs.data().get();
    params.kv_cache_block_size = kBlockSz;

    params.finished           = finished.data().get();
    params.per_sample_length  = sequence_lengths.data().get();
    params.per_sample_k_cache = k_cache_ref_ptrs.data().get();
    params.per_sample_v_cache = v_cache_ref_ptrs.data().get();
    params.layer_offset       = 0;

    params.num_heads     = kHeadNum;
    params.num_kv_heads  = KvHeadNum;
    params.size_per_head = kHeadDim;
    params.inv_sqrt_dh   = 1.f / std::sqrt((float)params.size_per_head);

    params.rotary_embedding_dim  = kHeadDim;
    params.rotary_embedding_base = 10000.f;

    params.partial_L = partial_L.data().get();
    params.partial_M = partial_M.data().get();
    params.partial_O = partial_O.data().get();

    for (int i = 0; i < kTestIter; ++i) {
        mmha_ft_reference(params, cudaStream_t{});
    }

    cudaDeviceSynchronize();
    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "\n";
        return -1;
    }
    std::cout << "---------------------------------------------------\n";

    params.out                = output.data().get();
    params.per_sample_k_cache = k_cache_ptrs.data().get();
    params.per_sample_v_cache = v_cache_ptrs.data().get();

    params.max_split_k = kMaxSplitK;
    params.max_seq_len = kContextLen;

    params.arch = 80;

    std::vector<thrust::universal_vector<half>> outputs;

    for (int i = 0; i < std::max(kTestIter, 10); ++i) {
        DispatchDecoderMultiheadAttention<half>(params);
        if (auto err = cudaGetLastError(); err != cudaSuccess) {
            std::cout << cudaGetErrorString(err) << "\n";
            return -1;
        }
        if (1) {
            outputs.push_back(output);
        }
    }

    thrust::universal_vector<int> seq_lens(kBatchSize);
    for (auto& x : seq_lens) {
        x = kContextLen + 1;
    }

    if (1) {
        ConvertBlocksToLinear((const half**)k_ptrs.data().get(),
                              k_cache.data().get(),
                              cu_block_cnts.data().get(),
                              seq_lens.data().get(),
                              0,
                              kBlockSz,
                              kSequenceLen,
                              KvHeadNum,
                              kHeadDim,
                              kBatchSize,
                              0);
        ConvertBlocksToLinear((const half**)v_ptrs.data().get(),
                              v_cache.data().get(),
                              cu_block_cnts.data().get(),
                              seq_lens.data().get(),
                              0,
                              kBlockSz,
                              kSequenceLen,
                              KvHeadNum,
                              kHeadDim,
                              kBatchSize,
                              0);
    }

    cudaDeviceSynchronize();

    if (outputs.size() > 1) {
        std::cout << "Evaluating consistency..." << std::endl;
        for (size_t i = 1; i < outputs.size(); ++i) {
            Compare(outputs[i].data().get(), outputs[0].data().get(), kHeadDim, kHeadDim, kHeadNum);
        }
    }

    std::cout << "---------------------------------------------------\n";

    Compare(output.data().get(), output_ref.data().get(), kHeadDim, kHeadDim, kHeadNum, false);

    // [H, S, D]

    Compare(k_cache.data().get() + kContextLen * kHeadDim,
            k_cache_ref.data().get() + kContextLen * kHeadDim,
            kSequenceLen * kHeadDim,
            kHeadDim,
            KvHeadNum);

    Compare(v_cache.data().get() + kContextLen * kHeadDim,
            v_cache_ref.data().get() + kContextLen * kHeadDim,
            kSequenceLen * kHeadDim,
            kHeadDim,
            KvHeadNum);

    return 0;
}
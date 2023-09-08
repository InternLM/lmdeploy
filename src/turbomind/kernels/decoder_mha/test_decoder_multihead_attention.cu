

#include "decoder_multihead_attention.h"
#include "kv_cache.h"
#include "test_utils.h"
#include <cmath>
#include <iostream>
#include <thrust/universal_vector.h>

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

void TestBlocks(thrust::universal_vector<half>& linear,
                thrust::universal_vector<half>& _blocks,
                thrust::universal_vector<half*> _ptrs,
                int                             head_num,
                int                             head_dim,
                int                             block_size)
{
    int seq_len  = linear.size() / head_num / head_dim;
    int n_blocks = (seq_len + block_size - 1) / block_size;

    std::cout << "seq_len = " << seq_len << ", block_num = " << n_blocks << ", block_size = " << block_size << "\n";

    thrust::universal_vector<half>  blocks(n_blocks * head_num * block_size * head_dim);
    thrust::universal_vector<half*> ptrs(n_blocks);

    std::vector<size_t> idxs(n_blocks);
    std::iota(idxs.begin(), idxs.end(), 0);

    std::random_shuffle(idxs.begin(), idxs.end());

    for (int i = 0; i < n_blocks; ++i) {
        ptrs[i] = blocks.data().get() + idxs[i] * head_num * block_size * head_dim;
    }

    for (int i = 0; i < 10; ++i) {
        ConvertLinearToBlocks(
            (const half*)linear.data().get(), ptrs.data().get(), block_size, head_num, head_dim, seq_len, 0);
    }
    thrust::universal_vector<half> _linear(linear.size());

    for (int i = 0; i < 10; ++i) {
        ConvertBlocksToLinear(
            (const half**)ptrs.data().get(), _linear.data().get(), block_size, head_num, head_dim, seq_len, 0);
    }
    cudaDeviceSynchronize();

    Compare(_linear.data().get(), linear.data().get(), head_dim, head_num * seq_len);
    exit(0);
}

int main(int argc, char* argv[])
{
    DecoderMultiHeadAttentionParams<half> params{};

    // constexpr int kHeadNum = 108 * 4;
    constexpr int kHeadNum    = 32;
    constexpr int kHeadDim    = 128;
    constexpr int kBatchSize  = 1;
    constexpr int kContextLen = 8192;
    constexpr int kTestIter   = 1;

    RNG rng{};

    thrust::universal_vector<half>  output(kBatchSize * kHeadNum * kHeadDim);
    thrust::universal_vector<half>  qkv(kBatchSize * kHeadNum * 3 * kHeadDim);
    thrust::universal_vector<bool>  finished(kBatchSize);
    thrust::universal_vector<half>  k_cache(kBatchSize * (kContextLen + 1) * kHeadNum * kHeadDim);
    thrust::universal_vector<half>  v_cache(kBatchSize * (kContextLen + 1) * kHeadNum * kHeadDim);
    thrust::universal_vector<int>   sequence_lengths(kBatchSize);
    thrust::universal_vector<void*> k_cache_ptrs(kBatchSize);
    thrust::universal_vector<void*> v_cache_ptrs(kBatchSize);

    rng.GenerateNormal(qkv.data().get(), qkv.size(), 1.f, 0.f);

    if (kContextLen) {
        rng.GenerateNormal(k_cache.data().get(), kContextLen * kHeadNum * kHeadDim);
        rng.GenerateNormal(v_cache.data().get(), kContextLen * kHeadNum * kHeadDim);
    }

    thrust::universal_vector<half>  k_blocks;
    thrust::universal_vector<half*> k_ptrs;

    TestBlocks(k_cache, k_blocks, k_ptrs, kHeadNum, kHeadDim, 128);

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

        align(k_cache_ptrs[i], 256);
        align(v_cache_ptrs[i], 256);
    }

    // getchar();

    params.out    = output_ref.data().get();
    params.q      = qkv.data().get();
    params.k      = params.q + kHeadNum * kHeadDim;
    params.v      = params.k + kHeadNum * kHeadDim;
    params.stride = 3 * kHeadNum * kHeadDim;

    params.batch_size   = kBatchSize;
    params.max_seq_len  = kContextLen + 1;
    params.max_timestep = kContextLen;

    params.finished           = finished.data().get();
    params.per_sample_length  = sequence_lengths.data().get();
    params.per_sample_k_cache = k_cache_ref_ptrs.data().get();
    params.per_sample_v_cache = v_cache_ref_ptrs.data().get();

    params.per_sample_kv_cache_offset = 0;

    params.num_heads     = kHeadNum;
    params.num_kv_heads  = kHeadNum;
    params.size_per_head = kHeadDim;
    params.inv_sqrt_dh   = 1.f / std::sqrt((float)params.size_per_head);

    params.rotary_embedding_dim  = kHeadDim;
    params.rotary_embedding_base = 10000.f;

    for (int i = 0; i < kTestIter; ++i) {
        mmha_ft_reference(params, cudaStream_t{});
    }

    cudaDeviceSynchronize();
    // if (auto err = cudaGetLastError(); err != cudaSuccess) {
    //     std::cout << cudaGetErrorString(err) << "\n";
    //     return -1;
    // }
    std::cout << "---------------------------------------------------\n";

    params.out                = output.data().get();
    params.per_sample_k_cache = k_cache_ptrs.data().get();
    params.per_sample_v_cache = v_cache_ptrs.data().get();

    std::vector<thrust::universal_vector<half>> outputs;

    for (int i = 0; i < std::max(kTestIter, 10); ++i) {
        LaunchDecoderMultiheadAttention<half, 128>(params);
        if (auto err = cudaGetLastError(); err != cudaSuccess) {
            std::cout << cudaGetErrorString(err) << "\n";
            return -1;
        }
        if (1) {
            outputs.push_back(output);
        }
    }

    cudaDeviceSynchronize();

    if (outputs.size() > 1) {
        std::cout << "Evaluating consistency..." << std::endl;
        for (size_t i = 1; i < outputs.size(); ++i) {
            Compare(outputs[i].data().get(), outputs[0].data().get(), kHeadDim, kHeadNum);
        }
    }

    std::cout << "---------------------------------------------------\n";

    Compare(output.data().get(), output_ref.data().get(), kHeadDim, kHeadNum, 0);

    Compare(v_cache.data().get() + (kContextLen - 0) * kHeadNum * kHeadDim,
            v_cache_ref.data().get() + (kContextLen - 0) * kHeadNum * kHeadDim,
            kHeadDim,
            kHeadNum);

    Compare(k_cache.data().get() + (kContextLen - 0) * kHeadNum * kHeadDim,
            k_cache_ref.data().get() + (kContextLen - 0) * kHeadNum * kHeadDim,
            kHeadDim,
            kHeadNum);

    return 0;
}
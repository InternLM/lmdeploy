/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
// modify from: https://github.com/Dao-AILab/flash-attention

#pragma once

#include "flash.h"
#include "flash_fwd_kernel.h"
#include "kernel_traits.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "static_switch.h"

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_N, bool Is_even_K, bool Return_softmax>
__global__ void flash_fwd_kernel(Flash_fwd_params params)
{
    flash::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_even_N, Is_even_K, Return_softmax>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3      grid(num_m_block, params.b, params.h);
    // We also use is_even_N to set Unpadded in the BlockInfo constructor, so we need to check
    // for cu_seqlens_q as well.
    const bool is_even_N = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr
                           && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    // const bool return_softmax = params.p_ptr != nullptr;
    constexpr bool return_softmax = false;
    BOOL_SWITCH(is_even_N, IsEvenNConst, [&] {
        BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                // Will only return softmax if dropout, to reduce compilation time.
                auto kernel = &flash_fwd_kernel < Kernel_traits, Is_dropout, Is_causal, IsEvenNConst, IsEvenKConst,
                     ReturnSoftmaxConst && Is_dropout > ;
                // auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout, Is_causal, IsEvenNConst, true,
                // ReturnSoftmaxConst && Is_dropout>;
                if (smem_size >= 48 * 1024) {
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
                }
                int ctas_per_sm;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            });
        });
    });
}

template<typename T>
void run_mha_fwd_hdim32(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr int         Headdim    = 32;
    static constexpr bool Is_dropout = false;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params,
                                                                                                             stream);
    });
}

template<typename T>
void run_mha_fwd_hdim64(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr int         Headdim    = 64;
    static constexpr bool Is_dropout = false;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        if constexpr (!Is_dropout) {
            // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
            // Using block size (64 x 256) is 27% slower for seqlen=2k
            // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(
                params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout,
            // Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>,
            // Is_dropout, Is_causal>(params, stream);
        }
        else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params,
                                                                                                                stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params,
            // stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout,
            // Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>,
            // Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T>
void run_mha_fwd_hdim128(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr int         Headdim    = 128;
    bool                  is_sm8x    = (turbomind::getSMVersion() >= 80);
    static constexpr bool Is_dropout = false;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        if constexpr (!Is_dropout) {
            // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
            // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
            if (is_sm8x) {
                if constexpr (!Is_causal) {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(
                        params, stream);
                }
                else {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(
                        params, stream);
                }
            }
            else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(
                    params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout,
            // Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>,
            // Is_dropout, Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false,
            // false, T>, Is_dropout, Is_causal>(params, stream); Using 8 warps (128 x 128 and 256 x 64) is 28% slower
            // for seqlen=2k run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, false, T>, Is_dropout,
            // Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>,
            // Is_dropout, Is_causal>(params, stream); 1st ones are good for H100, A100 2nd one is good for A6000 bc we
            // get slightly better occupancy
        }
        else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params,
                                                                                                                stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout,
            // Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, false, T>,
            // Is_dropout, Is_causal>(params, stream); run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true,
            // true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T>
void run_mha_fwd_hdim256(Flash_fwd_params& params, cudaStream_t stream)
{
    constexpr int Headdim = 256;
    int           device;
    cudaGetDevice(&device);
    int max_smem_per_sm, max_smem_per_block;
    cudaDeviceGetAttribute(&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // printf("max_smem_per_sm = %d, max_smem_per_block = %d\n", max_smem_per_sm, max_smem_per_block);

    static constexpr bool Is_dropout = false;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // For A100, we want to run with 128 x 64 (128KB smem).
        // For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM.
        if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) && max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params,
                                                                                                                stream);
        }
        else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params,
                                                                                                               stream);
        }
        // 64 KB
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params,
        // stream); 96 KB run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout,
        // Is_causal>(params, stream);
    });
}

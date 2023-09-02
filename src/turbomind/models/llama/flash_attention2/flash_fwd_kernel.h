/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
// modify from: https://github.com/Dao-AILab/flash-attention

#pragma once

#include <cmath>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "softmax.h"
#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Is_first, bool Check_inf = false, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void
softmax_rescale_o(Tensor0& scores, Tensor1& scores_max, Tensor1& scores_sum, Tensor2& acc_o, float softmax_scale_log2)
{
    if (Is_first) {
        flash::template reduce_max</*zero_init=*/true>(scores, scores_max);
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        flash::reduce_sum(scores, scores_sum);
    }
    else {
        Tensor scores_max_prev = make_fragment_like(scores_max);
        copy(scores_max, scores_max_prev);
        flash::template reduce_max</*zero_init=*/false>(scores, scores_max);
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) {
            float scores_max_cur = !Check_inf ? scores_max(mi) : (scores_max(mi) == -INFINITY ? 0.0f : scores_max(mi));
            float scores_scale   = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            scores_sum(mi) *= scores_scale;
#pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= scores_scale;
            }
        }
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        Tensor scores_sum_cur = make_fragment_like(scores_sum);
        flash::reduce_sum(scores, scores_sum_cur);
#pragma unroll
        for (int mi = 0; mi < size(scores_sum); ++mi) {
            scores_sum(mi) += scores_sum_cur(mi);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename TiledCopy>
inline __device__ void
write_softmax_to_gmem(Tensor<Engine0, Layout0> const& tOrP, Tensor<Engine1, Layout1>& tPgP, TiledCopy gmem_thr_copy_P)
{
#if (__CUDA_ARCH__ >= 800)
    // Reshape tOrP from (8, MMA_M, MMA_N) to (8, MMA_M * MMA_N)
    Layout l    = tOrP.layout();
    Tensor tPrP = make_tensor(tOrP.data(), make_layout(get<0>(l), make_layout(get<1>(l), get<2>(l))));
    CUTE_STATIC_ASSERT_V(size<2>(tPgP) == _1{});
    CUTE_STATIC_ASSERT_V(size<1>(tPrP) == size<1>(tPgP));
#pragma unroll
    for (int mi = 0; mi < size<1>(tPrP); ++mi) {
        copy(gmem_thr_copy_P, tPrP(_, mi), tPgP(_, mi, 0));
    }
#else
    // do not support return softmax on device < sm80
    assert(false);
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits,
         bool Is_dropout,
         bool Is_causal,
         bool Is_even_N,
         bool Is_even_K,
         bool Return_softmax,
         typename Params>
inline __device__ void compute_attn_1rowblock(const Params& params, const int bidb, const int bidh, const int m_block)
{

    using Element      = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t      = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM  = Kernel_traits::kBlockM;
    constexpr int kBlockN  = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps  = Kernel_traits::kNWarps;
    constexpr int MMA_M    = kBlockM / decltype(size<0>(typename Kernel_traits::TiledMma::TiledShape_MNK{}))::value;

    const BlockInfo</*Varlen=*/!Is_even_N> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0)
        return;

    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal) {
        int seq_diff = max(binfo.actual_seqlen_k - binfo.actual_seqlen_q, int(0));
        n_block_max  = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + seq_diff, kBlockN));
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
                           + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    if (!params.q_enable_seqlen) {
        row_offset_q =
            bidb * params.q_batch_stride + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    }

    // We move K and V to the last block.
    auto k_ptr          = params.k_ptr;
    auto k_batch_stride = params.k_batch_stride;
    if (params.k_batched_ptr != nullptr) {
        k_ptr          = (reinterpret_cast<Element**>(params.k_batched_ptr))[bidb] + params.k_batched_offset;
        k_batch_stride = 0;
    }
    const index_t row_offset_k = binfo.k_offset(k_batch_stride, params.k_row_stride, bidb)
                                 + (n_block_max - 1) * kBlockN * params.k_row_stride
                                 + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    auto v_ptr          = params.v_ptr;
    auto v_batch_stride = params.v_batch_stride;
    if (params.v_batched_ptr != nullptr) {
        v_ptr          = (reinterpret_cast<Element**>(params.v_batched_ptr))[bidb] + params.v_batched_offset;
        v_batch_stride = 0;
    }
    const index_t row_offset_v = binfo.k_offset(v_batch_stride, params.v_row_stride, bidb)
                                 + (n_block_max - 1) * kBlockN * params.v_row_stride
                                 + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p =
        ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) * params.seqlen_k_rounded
        + (n_block_max - 1) * kBlockN;

    // gmem tensor
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    // smem tensor
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)), typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK =
        make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV           = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt          = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // g->s thread copy
    auto gmem_tiled_copy_QKV = typename Kernel_traits::GmemTiledCopyQKV{};
    auto gmem_thr_copy_QKV   = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    auto gmem_tiled_copy_P   = typename Kernel_traits::GmemTiledCopyP{};
    auto gmem_thr_copy_P     = gmem_tiled_copy_P.get_thread_slice(tidx);

    // tiles of g->s copy
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tPgP = gmem_thr_copy_P.partition_D(gP);

    // tiles of mma
    typename Kernel_traits::TiledMma tiled_mma;
    auto                             thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor                           tSrQ    = thr_mma.partition_fragment_A(sQ);            // (MMA,MMA_M,MMA_K)
    Tensor                           tSrK    = thr_mma.partition_fragment_B(sK);            // (MMA,MMA_N,MMA_K)
    Tensor                           tOrVt   = thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto   smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto   smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ              = smem_thr_copy_Q.partition_S(sQ);

    auto   smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto   smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK              = smem_thr_copy_K.partition_S(sK);

    auto   smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto   smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt             = smem_thr_copy_V.partition_S(sVt);

    // TODO: this might need to change if we change the mma instruction in SM70
    Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(acc_o)>>{});
    Tensor scores_sum = make_fragment_like(scores_max);

    //
    // PREDICATES
    //

    Tensor cQ  = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ   = gmem_thr_copy_QKV.partition_S(cQ);   // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ   = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) {
            tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
        }
#pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
        }
    }

    // Prologue

    Tensor tQrQ = make_fragment_like(tQgQ);
    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy</*Is_even_MN=*/false, Is_even_K>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    if (Kernel_traits::Is_Q_in_regs) {
        cute::cp_async_fence();
    }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));  // M
        copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    // copy K
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy<Is_even_N, Is_even_K>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));  // M
        copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    constexpr int n_masking_steps = Is_causal ? cute::ceil_div(kBlockM, kBlockN) + 1 : 2;
#pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        }
        else {
            // Clear the smem tiles to account for predicated off loads
            flash::copy<Is_even_N, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(acc_s,
                                                               tSrQ,
                                                               tSrK,
                                                               tSsQ,
                                                               tSsK,
                                                               tiled_mma,
                                                               smem_tiled_copy_Q,
                                                               smem_tiled_copy_K,
                                                               smem_thr_copy_Q,
                                                               smem_thr_copy_K);

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        // We don't put the masking before the matmul S = Q K^T because we don't clear sK
        // for rows outside actual_seqlen_k. So those rows could have Inf / NaN, and the matmul
        // can produce Inf / NaN.
        if (!Is_causal) {
            if (!Is_even_N) {
                flash::apply_mask(scores, binfo.actual_seqlen_k - n_block * kBlockN);
            }
        }
        else {
            flash::apply_mask_causal(scores,
                                     n_block * kBlockN,
                                     binfo.actual_seqlen_k,
                                     m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4
                                         + max(binfo.actual_seqlen_k - binfo.actual_seqlen_q, 0),
                                     kNWarps * 16);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > 0) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // softmax
        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0 ? softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal>(
            scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2) :
                            softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal>(
                                scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        // Convert scores from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(scores);
        // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        if (Return_softmax) {
            Tensor tOrP_copy = make_fragment_like(tOrP);
            copy(tOrP, tOrP_copy);
            flash::write_softmax_to_gmem(tOrP_copy, tPgP, gmem_tiled_copy_P);
            tPgP.data() = tPgP.data() + (-kBlockN);
        }

        flash::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= 0; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // Advance gV
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(acc_s,
                                                               tSrQ,
                                                               tSrK,
                                                               tSsQ,
                                                               tSsK,
                                                               tiled_mma,
                                                               smem_tiled_copy_Q,
                                                               smem_tiled_copy_K,
                                                               smem_thr_copy_Q,
                                                               smem_thr_copy_K);

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > 0) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        softmax_rescale_o</*Is_first=*/false>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(scores);
        // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        if (Return_softmax) {
            Tensor tOrP_copy = make_fragment_like(tOrP);
            copy(tOrP, tOrP_copy);
            flash::write_softmax_to_gmem(tOrP_copy, tPgP, gmem_tiled_copy_P);
            tPgP.data() = tPgP.data() + (-kBlockN);
        }
        flash::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));

#pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum     = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        float scale   = inv_sum;
#pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= scale;
        }
    }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto   smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto   smem_thr_copy_O   = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO           = smem_thr_copy_O.retile_S(rO);     // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO           = smem_thr_copy_O.partition_D(sO);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
    }

    copy(smem_tiled_copy_O, taccOrO, taccOsO);
    __syncthreads();

    index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
                           + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    if (!params.o_enable_seqlen) {
        row_offset_o =
            bidb * params.o_batch_stride + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    }
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));

    auto   gmem_tiled_copy_O = typename Kernel_traits::GmemTiledCopyO{};
    auto   gmem_thr_copy_O   = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO              = gmem_thr_copy_O.partition_S(sO);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO              = gmem_thr_copy_O.partition_D(gO);

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    copy(gmem_tiled_copy_O, tOsO, tOrO);

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
        }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits,
         bool Is_dropout,
         bool Is_causal,
         bool Is_even_N,
         bool Is_even_K,
         bool Return_softmax,
         typename Params>
inline __device__ void compute_attn(const Params& params)
{
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    flash::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_even_N, Is_even_K, Return_softmax>(
        params, bidb, bidh, m_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash

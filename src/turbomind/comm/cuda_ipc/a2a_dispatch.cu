// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/semaphore.cuh"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/kernels/gemm/moe_utils_v2.h"

namespace turbomind::comm {

__global__ void AllToAllNotifyDispatch_Simple_Push(Array<int*, kMaxRanks> symm_meta,  //
                                                   int*                   num_input_hidden,
                                                   int*                   num_output_hidden,
                                                   SystemSemaphoreInfo*   semaphores,
                                                   int*                   token_idx_in_rank,
                                                   int                    rank,
                                                   int                    ranks,
                                                   int                    token_num)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(true);
    sem.Wait(true);

    __syncthreads();

    // use two warps to get token num and expert num for each rank
    const int offset_r   = warp_id == 0 ? 0 : 1;
    const int offset_w   = warp_id == 0 ? 0 : ranks;
    auto      num_hidden = warp_id == 0 ? num_input_hidden : num_output_hidden;

    if (lane_id < ranks) {
        int num = token_idx_in_rank[lane_id * (token_num + 2) + token_num + offset_r];
        for (int i = 0; i < ranks; ++i) {
            auto chn                                 = cvta_generic_to_global(symm_meta[i]);
            chn[(offset_w + rank) * ranks + lane_id] = num;
        }
    }

    __syncthreads();

    sem.Signal(false);
    sem.Wait(false);

    __syncthreads();

    if (lane_id < ranks) {
        auto chn = cvta_generic_to_global(symm_meta[rank]);
        for (int i = 1; i < ranks; ++i) {
            chn[(offset_w + i) * ranks + lane_id] += chn[(offset_w + i - 1) * ranks + lane_id];
        }
        if (lane_id == rank) {
            *num_hidden = chn[(offset_w + ranks - 1) * ranks + lane_id];
        }
    }

    sem.Signal(true);
    sem.Wait(true);
    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

void CudaIpcCommImpl::AllToAllNotifyDispatch(int*         symm_meta,
                                             int*         num_input_hidden,
                                             int*         num_output_hidden,
                                             int*         token_idx_in_rank,
                                             int          token_num,
                                             int          group,
                                             cudaStream_t stream)
{
    const int n_ranks = this->n_ranks(group);
    const int rank    = this->rank(group);

    auto semaphore     = groups_.at(group).semaphore.handle();
    auto symm_meta_ptr = get_symmetric_v2(symm_meta, group);

    AllToAllNotifyDispatch_Simple_Push<<<1, WARP_SIZE * 2, 0, stream>>>(  //
        symm_meta_ptr.uc,
        num_input_hidden,
        num_output_hidden,
        semaphore,
        token_idx_in_rank,
        rank,
        n_ranks,
        token_num);

    sync_check_cuda_error();
}

template<class T, int vec_size>
__global__ void __launch_bounds__(1024, 1)  //
    AllToAllDispatch_Simple_Push(Array<T*, kMaxRanks>      symm_hidden,
                                 Array<float*, kMaxRanks>  symm_scales,
                                 Array<int8_t*, kMaxRanks> symm_masks,
                                 SystemSemaphoreInfo*      semaphores,
                                 int*                      meta,
                                 T*                        hidden,
                                 int                       hidden_load_iters,
                                 float*                    topk_scales,
                                 int*                      topk_experts,
                                 int*                      token_idx_in_rank,
                                 int                       rank,
                                 int                       ranks,
                                 int                       token_num,
                                 int                       dim,
                                 int                       expert_per_token,
                                 constant<vec_size>)
{
    const int bi = blockIdx.x;

    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);
    sem.Signal(true);
    sem.Wait(true);

    __shared__ int rank_send_offset[kMaxRanks];   // token send offset for each rank
    __shared__ int rank_token_count[kMaxRanks];   // token count for each rank
    __shared__ int rank_token_padded[kMaxRanks];  // padded token count

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id == 0 && lane_id < ranks) {
        rank_send_offset[lane_id] = (rank == 0) ? 0 : __ldg(&meta[(rank - 1) * ranks + lane_id]);
    }
    if (warp_id == 1 && lane_id < ranks) {
        rank_token_count[lane_id]  = __ldg(&meta[(ranks - 1) * ranks + lane_id]);
        rank_token_padded[lane_id] = round_up(rank_token_count[lane_id], kMoeGateVecSize);
    }

    const int num_threads_per_rank = blockDim.x / ranks;
    const int dst_rank             = threadIdx.x / num_threads_per_rank;
    const int num_warps_per_rank   = num_threads_per_rank / WARP_SIZE;
    const int send_warp_id_in_rank = threadIdx.x / WARP_SIZE % num_warps_per_rank;

    const int tok_per_cta     = cdiv(token_num, (int)gridDim.x);
    const int token_start_idx = bi * tok_per_cta;
    const int token_end_idx   = min(token_start_idx + tok_per_cta, token_num);

    __syncthreads();

    for (int token_idx = token_start_idx + send_warp_id_in_rank; token_idx < token_end_idx;
         token_idx += num_warps_per_rank) {
        int send_idx = token_idx_in_rank[dst_rank * (token_num + 2) + token_idx];
        int dst_idx  = rank_send_offset[dst_rank] + send_idx;
        if (send_idx >= 0) {
            // hidden
            using Vec = Array<T, vec_size>;
            T* src    = (T*)(hidden + token_idx * dim);
            T* dst    = (T*)(symm_hidden[dst_rank] + dst_idx * dim);
            for (int i = lane_id; i < hidden_load_iters; i += WARP_SIZE) {
                Vec tmp;
                Ldg(tmp, src + i * vec_size);
                Stcg(dst + i * vec_size, tmp);
            }
            // scales and masks
            if (lane_id < expert_per_token) {
                const int index = token_idx * ranks * expert_per_token + dst_rank * expert_per_token + lane_id;

                float s = __ldg(&topk_scales[index]);
                int   e = __ldg(&topk_experts[index]);

                auto scales_chn = cvta_generic_to_global(symm_scales[dst_rank]);
                auto masks_chn  = cvta_generic_to_global(symm_masks[dst_rank]);

                scales_chn[lane_id * rank_token_count[dst_rank] + dst_idx] = s;
                if (e >= 0) {
                    masks_chn[e * rank_token_padded[dst_rank] + dst_idx] = lane_id;
                }
            }
        }
    }

    __syncthreads();

    sem.Signal(true);
    sem.Wait(true);
    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

void CudaIpcCommImpl::AllToAllDispatch(void*        symm_hidden,
                                       float*       symm_scales,
                                       int8_t*      symm_masks,
                                       int*         meta,
                                       void*        hidden,
                                       float*       topk_scales,
                                       int*         topk_experts,
                                       int*         token_idx_in_rank,
                                       int          token_num,
                                       int          dim,
                                       int          expert_per_token,
                                       DataType     type,
                                       int          group,
                                       cudaStream_t stream)
{
    const int n_ranks = this->n_ranks(group);
    const int rank    = this->rank(group);

    auto semaphore = groups_.at(group).semaphore.handle();

    auto invoke = [&](auto t) {
        using T              = decltype(t);
        auto symm_hidden_ptr = get_symmetric_v2((T*)symm_hidden, group);
        auto symm_scales_ptr = get_symmetric_v2(symm_scales, group);
        auto symm_masks_ptr  = get_symmetric_v2(symm_masks, group);

        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        constexpr int threads  = 1024;
        const int     max_ctas = max_ctas_.apply(24);

        TM_CHECK(dim % vec_size == 0);

        AllToAllDispatch_Simple_Push<<<max_ctas, threads, 0, stream>>>(  //
            symm_hidden_ptr.uc,
            symm_scales_ptr.uc,
            symm_masks_ptr.uc,
            semaphore,
            meta,
            (T*)hidden,
            dim / vec_size,
            topk_scales,
            topk_experts,
            token_idx_in_rank,
            rank,
            n_ranks,
            token_num,
            dim,
            expert_per_token,
            constant<vec_size>{});

        sync_check_cuda_error();
    };

    TM_DISPATCH_PRIMARY_DTYPES(type, invoke);
}

}  // namespace turbomind::comm

// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/cuda_ipc/common.h"

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/group_sum.h"
#include "src/turbomind/comm/cuda_ipc/semaphore.cuh"

#include "src/turbomind/comm/cuda_ipc/multimem.cuh"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

template<class T, int vec_size, int block_dim, int groups, class Relaxed>
__global__ void AllreduceResidualBiasRMSnormV_Simple_Pull(T*                     buf,
                                                          T*                     res,
                                                          const T*               bias,
                                                          const T*               weights,
                                                          Array<T*, kMaxRanks>   rs_buf,
                                                          Array<T*, kMaxRanks>   ag_buf,
                                                          SystemSemaphoreInfo*   g_semaphores,
                                                          int                    rs_rank,
                                                          int                    ag_rank,
                                                          int                    rs_ranks,
                                                          int                    ag_ranks,
                                                          int                    g_rank,
                                                          int                    g_ranks,
                                                          int                    offset,
                                                          int                    first,
                                                          int                    last,
                                                          Array<int2, kMaxRanks> ag_ranges,
                                                          int                    vdim,
                                                          float                  inv_dim,
                                                          float                  eps,
                                                          constant<vec_size>,
                                                          constant<block_dim>,
                                                          constant<groups>,
                                                          Relaxed relaxed)
{
    SystemSemaphore sem(g_semaphores, g_ranks, blockIdx.x, threadIdx.x);

    sem.Signal(relaxed);

    using Vec = Array<T, vec_size>;

    using namespace ops;

    static_assert(block_dim % groups == 0);
    constexpr int threads = block_dim / groups;

    static_assert(threads % WARP_SIZE == 0);
    constexpr int warps = threads / WARP_SIZE;

    const int xi = threadIdx.x / threads;
    const int di = threadIdx.x % threads;

    Vec b_vec{};
    if (bias && di < vdim) {
        Ldg(b_vec, bias + di * vec_size);
    }

    Vec w_vec;
    if (di < vdim) {
        Ldg(w_vec, weights + di * vec_size);
    }

    sem.Wait(relaxed);

    __syncthreads();

    const int bi = blockIdx.x * groups + xi;
    const int bn = gridDim.x * groups;

    for (int i = 1; i < rs_ranks - 1; ++i) {
        const int  p   = rs_rank + i < rs_ranks ? rs_rank + i : rs_rank + i - rs_ranks;
        const auto src = cvta_generic_to_global(rs_buf[p]);
        Vec        acc, tmp;
        for (int ti = offset + first + bi; ti < offset + last; ti += bn) {
            const int idx = (ti * vdim + di) * vec_size;
            if (di < vdim) {
                Load(tmp, src + idx);
                Load(acc, buf + idx);
                acc = acc + tmp;
                Store(buf + idx, acc);
            }
        }
    }

    auto syncgroup = [&] {  //
        asm volatile("bar.sync %0, %1;" : : "r"(15 - xi), "r"(threads) : "memory");
    };

    {
        const T* chn{};
        if (rs_ranks > 1) {
            const int p = rs_rank > 0 ? rs_rank - 1 : rs_ranks - 1;  // last peer
            chn         = cvta_generic_to_global(rs_buf[p]);
        }
        for (int ti = first + bi; ti < last; ti += bn) {
            const int idx = ((offset + ti) * vdim + di) * vec_size;
            Vec       acc, tmp;
            Vec       r_vec{};
            float     sum{};
            if (di < vdim) {
                if (chn) {
                    Load(tmp, chn + idx);
                }
                Load(acc, buf + idx);
                if (chn) {
                    acc = acc + tmp;
                }
                Load(r_vec, res + (ti * vdim + di) * vec_size);
                r_vec = r_vec + acc;
                if (bias) {
                    r_vec = r_vec + b_vec;
                }
                Store(res + (ti * vdim + di) * vec_size, r_vec);
                PRAGMA_UNROLL
                for (int i = 0; i < vec_size; ++i) {
                    sum += (float)r_vec[i] * (float)r_vec[i];
                }
            }
            sum = detail::GroupSum(sum, warps, syncgroup);
            __shared__ float shared_sum[groups];
            if (di == 0) {
                shared_sum[xi] = rsqrtf(sum * inv_dim + eps);
            }
            syncgroup();
            sum = shared_sum[xi];
            if (di < vdim) {
                PRAGMA_UNROLL
                for (int i = 0; i < vec_size; ++i) {
                    r_vec[i] = static_cast<T>(((float)r_vec[i] * sum)) * w_vec[i];
                }
                Store(buf + idx, r_vec);
            }
        }
    }

    __syncthreads();

    sem.Signal(relaxed);
    sem.Wait(relaxed);

    __syncthreads();

#if 1
    for (int i = 1; i < ag_ranks; ++i) {
        const int p   = ag_rank + i < ag_ranks ? ag_rank + i : ag_rank + i - ag_ranks;
        auto      dst = cvta_generic_to_global(ag_buf[p]);
        for (int ti = offset + first + bi; ti < offset + last; ti += bn) {
            const int idx = (ti * vdim + di) * vec_size;
            if (di < vdim) {
                Vec vec;
                Load(vec, buf + idx);
                Store(dst + idx, vec);
            }
        }
    }
#else
    for (int i = 1; i < ag_ranks; ++i) {
        const int p              = ag_rank + i < ag_ranks ? ag_rank + i : ag_rank + i - ag_ranks;
        const auto [first, last] = ag_ranges[p];
        auto src                 = cvta_generic_to_global(ag_buf[p]);
        for (int ti = first + bi; ti < last; ti += bn) {
            const int idx = (ti * vdim + di) * vec_size;
            if (di < vdim) {
                Vec vec;
                Load(vec, src + idx);
                Store(buf + idx, vec);
            }
        }
    }
#endif

    __syncthreads();

    sem.Signal(true);
    sem.Wait(true);

    sem.Update(g_semaphores, g_ranks, blockIdx.x, threadIdx.x);
}

template<class T, int vec_size, int block_dim, int groups, class Relaxed>
__global__ void AllreduceResidualBiasRMSnormV_NVLS(T*                   rs_mc_buf,
                                                   T*                   ag_mc_buf,
                                                   T*                   res,
                                                   const T*             bias,
                                                   const T*             weights,
                                                   SystemSemaphoreInfo* semaphores,
                                                   int                  g_rank,
                                                   int                  g_ranks,
                                                   int                  first,
                                                   int                  last,
                                                   int                  offset,
                                                   int                  vdim,
                                                   float                inv_dim,
                                                   float                eps,
                                                   constant<vec_size>,
                                                   constant<block_dim>,
                                                   constant<groups>,
                                                   Relaxed relaxed)
{

#if TURBOMIND_ARCH_SM90

    SystemSemaphore sem(semaphores, g_ranks, blockIdx.x, threadIdx.x);

    sem.Signal(relaxed);

    using Vec = Array<T, vec_size>;

    using namespace ops;

    static_assert(block_dim % groups == 0);
    constexpr int threads = block_dim / groups;

    static_assert(threads % WARP_SIZE == 0);
    constexpr int warps = threads / WARP_SIZE;

    const int xi = threadIdx.x / threads;
    const int di = threadIdx.x % threads;

    using Vec = Array<T, vec_size>;

    Vec b_vec{};
    if (bias && di < vdim) {
        Ldg(b_vec, bias + di * vec_size);
    }

    Vec w_vec;
    if (di < vdim) {
        Ldg(w_vec, weights + di * vec_size);
    }

    sem.Wait(relaxed);

    __syncthreads();

    const int bi = blockIdx.x * groups + xi;
    const int bn = gridDim.x * groups;

    auto syncgroup = [&] {  //
        asm volatile("bar.sync %0, %1;" : : "r"(15 - xi), "r"(threads) : "memory");
    };

    for (int ti = first + bi; ti < last; ti += bn) {
        const int idx = ((offset + ti) * vdim + di) * vec_size;
        float     sum{};
        Vec       vec;
        if (di < vdim) {
            Vec acc = multimem_ld_reduce_sum((const Vec*)(rs_mc_buf + idx));
            Load(vec, res + (ti * vdim + di) * vec_size);
            vec = vec + acc;
            if (bias) {
                vec = vec + b_vec;
            }
            Store(res + (ti * vdim + di) * vec_size, vec);
            PRAGMA_UNROLL
            for (int i = 0; i < vec_size; ++i) {
                sum += (float)vec[i] * (float)vec[i];
            }
        }
        sum = detail::GroupSum(sum, warps, syncgroup);
        __shared__ float shared_sum[groups];
        if (di == 0) {
            shared_sum[xi] = rsqrtf(sum * inv_dim + eps);
        }
        syncgroup();
        sum = shared_sum[xi];
        if (di < vdim) {
            PRAGMA_UNROLL
            for (int i = 0; i < vec_size; ++i) {
                vec[i] = static_cast<T>(((float)vec[i] * sum)) * w_vec[i];
            }
            multimem_st(ag_mc_buf + idx, vec);
        }
    }

    __syncthreads();

    sem.Signal(true);
    sem.Wait(true);

    sem.Update(semaphores, g_ranks, blockIdx.x, threadIdx.x);
#endif
}

void CudaIpcCommImpl::AllreduceResidualBiasRMSnormEx(void*        hidden,
                                                     void*        residual,
                                                     const void*  bias,
                                                     const void*  weights,
                                                     float        eps,
                                                     int          dim,
                                                     DataType     dtype,
                                                     int          group0,
                                                     int          group1,
                                                     const int*   local_token_nums,
                                                     cudaStream_t stream)
{
    FT_CHECK(group0 * group1 == 0);

    const auto& g0 = groups_.at(group0);
    const auto& g1 = groups_.at(group1);

    const int tp0 = n_ranks(group0);
    const int tp1 = n_ranks(group1);

    const int inner_tp = std::min(tp0, tp1);

    FT_CHECK(tp0 % inner_tp == 0 && tp1 % inner_tp == 0);

    Array<int, kMaxRanks> offsets{};
    Array<int, kMaxRanks> firsts{};
    Array<int, kMaxRanks> lasts{};

    for (int i = 0, offset = 0; i < global_n_ranks_; ++i) {
        const int num   = local_token_nums[i / inner_tp];
        const int slice = (num + inner_tp - 1) / inner_tp;
        const int first = std::min(num, i % inner_tp * slice);
        const int last  = std::min(num, first + slice);

        std::tie(offsets[i], firsts[i], lasts[i]) = std::tie(offset, first, last);

        if ((i + 1) % inner_tp == 0) {
            offset += num;
        }
    }
    const int g_rank = rank(0);

    const int first  = firsts[g_rank];
    const int last   = lasts[g_rank];
    const int offset = offsets[g_rank];

    auto semaphore = groups_.at(0).semaphore.handle();

    auto invoke = [&](auto t, auto groups) {
        using T                = decltype(t);
        constexpr int vec_size = sizeof(uint4) / sizeof(T);

        auto rs_symm_ptr = get_symmetric_v2((T*)hidden, group0);
        auto ag_symm_ptr = get_symmetric_v2((T*)hidden, group1);

        if (rs_symm_ptr.mc && ag_symm_ptr.mc) {
            const int max_ctas = max_ctas_.apply(40);
            AllreduceResidualBiasRMSnormV_NVLS<<<max_ctas, 1024, 0, stream>>>(rs_symm_ptr.mc,
                                                                              ag_symm_ptr.mc,
                                                                              (T*)residual,
                                                                              (const T*)bias,
                                                                              (const T*)weights,
                                                                              semaphore,
                                                                              g_rank,
                                                                              n_ranks(0),
                                                                              first,
                                                                              last,
                                                                              offset,
                                                                              dim / vec_size,
                                                                              1.f / dim,
                                                                              eps,
                                                                              constant<vec_size>{},
                                                                              constant<1024>{},
                                                                              constant<1>{},
                                                                              std::true_type{});
        }
        else {
            Array<int2, kMaxRanks> ag_ranges{};
            for (int i = 0; i < tp1; ++i) {
                const auto r = g1.l2g[i];
                ag_ranges[i] = {offsets[r] + firsts[r], offsets[r] + lasts[r]};
            }
            const int max_ctas = max_ctas_.apply(48);
            AllreduceResidualBiasRMSnormV_Simple_Pull<<<max_ctas, 1024, 0, stream>>>((T*)hidden,
                                                                                     (T*)residual,
                                                                                     (const T*)bias,
                                                                                     (const T*)weights,
                                                                                     rs_symm_ptr.uc,
                                                                                     ag_symm_ptr.uc,
                                                                                     semaphore,
                                                                                     rank(group0),
                                                                                     rank(group1),
                                                                                     tp0,
                                                                                     tp1,
                                                                                     rank(0),
                                                                                     n_ranks(0),
                                                                                     offset,
                                                                                     first,
                                                                                     last,
                                                                                     ag_ranges,
                                                                                     dim / vec_size,
                                                                                     1.f / dim,
                                                                                     eps,
                                                                                     constant<vec_size>{},
                                                                                     constant<1024>{},
                                                                                     constant<1>{},
                                                                                     std::true_type{});
        }
        return true;
    };

    sync_check_cuda_error();

    auto dispatch_D = [&](auto t) {
        using T                = decltype(t);
        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        if (dim % vec_size) {
            return false;  // non-aligned
        }
        const int vdim = dim / vec_size;
        if (0) {}
        else if (vdim <= 256) {
            return invoke(t, constant<4>{});
        }
        else if (vdim <= 512) {
            return invoke(t, constant<2>{});
        }
        else if (vdim <= 1024) {
            return invoke(t, constant<1>{});
        }
        return false;  // > 1024 vdim
    };

    auto dispatch = [&]() -> bool {  //
        TM_DISPATCH_PRIMARY_DTYPES_RET(dtype, dispatch_D);
    };

    TM_CHECK(dispatch());
}

}  // namespace turbomind::comm

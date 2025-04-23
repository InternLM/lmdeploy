// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/device_semaphore.h"
#include "src/turbomind/comm/cuda_ipc/group_sum.h"

#include "src/turbomind/comm/cuda_ipc/mscclpp.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

using mscclpp::D2DSemaphoreHandle;

template<class T, int vec_size, int block_dim, int groups, class Relaxed>
__global__ void AllreduceResidualBiasRMSnorm_Simple_Pull_V(T*                       buf,
                                                           T*                       res,
                                                           const T*                 bias,
                                                           const T*                 weights,
                                                           Array<T*, kMaxNearPeers> rs_buf,
                                                           Array<T*, kMaxNearPeers> ag_buf,
                                                           D2DSemaphoreHandle*      rs_sem,
                                                           D2DSemaphoreHandle*      ag_sem,
                                                           D2DSemaphoreHandle*      semaphores,
                                                           int                      rs_rank,
                                                           int                      ag_rank,
                                                           int                      rs_peers,
                                                           int                      ag_peers,
                                                           int                      g_rank,
                                                           int                      g_peers,
                                                           Array<int, kMaxRanks>    offsets,
                                                           Array<int, kMaxRanks>    firsts,
                                                           Array<int, kMaxRanks>    lasts,
                                                           Array<int, kMaxRanks>    ag_g_ranks,
                                                           int                      vdim,
                                                           float                    inv_dim,
                                                           float                    eps,
                                                           constant<vec_size>,
                                                           constant<block_dim>,
                                                           constant<groups>,
                                                           Relaxed relaxed)
{
    DeviceSemaphore sem;

    if (threadIdx.x < g_peers) {
        sem.Load(&semaphores[blockIdx.x * g_peers + threadIdx.x]);
        sem.SignalAndWait(relaxed);
    }

    __syncthreads();

    using Vec = Array<T, vec_size>;

    using namespace ops;

    static_assert(block_dim % groups == 0);
    constexpr int threads = block_dim / groups;

    static_assert(threads % WARP_SIZE == 0);
    constexpr int warps = threads / WARP_SIZE;

    const int xi = threadIdx.x / threads;
    const int di = threadIdx.x % threads;
    const int bi = blockIdx.x * groups + xi;
    const int bn = gridDim.x * groups;

    auto syncgroup = [&] {  //
        asm volatile("bar.sync %0, %1;" : : "r"(15 - xi), "r"(threads) : "memory");
    };

    Rank r{rs_rank, rs_peers};

    const int first  = firsts[g_rank];
    const int last   = lasts[g_rank];
    const int offset = offsets[g_rank];

    for (int i = 0; i < rs_peers - 1; ++i) {
        const int  p   = r.get_next_peer(i);
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

    {
        const T* chn{};
        if (rs_peers) {
            const int p = r.get_next_peer(rs_peers - 1);  // last peer
            chn         = cvta_generic_to_global(rs_buf[p]);
        }
        for (int ti = first + bi; ti < last; ti += bn) {
            const int idx = ((offset + ti) * vdim + di) * vec_size;
            Vec       acc, tmp;
            Vec       r_vec{};
            float     sum{};
            if (di < vdim) {
                if (rs_peers) {
                    Load(tmp, chn + idx);
                }
                Load(acc, buf + idx);
                if (rs_peers) {
                    acc = acc + tmp;
                }
                Load(r_vec, res + (ti * vdim + di) * vec_size);
                r_vec = r_vec + acc;
                if (bias) {
                    Vec b_vec;
                    Ldg(b_vec, bias + di * vec_size);
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
                Vec w_vec;
                Ldg(w_vec, weights + di * vec_size);
                PRAGMA_UNROLL
                for (int i = 0; i < vec_size; ++i) {
                    r_vec[i] = static_cast<T>(((float)r_vec[i] * sum)) * w_vec[i];
                }
                Store(buf + idx, r_vec);
            }
        }
    }

    __syncthreads();

    if (threadIdx.x < g_peers) {
        sem.SignalAndWait(relaxed);
    }

    __syncthreads();

    r = Rank{ag_rank, ag_peers};

    for (int i = 0; i < ag_peers; ++i) {
        const int p      = r.get_next_peer(i);
        const int p_rank = ag_g_ranks[p];  // global rank
        const int offset = offsets[p_rank];
        const int first  = firsts[p_rank];
        const int last   = lasts[p_rank];
        auto      src    = cvta_generic_to_global(ag_buf[p]);
        for (int ti = offset + first + bi; ti < offset + last; ti += bn) {
            const int idx = (ti * vdim + di) * vec_size;
            if (di < vdim) {
                Vec vec;
                Load(vec, src + idx);
                Store(buf + idx, vec);
            }
        }
    }

    __syncthreads();

    if (threadIdx.x < g_peers) {
        // this and the `__syncthreads` above are used to block later kernels from modifying shared `buf` before all
        // ranks done copying from it
        sem.SignalAndWait(true);
        sem.Save(&semaphores[blockIdx.x * g_peers + threadIdx.x]);
    }
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

    auto l2g1 = g1.l2g;
    l2g1.erase(l2g1.begin() + rank(group1));
    Array<int, kMaxRanks> ag_g_ranks{};
    std::copy(l2g1.begin(), l2g1.end(), ag_g_ranks.begin());

    auto invoke = [&](auto t, auto groups) {
        using T                = decltype(t);
        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        AllreduceResidualBiasRMSnorm_Simple_Pull_V<<<48, 1024, 0, stream>>>((T*)hidden,
                                                                            (T*)residual,
                                                                            (const T*)bias,
                                                                            (const T*)weights,
                                                                            get_symmetric((T*)hidden, group0),
                                                                            get_symmetric((T*)hidden, group1),
                                                                            g0.d2d_semaphores,
                                                                            g1.d2d_semaphores,
                                                                            groups_.at(0).d2d_semaphores,
                                                                            rank(group0),
                                                                            rank(group1),
                                                                            tp0 - 1,
                                                                            tp1 - 1,
                                                                            rank(0),
                                                                            n_ranks(0) - 1,
                                                                            offsets,
                                                                            firsts,
                                                                            lasts,
                                                                            ag_g_ranks,
                                                                            dim / vec_size,
                                                                            1.f / dim,
                                                                            eps,
                                                                            constant<vec_size>{},
                                                                            constant<1024>{},
                                                                            constant<1>{},
                                                                            std::true_type{});
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

// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_bf16.h>
#include <type_traits>

#include "cub/block/block_reduce.cuh"

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/group_sum.h"
#include "src/turbomind/comm/cuda_ipc/multimem.cuh"
#include "src/turbomind/comm/cuda_ipc/semaphore.cuh"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

template<class T, int vec_size, int block_dim, int groups, class Relaxed>
__global__ void AllreduceResidualBiasRMSnorm_Simple_Pull(T*                   buf,
                                                         T*                   res,
                                                         const T*             bias,
                                                         const T*             weights,
                                                         Array<T*, kMaxRanks> symm,
                                                         SystemSemaphoreInfo* semaphores,
                                                         int                  rank,
                                                         int                  ranks,
                                                         int                  slice,
                                                         int                  count,
                                                         int                  vdim,
                                                         float                inv_dim,
                                                         float                eps,
                                                         constant<vec_size>,
                                                         constant<block_dim>,
                                                         constant<groups>,
                                                         Relaxed relaxed)
{
    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(relaxed);
    sem.Wait(relaxed);

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

    const int first = rank * slice;
    const int last  = min(count, first + slice);

    for (int i = 1; i < ranks - 1; ++i) {
        const int  p   = rank + i < ranks ? rank + i : rank + i - ranks;
        const auto src = cvta_generic_to_global(symm[p]);
        Vec        acc, tmp;
        for (int ti = first + bi; ti < last; ti += bn) {
            const int idx = (ti * vdim + di) * vec_size;
            if (di < vdim) {
                Load(tmp, src + idx);
                Load(acc, buf + idx);
                acc = acc + tmp;
                Store(buf + idx, acc);
            }
        }
    }

    Vec b_vec{};
    if (bias && di < vdim) {
        Ldg(b_vec, bias + di * vec_size);
    }

    Vec w_vec;
    if (di < vdim) {
        Ldg(w_vec, weights + di * vec_size);
    }

    {
        const int p   = rank > 0 ? rank - 1 : ranks - 1;  // last peer
        auto      chn = cvta_generic_to_global(symm[p]);
        for (int ti = first + bi; ti < last; ti += bn) {
            const int idx = (ti * vdim + di) * vec_size;
            Vec       acc, tmp;
            Vec       r_vec{};
            float     sum{};
            if (di < vdim) {
                Load(tmp, chn + idx);
                Load(acc, buf + idx);
                acc = acc + tmp;
                Load(r_vec, res + idx);
                r_vec = r_vec + acc;
                if (bias) {
                    r_vec = r_vec + b_vec;
                }
                Store(res + idx, r_vec);
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

    for (int i = 1; i < ranks; ++i) {
        const int p     = rank + i < ranks ? rank + i : rank + i - ranks;
        const int first = slice * p;
        const int last  = min(count, first + slice);
        auto      src   = cvta_generic_to_global(symm[p]);
        for (int ti = first + bi; ti < last; ti += bn) {
            const int idx = (ti * vdim + di) * vec_size;
            if (di < vdim) {
                Vec vec;
                Load(vec, src + idx);
                Store(buf + idx, vec);
            }
        }
    }

    __syncthreads();

    sem.Signal(true);
    sem.Wait(true);

    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

template<class T, int vec_size, int block_dim, int groups, class Relaxed>
__global__ void AllreduceResidualBiasRMSnorm_NVLS(T*                   mc_buf,
                                                  T*                   uc_buf,
                                                  T*                   res,
                                                  const T*             bias,
                                                  const T*             weights,
                                                  SystemSemaphoreInfo* semaphores,
                                                  int                  rank,
                                                  int                  ranks,
                                                  int                  slice,
                                                  int                  count,
                                                  int                  vdim,
                                                  float                inv_dim,
                                                  float                eps,
                                                  constant<vec_size>,
                                                  constant<block_dim>,
                                                  constant<groups>,
                                                  Relaxed relaxed)
{

#if TURBOMIND_ARCH_SM90

    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(relaxed);

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

    using namespace ops;

    const int bi = blockIdx.x * groups + xi;
    const int bn = gridDim.x * groups;

    auto syncgroup = [&] {  //
        asm volatile("bar.sync %0, %1;" : : "r"(15 - xi), "r"(threads) : "memory");
    };

    const int first = rank * slice;
    const int last  = min(count, first + slice);

    for (int ti = first + bi; ti < last; ti += bn) {
        const int idx = (ti * vdim + di) * vec_size;
        float     sum{};
        Vec       vec;
        if (di < vdim) {
            Vec acc = multimem_ld_reduce_sum((const Vec*)(mc_buf + idx));
            Load(vec, res + idx);
            vec = vec + acc;
            if (bias) {
                vec = vec + b_vec;
            }
            Store(res + idx, vec);
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
            multimem_st(mc_buf + idx, vec);
        }
    }

    __syncthreads();

    sem.Signal(true);
    sem.Wait(true);

    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);

#endif
}

template<class T, int vec_size, int block_dim, int groups, class Relaxed>
__global__ void AllreduceResidualBiasRMSnorm_Simple_Push(T*                   buf,
                                                         T*                   res,
                                                         const T*             bias,
                                                         const T*             weights,
                                                         T*                   scratch,
                                                         Array<T*, kMaxRanks> symm_buf,
                                                         Array<T*, kMaxRanks> symm_scratch,
                                                         SystemSemaphoreInfo* semaphores,
                                                         int                  rank,
                                                         int                  ranks,
                                                         int                  slice,
                                                         int                  count,
                                                         int                  vdim,
                                                         float                inv_dim,
                                                         float                eps,
                                                         constant<vec_size>,
                                                         constant<block_dim>,
                                                         constant<groups>,
                                                         Relaxed relaxed)
{
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

    for (int i = 1; i < ranks; ++i) {
        const int  p   = rank + i < ranks ? rank + i : rank + i - ranks;
        const int  n   = min(count, p * slice + slice) - p * slice;
        const auto src = buf + p * slice * vdim * vec_size;
        const auto dst = symm_scratch[p] + rank * slice * vdim * vec_size;
        for (int ti = bi; ti < n; ti += bn) {
            if (di < vdim) {
                Vec vec;
                Load(vec, src + (ti * vdim + di) * vec_size);
                Store(dst + (ti * vdim + di) * vec_size, vec);
            }
        }
    }

    __syncthreads();

    SystemSemaphore sem(semaphores, ranks, blockIdx.x, threadIdx.x);

    sem.Signal(relaxed);
    sem.Wait(relaxed);

    __syncthreads();

    Vec b_vec{};
    if (bias && di < vdim) {
        Ldg(b_vec, bias + di * vec_size);
    }

    Vec w_vec;
    if (di < vdim) {
        Ldg(w_vec, weights + di * vec_size);
    }

    const int n = min(count, rank * slice + slice) - rank * slice;

    for (int ti = bi; ti < n; ti += bn) {
        const int idx = ((rank * slice + ti) * vdim + di) * vec_size;  // idx into local buffers
        Vec       r_vec{};
        float     sum{};
        if (di < vdim) {
            Vec acc;
            Load(acc, buf + idx);
            for (int i = 1; i < ranks; ++i) {
                const int p = rank + i < ranks ? rank + i : rank + i - ranks;
                Vec       tmp;
                Load(tmp, scratch + ((p * slice + ti) * vdim + di) * vec_size);
                acc = acc + tmp;
            }
            Load(r_vec, res + idx);
            r_vec = r_vec + acc;
            if (bias) {
                r_vec = r_vec + b_vec;
            }
            Store(res + idx, r_vec);
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
            for (int i = 1; i < ranks; ++i) {
                const int p = rank + i < ranks ? rank + i : rank + i - ranks;
                Store(symm_buf[p] + ((rank * slice + ti) * vdim + di) * vec_size, r_vec);
            }
        }
    }

    __syncthreads();

    sem.Signal(true);
    sem.Wait(true);

    sem.Update(semaphores, ranks, blockIdx.x, threadIdx.x);
}

void CudaIpcCommImpl::AllreduceResidualBiasRMSnorm(void*        hidden,
                                                   void*        residual,
                                                   const void*  bias,
                                                   const void*  weights,
                                                   float        eps,
                                                   int          dim,
                                                   int          token_num,
                                                   DataType     dtype,
                                                   int          group,
                                                   cudaStream_t stream)
{

    const size_t elemsize = byte_size(dtype);
    const size_t bytesize = elemsize * token_num * dim;

    const int n_ranks = this->n_ranks(group);
    const int rank    = this->rank(group);

    auto semaphore = groups_.at(group).semaphore.handle();

    auto invoke = [&](auto t, auto groups) {
        using T                = decltype(t);
        auto          symm_ptr = get_symmetric_v2((T*)hidden, group);
        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        const int     slice    = (token_num + n_ranks - 1) / n_ranks;
        const int     count    = token_num;

        if (symm_ptr.mc) {
            constexpr int block_dim = 1024;
            const int     max_ctas  = max_ctas_.apply(8);
            const int     blocks    = std::min((slice + groups - 1) / groups, max_ctas);
            AllreduceResidualBiasRMSnorm_NVLS<<<blocks, block_dim, 0, stream>>>(symm_ptr.mc,
                                                                                (T*)hidden,
                                                                                (T*)residual,
                                                                                (const T*)bias,
                                                                                (const T*)weights,
                                                                                semaphore,
                                                                                rank,
                                                                                n_ranks,
                                                                                slice,
                                                                                count,
                                                                                dim / vec_size,
                                                                                1.f / dim,
                                                                                eps,
                                                                                constant<vec_size>{},
                                                                                constant<block_dim>{},
                                                                                groups,
                                                                                std::false_type{});
        }
#if 1
        else if (bytesize <= 1 << 19) {
            return false;
        }
#endif
        else if (bytesize <= kScratchBuffSize && bytesize <= 6 << 20) {
            constexpr int block_dim    = 1024;
            const int     max_ctas     = max_ctas_.apply(48);
            const int     blocks       = std::min((slice + groups - 1) / groups, max_ctas);
            auto          symm_scratch = get_symmetric_v2((T*)scratch_buff_, group).uc;
            AllreduceResidualBiasRMSnorm_Simple_Push<<<blocks, block_dim, 0, stream>>>((T*)hidden,
                                                                                       (T*)residual,
                                                                                       (const T*)bias,
                                                                                       (const T*)weights,
                                                                                       (T*)scratch_buff_,
                                                                                       symm_ptr.uc,
                                                                                       symm_scratch,
                                                                                       semaphore,
                                                                                       rank,
                                                                                       n_ranks,
                                                                                       slice,
                                                                                       count,
                                                                                       dim / vec_size,
                                                                                       1.f / dim,
                                                                                       eps,
                                                                                       constant<vec_size>{},
                                                                                       constant<block_dim>{},
                                                                                       groups,
                                                                                       std::false_type{});
        }
        else {
            constexpr int block_dim = 1024;
            const int     max_ctas  = max_ctas_.apply(48);
            const int     blocks    = std::min((slice + groups - 1) / groups, max_ctas);
            AllreduceResidualBiasRMSnorm_Simple_Pull<<<blocks, block_dim, 0, stream>>>((T*)hidden,
                                                                                       (T*)residual,
                                                                                       (const T*)bias,
                                                                                       (const T*)weights,
                                                                                       symm_ptr.uc,
                                                                                       semaphore,
                                                                                       rank,
                                                                                       n_ranks,
                                                                                       slice,
                                                                                       count,
                                                                                       dim / vec_size,
                                                                                       1.f / dim,
                                                                                       eps,
                                                                                       constant<vec_size>{},
                                                                                       constant<block_dim>{},
                                                                                       groups,
                                                                                       std::false_type{});
        }

        return true;
    };

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

    auto dispatch = [&]() -> bool { TM_DISPATCH_PRIMARY_DTYPES_RET(dtype, dispatch_D); };

    if (dispatch()) {
        return;
    }

    // fallback
    AllReduceSum(hidden, hidden, token_num * dim, dtype, group, stream);
    invokeResidualBiasRMSNorm(hidden, residual, weights, bias, dtype, dim, token_num, eps, stream);
}

}  // namespace turbomind::comm

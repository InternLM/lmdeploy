// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_bf16.h>
#include <type_traits>

#include "cub/block/block_reduce.cuh"

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/comm/cuda_ipc/device_semaphore.h"
#include "src/turbomind/comm/cuda_ipc/group_sum.h"
#include "src/turbomind/comm/cuda_ipc/multimem.cuh"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

template<class T, int vec_size, int block_dim, int groups, class Relaxed>
__global__ void AllreduceResidualBiasRMSnorm_Simple_Pull(T*                           buf,
                                                         T*                           res,
                                                         const T*                     bias,
                                                         const T*                     weights,
                                                         Array<T*, kMaxNearPeers>     near,
                                                         mscclpp::D2DSemaphoreHandle* semaphores,
                                                         int                          rank,
                                                         int                          peers,
                                                         int                          slice,
                                                         int                          count,
                                                         int                          vdim,
                                                         float                        inv_dim,
                                                         float                        eps,
                                                         constant<vec_size>,
                                                         constant<block_dim>,
                                                         constant<groups>,
                                                         Relaxed relaxed)
{
    DeviceSemaphore sem;

    if (threadIdx.x < peers) {
        sem.Load(&semaphores[blockIdx.x * peers + threadIdx.x]);
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

    Rank r{rank, peers};

    const int first = rank * slice;
    const int last  = min(count, first + slice);

    for (int i = 0; i < peers - 1; ++i) {
        const int  p   = r.get_next_peer(i);
        const auto src = cvta_generic_to_global(near[p]);
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
        const int p   = r.get_next_peer(peers - 1);  // last peer
        auto      chn = cvta_generic_to_global(near[p]);
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

    if (threadIdx.x < peers) {
        sem.SignalAndWait(relaxed);
    }

    __syncthreads();

    for (int i = 0; i < peers; ++i) {
        const int p      = r.get_next_peer(i);
        const int p_rank = r.get_peer_rank(p);
        const int first  = slice * p_rank;
        const int last   = min(count, first + slice);
        auto      src    = cvta_generic_to_global(near[p]);
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

    if (threadIdx.x < peers) {
        // this and the `__syncthreads` above are used to block later kernels from modifying shared `buf` before all
        // ranks done copying from it
        sem.SignalAndWait(true);
        sem.Save(&semaphores[blockIdx.x * peers + threadIdx.x]);
    }
}

template<class T, int vec_size, int block_dim, int groups, class Relaxed>
__global__ void AllreduceResidualBiasRMSnorm_NVLS(T*                           mc_buf,
                                                  T*                           uc_buf,
                                                  T*                           res,
                                                  const T*                     bias,
                                                  const T*                     weights,
                                                  Array<T*, kMaxNearPeers>     near,
                                                  mscclpp::D2DSemaphoreHandle* semaphores,
                                                  int                          rank,
                                                  int                          peers,
                                                  int                          slice,
                                                  int                          count,
                                                  int                          vdim,
                                                  float                        inv_dim,
                                                  float                        eps,
                                                  constant<vec_size>,
                                                  constant<block_dim>,
                                                  constant<groups>,
                                                  Relaxed relaxed)
{
    DeviceSemaphore sem;

    if (threadIdx.x < peers) {
        sem.Load(&semaphores[blockIdx.x * peers + threadIdx.x]);
        sem.Signal();
    }

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

    if (threadIdx.x < peers) {
        sem.Wait();
    }

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

    if (threadIdx.x < peers) {
        // this and the `__syncthreads` above are used to block later kernels from modifying shared `buf` before all
        // ranks done copying from it
        sem.SignalAndWait(true);
        sem.Save(&semaphores[blockIdx.x * peers + threadIdx.x]);
    }
}

template<class T, int vec_size, int block_dim, bool aligned, class Peers, class Relaxed>
__global__ void AllreduceResidualBiasRMSnormKernel_Simple_v3(T*                           buf,
                                                             T*                           res,
                                                             const T*                     bias,
                                                             const T*                     weights,
                                                             Array<T*, kMaxNearPeers>     chns,
                                                             mscclpp::D2DSemaphoreHandle* semaphores,
                                                             int                          rank,
                                                             Peers                        peers,
                                                             int                          slice,
                                                             int                          count,
                                                             int                          vdim,
                                                             float                        inv_dim,
                                                             float                        eps,
                                                             constant<vec_size>,
                                                             constant<block_dim>,
                                                             constant<aligned>,
                                                             Relaxed relaxed)
{
    const int bi        = blockIdx.x;
    const int block_num = gridDim.x;

    DeviceSemaphore sem;

    if (threadIdx.x < peers) {
        sem.Load(&semaphores[blockIdx.x * peers + threadIdx.x]);
        sem.SignalAndWait(relaxed);
    }

    __syncthreads();

    using Vec = Array<T, vec_size>;

    using namespace ops;

    const int  di       = threadIdx.x;
    const bool is_valid = di < vdim;

    if (aligned || is_valid) {

        const int first = rank * slice;
        const int last  = min(count, first + slice);

        __shared__ const T* chs[8];
        for (int p = 0; p < peers; ++p) {
            const int peer = p + rank < peers ? p + rank : p + rank - peers;
            chs[p]         = chns[peer];
        }

        for (int ti = first + bi; ti < last; ti += block_num) {
            const int idx = (ti * vdim + di) * vec_size;
            Vec       acc;
            Load(acc, buf + idx);
            for (int p = 0; p < peers; ++p) {
                Vec tmp;
                Load(tmp, chs[p] + idx);
                acc = acc + tmp;
            }
            Vec r_vec, x_vec;
            Load(r_vec, res + idx);
            r_vec = r_vec + acc;
            if (bias) {
                Load(x_vec, bias + di * vec_size);
                r_vec = r_vec + x_vec;
            }
            Store(res + idx, r_vec);
            float sum{};
            PRAGMA_UNROLL
            for (int i = 0; i < vec_size; ++i) {
                sum += (float)r_vec[i] * (float)r_vec[i];
            }
            using BlockReduce = cub::BlockReduce<float, block_dim>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            __shared__ float                             shared_sum;
            sum = BlockReduce{temp_storage}.Sum(sum);
            if (di == 0) {
                shared_sum = rsqrtf(sum * inv_dim + eps);
            }
            __syncthreads();
            sum = shared_sum;
            Load(x_vec, weights + di * vec_size);
            PRAGMA_UNROLL
            for (int i = 0; i < vec_size; ++i) {
                r_vec[i] = static_cast<T>(((float)r_vec[i] * sum)) * x_vec[i];
            }
            Store(buf + idx, r_vec);
        }
    }

    __syncthreads();

    if (threadIdx.x < peers) {
        sem.SignalAndWait(relaxed);
    }

    __syncthreads();

    if (aligned || is_valid) {
        for (int p = 0; p < peers; ++p) {
            const int peer      = p + rank < peers ? p + rank : p + rank - peers;
            const int peer_rank = peer < rank ? peer : peer + 1;
            const int first     = slice * peer_rank;
            const int last      = min(count, first + slice);
            auto      chn       = cvta_generic_to_global(chns[peer]);
            Vec       vec;
            for (int ti = first + bi; ti < last; ti += block_num) {
                const int idx = (ti * vdim + di) * vec_size;
                Load(vec, chn + idx);
                Store(buf + idx, vec);
            }
        }
    }

    __syncthreads();

    if (threadIdx.x < peers) {
        sem.SignalAndWait(relaxed);
        sem.Save(&semaphores[blockIdx.x * peers + threadIdx.x]);
    }
}

template<class T, int vec_size, int block_dim, int groups, class Peers, class Relaxed>
__global__ void AllreduceResidualBiasRMSnorm_Simple_Push(T*                           buf,
                                                         T*                           res,
                                                         const T*                     bias,
                                                         const T*                     weights,
                                                         T*                           scratch,
                                                         Array<T*, kMaxNearPeers>     near_buf,
                                                         Array<T*, kMaxNearPeers>     near_scratch,
                                                         mscclpp::D2DSemaphoreHandle* semaphores,
                                                         int                          rank,
                                                         Peers                        peers,
                                                         int                          slice,
                                                         int                          count,
                                                         int                          vdim,
                                                         float                        inv_dim,
                                                         float                        eps,
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

    Rank r{rank, peers};

    for (int i = 0; i < peers; ++i) {
        const int  p      = r.get_next_peer(i);
        const int  p_rank = r.get_peer_rank(p);
        const int  n      = min(count, p_rank * slice + slice) - p_rank * slice;
        const auto src    = buf + p_rank * slice * vdim * vec_size;
        const auto dst    = near_scratch[p] + r.inverse_peer(p) * slice * vdim * vec_size;
        for (int ti = bi; ti < n; ti += bn) {
            if (di < vdim) {
                Vec vec;
                Load(vec, src + (ti * vdim + di) * vec_size);
                Store(dst + (ti * vdim + di) * vec_size, vec);
            }
        }
    }

    __syncthreads();

    DeviceSemaphore sem;

    if (threadIdx.x < peers) {
        sem.Load(&semaphores[blockIdx.x * peers + threadIdx.x]);
        sem.SignalAndWait(relaxed);
    }

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
            for (int i = 0; i < peers; ++i) {
                Vec tmp;
                Load(tmp, scratch + ((i * slice + ti) * vdim + di) * vec_size);
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
            for (int i = 0; i < peers; ++i) {
                const int p = r.get_next_peer(i);
                Store(near_buf[p] + ((rank * slice + ti) * vdim + di) * vec_size, r_vec);
            }
        }
    }

    __syncthreads();

    if (threadIdx.x < peers) {
        sem.SignalAndWait(relaxed);
        sem.Save(&semaphores[blockIdx.x * peers + threadIdx.x]);
    }
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

    auto semaphores = groups_.at(group).d2d_semaphores;

    auto invoke = [&](auto t, auto groups) {
        using T                 = decltype(t);
        constexpr int vec_size  = sizeof(uint4) / sizeof(T);
        const int     slice     = (token_num + n_ranks - 1) / n_ranks;
        const int     count     = token_num;
        constexpr int block_dim = 1024;
        const int     max_ctas  = 48;
        const int     blocks    = std::min((slice + groups - 1) / groups, max_ctas);
        if (0 && bytesize <= kScratchBuffSize && bytesize <= 6 << 20) {
            AllreduceResidualBiasRMSnorm_Simple_Push<<<blocks, block_dim, 0, stream>>>(
                (T*)hidden,
                (T*)residual,
                (const T*)bias,
                (const T*)weights,
                (T*)scratch_buff_,
                get_symmetric((T*)hidden, group),
                get_symmetric((T*)scratch_buff_, group),
                semaphores,
                rank,
                n_ranks - 1,
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
        else if (0) {
            AllreduceResidualBiasRMSnorm_Simple_Pull<<<blocks, block_dim, 0, stream>>>((T*)hidden,
                                                                                       (T*)residual,
                                                                                       (const T*)bias,
                                                                                       (const T*)weights,
                                                                                       get_symmetric((T*)hidden, group),
                                                                                       semaphores,
                                                                                       rank,
                                                                                       n_ranks - 1,
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
            void* mc_ptr{};
            for (auto& [p, a] : allocations_) {
                if ((char*)p <= (char*)hidden && (char*)hidden < (char*)p + a.size) {
                    auto offset = (char*)hidden - (char*)p;
                    mc_ptr      = (char*)a.mc_ptr + offset;
                }
            }
            FT_CHECK(mc_ptr);

            constexpr int block_dim = 1024;
            const int     max_ctas  = 3;
            const int     blocks    = std::min((slice + groups - 1) / groups, max_ctas);

            AllreduceResidualBiasRMSnorm_NVLS<<<blocks, block_dim, 0, stream>>>((T*)mc_ptr,
                                                                                (T*)hidden,
                                                                                (T*)residual,
                                                                                (const T*)bias,
                                                                                (const T*)weights,
                                                                                get_symmetric((T*)hidden, group),
                                                                                semaphores,
                                                                                rank,
                                                                                n_ranks - 1,
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

    dispatch();
    return;

    // if (bytesize > (1 << 19)) {
    //     if (dispatch()) {
    //         return;
    //     }
    // }

    // fallback
    AllReduceSum(hidden, hidden, token_num * dim, dtype, group, stream);
    invokeResidualBiasRMSNorm(hidden, residual, weights, bias, dtype, dim, token_num, eps, stream);
}

}  // namespace turbomind::comm

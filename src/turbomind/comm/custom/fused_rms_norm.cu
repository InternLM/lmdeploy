
#include <atomic>
#include <stdexcept>
#include <type_traits>

#include "cub/block/block_reduce.cuh"

#include "src/turbomind/comm/common.h"

#include "mscclpp/packet_device.hpp"

#include "src/turbomind/comm/custom/custom_comm.h"
#include "src/turbomind/comm/custom/device_semaphore.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/kernels/norm/rms_norm.h"

namespace turbomind {

using LLPacket = mscclpp::LLPacket;

template<class T, int ctas_per_peer, int vec_size, int block_dim>
__global__ void __launch_bounds__(1024, 1) AllreduceResidualBiasRMSnormKernel_LL(T*                  buf,
                                                                                 T*                  res,
                                                                                 const T*            bias,
                                                                                 const T*            weights,
                                                                                 LLPacket*           packets,
                                                                                 Array<LLPacket*, 8> chns,
                                                                                 int                 rank,
                                                                                 int                 peers,
                                                                                 int                 slice0,
                                                                                 int                 count0,
                                                                                 int                 slice1,
                                                                                 int                 count1,
                                                                                 int                 dim,
                                                                                 float               inv_dim,
                                                                                 float               eps,
                                                                                 uint32_t            flag)
{

    static_assert(vec_size == sizeof(uint2) / sizeof(T));

    const int _p = blockIdx.x / ctas_per_peer;
    const int pi = _p + rank < peers ? _p + rank : _p + rank - peers;
    const int bi = blockIdx.x % ctas_per_peer;

    const int peer_rank = pi < rank ? pi : pi + 1;  // rank of `pi`

    {  // send local slice of `buf` to peers  (src -> packet0)
        const int    first = peer_rank * slice0;
        const int    slice = min(count0, first + slice0) - first;
        LLPacket*    chn   = chns[pi] + (pi < rank ? rank - 1 : rank) * slice0;
        const uint2* src   = (const uint2*)buf + first;
        if (threadIdx.x == 0) {
            printf("rank %d, first %d, count0 %d, slice0 %d, slice %d\n", rank, first, count0, slice0, slice);
        }
        for (int idx = threadIdx.x + bi * blockDim.x; idx < slice; idx += ctas_per_peer * blockDim.x) {
            chn[idx].write(src[idx], flag);
        }
    }

    {
        using Vec = Array<T, vec_size>;
        using namespace ops;

        // in tokens
        const int first = rank * slice1;
        const int slice = min(count1, first + slice1) - first;

        T* buf1 = buf + first * dim;
        T* res1 = res + first * dim;

        for (int token_idx = blockIdx.x; token_idx < slice; token_idx += gridDim.x) {
            Array<float, vec_size> acc{};
            for (int d = threadIdx.x; d < dim; d += block_dim) {
                const int idx = token_idx * dim + d;
                Vec       r, x;
                Load(x, buf1 + idx * vec_size);
                Load(r, res1 + idx * vec_size);
                for (int p = 0; p < peers; ++p) {
                    uint2 data = packets[p * slice1 * dim + idx].read(flag);
                    x          = x + (Vec&)data;
                }
                r = r + x;
                if (bias) {
                    Vec b;
                    Load(b, bias + d * vec_size);
                    r = r + b;
                }
                Store(res1 + idx * vec_size, r);
                auto tmp = cast<float>(r);
                acc      = acc + tmp * tmp;
            }
            // float sum{};
            // PRAGMA_UNROLL
            // for (int i = 0; i < vec_size; ++i) {
            //     sum += acc[i];
            // }
            // using BlockReduce = cub::BlockReduce<float, block_dim>;
            // __shared__ typename BlockReduce::TempStorage temp_storage;

            // sum = BlockReduce{temp_storage}.Sum(sum);

            // __shared__ float shared_sum;

            // if (threadIdx.x == 0) {
            //     shared_sum = rsqrtf(sum * inv_dim + eps);
            // }

            // __syncthreads();

            // sum = shared_sum;

            // for (int d = threadIdx.x; d < dim; d += block_dim) {
            //     const int idx = token_idx * dim + d;
            //     Vec       x, w;
            //     Load(x, res1 + idx * vec_size);
            //     Load(w, weights + d * vec_size);
            //     PRAGMA_UNROLL
            //     for (int i = 0; i < vec_size; ++i) {
            //         x[i] = (T)((float)x[i] * sum) * w[i];
            //     }
            //     Store(buf1 + idx * vec_size, x);
            //     for (int p = 0; p < peers; ++p) {
            //         const int peer = p + rank < peers ? p + rank : p + rank - peers;
            //         chns[peer][(peers + rank) * slice1 * dim + idx].write((uint2&)x, flag);
            //     }
            // }
        }
    }

    // if (0) {
    //     const int  first    = peer_rank * slice0;
    //     const int  slice    = min(count0, first + slice0) - first;
    //     const auto incoming = packets + (peers + peer_rank) * slice0;
    //     auto       dst      = (uint2*)buf + (peer_rank * slice0);
    //     for (int idx = threadIdx.x + bi * blockDim.x; idx < slice; idx += ctas_per_peer * blockDim.x) {
    //         dst[idx] = incoming[idx].read(flag);
    //     }
    // }
}

template<class T, int vec_size, int block_dim, class Relaxed>
__global__ void AllreduceResidualBiasRMSnormKernel_Simple(T*                                             buf,
                                                          T*                                             res,
                                                          const T*                                       bias,
                                                          const T*                                       weights,
                                                          Array<T*, 8>                                   chns,
                                                          mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* semaphores,
                                                          int                                            rank,
                                                          int                                            peers,
                                                          int                                            slice,
                                                          int                                            count,
                                                          float                                          inv_dim,
                                                          float                                          eps,
                                                          constant<vec_size>,
                                                          constant<block_dim>,
                                                          Relaxed relaxed)
{
    const int block_num  = gridDim.x;
    const int thread_num = blockDim.x * block_num;
    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int n_peer = peers;

    DeviceSemaphore sem;

    const int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id < n_peer) {
        sem.Load(&semaphores[blockIdx.x * n_peer + lane_id]);
    }

    if (threadIdx.x < n_peer) {
        if (relaxed) {
            sem.Signal(cuda::memory_order_relaxed);
            sem.Wait(cuda::memory_order_relaxed);
        }
        else {
            sem.Signal(cuda::memory_order_release);
            sem.Wait(cuda::memory_order_acquire);
        }
    }

    __syncthreads();

    const int first = rank * slice;
    const int last  = min(count, first + slice);

    using Vec = Array<T, vec_size>;

    using namespace ops;

    for (int p = 0; p < n_peer - 1; ++p) {
        const int peer = p + rank < n_peer ? p + rank : p + rank - n_peer;
        auto      chn  = cvta_generic_to_global(chns[peer]);
        Vec       acc, tmp;
        for (int idx = first + thread_idx; idx < last; idx += thread_num) {
            Load(tmp, chn + idx * vec_size);
            Load(acc, buf + idx * vec_size);
            acc = acc + tmp;
            Store(buf + idx * vec_size, acc);
        }
    }

    if (1) {
        const int p    = n_peer - 1;
        const int peer = p + rank < n_peer ? p + rank : p + rank - n_peer;
        auto      chn  = cvta_generic_to_global(chns[peer]);
        Vec       acc, tmp;
        for (int idx = first + thread_idx; idx < last; idx += thread_num) {
            Load(tmp, chn + idx * vec_size);
            Load(acc, buf + idx * vec_size);
            acc = acc + tmp;
            Vec r_vec, x_vec;
            Load(r_vec, res + idx * vec_size);
            r_vec = r_vec + acc;
            if (bias) {
                Load(x_vec, bias + threadIdx.x * vec_size);
                r_vec = r_vec + x_vec;
            }
            Store(res + idx * vec_size, r_vec);
            float sum{};
            PRAGMA_UNROLL
            for (int i = 0; i < vec_size; ++i) {
                sum += (float)r_vec[i] * (float)r_vec[i];
            }
            using BlockReduce = cub::BlockReduce<float, block_dim>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            __shared__ float                             shared_sum;
            sum = BlockReduce{temp_storage}.Sum(sum);
            if (threadIdx.x == 0) {
                shared_sum = rsqrtf(sum * inv_dim + eps);
            }
            __syncthreads();
            sum = shared_sum;
            Load(x_vec, weights + threadIdx.x * vec_size);
            PRAGMA_UNROLL
            for (int i = 0; i < vec_size; ++i) {
                r_vec[i] = static_cast<T>(((float)r_vec[i] * sum)) * x_vec[i];
            }
            Store(buf + idx * vec_size, r_vec);
        }
    }

    __syncthreads();

    if (threadIdx.x < n_peer) {
        if (relaxed) {  // It seems that fence is not needed on NVLink devices
            sem.Signal(cuda::memory_order_relaxed);
            sem.Wait(cuda::memory_order_relaxed);
        }
        else {
            sem.Signal(cuda::memory_order_release);
            sem.Wait(cuda::memory_order_acquire);
        }
    }

    __syncthreads();

    for (int p = 0; p < n_peer; ++p) {
        const int peer      = p + rank < n_peer ? p + rank : p + rank - n_peer;
        const int peer_rank = peer < rank ? peer : peer + 1;
        const int first     = slice * peer_rank;
        const int last      = min(count, first + slice);
        auto      chn       = cvta_generic_to_global(chns[peer]);
        for (size_t idx = first + thread_idx; idx < last; idx += thread_num) {
            Vec vec;
            Load(vec, chn + idx * vec_size);
            Store(buf + idx * vec_size, vec);
        }
    }

    if (threadIdx.x < n_peer) {
        sem.Save(&semaphores[blockIdx.x * n_peer + lane_id]);
    }
}

template<class T, int vec_size, int block_dim, bool aligned, class Relaxed>
__global__ void AllreduceResidualBiasRMSnormKernel_Simple_v2(T*                                             buf,
                                                             T*                                             res,
                                                             const T*                                       bias,
                                                             const T*                                       weights,
                                                             Array<T*, 8>                                   chns,
                                                             mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* semaphores,
                                                             int                                            rank,
                                                             int                                            peers,
                                                             int                                            slice,
                                                             int                                            count,
                                                             int                                            vdim,
                                                             float                                          inv_dim,
                                                             float                                          eps,
                                                             constant<vec_size>,
                                                             constant<block_dim>,
                                                             constant<aligned>,
                                                             Relaxed relaxed)
{
    const int bi        = blockIdx.x;
    const int block_num = gridDim.x;

    const int n_peer = peers;

    DeviceSemaphore sem;

    if (threadIdx.x < n_peer) {
        sem.Load(&semaphores[blockIdx.x * n_peer + threadIdx.x]);
    }

    if (threadIdx.x < n_peer) {
        if (relaxed) {
            sem.Signal(cuda::memory_order_relaxed);
            sem.Wait(cuda::memory_order_relaxed);
        }
        else {
            sem.Signal(cuda::memory_order_release);
            sem.Wait(cuda::memory_order_acquire);
        }
    }

    __syncthreads();

    using Vec = Array<T, vec_size>;

    using namespace ops;

    const int  di       = threadIdx.x;
    const bool is_valid = di < vdim;

    if (aligned || is_valid) {

        const int first = rank * slice;
        const int last  = min(count, first + slice);

        for (int p = 0; p < n_peer - 1; ++p) {
            const int  peer = p + rank < n_peer ? p + rank : p + rank - n_peer;
            const auto chn  = cvta_generic_to_global(chns[peer]);
            Vec        acc, tmp;
            for (int ti = first + bi; ti < last; ti += block_num) {
                const int idx = (ti * vdim + di) * vec_size;
                Load(tmp, chn + idx);
                Load(acc, buf + idx);
                acc = acc + tmp;
                Store(buf + idx, acc);
            }
        }

        {
            const int p    = n_peer - 1;
            const int peer = p + rank < n_peer ? p + rank : p + rank - n_peer;
            auto      chn  = cvta_generic_to_global(chns[peer]);
            Vec       acc, tmp;
            for (int ti = first + bi; ti < last; ti += block_num) {
                const int idx = (ti * vdim + di) * vec_size;
                Load(tmp, chn + idx);
                Load(acc, buf + idx);
                acc = acc + tmp;
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
    }

    __syncthreads();

    if (threadIdx.x < n_peer) {
        if (relaxed) {  // It seems that fence is not needed on NVLink devices
            sem.Signal(cuda::memory_order_relaxed);
            sem.Wait(cuda::memory_order_relaxed);
        }
        else {
            sem.Signal(cuda::memory_order_release);
            sem.Wait(cuda::memory_order_acquire);
        }
    }

    __syncthreads();

    if (aligned || is_valid) {
        for (int p = 0; p < n_peer; ++p) {
            const int peer      = p + rank < n_peer ? p + rank : p + rank - n_peer;
            const int peer_rank = peer < rank ? peer : peer + 1;
            const int first     = slice * peer_rank;
            const int last      = min(count, first + slice);
            auto      chn       = cvta_generic_to_global(chns[peer]);
            for (int ti = first + bi; ti < last; ti += block_num) {
                const int idx = (ti * vdim + di) * vec_size;
                Vec       vec;
                Load(vec, chn + idx);
                Store(buf + idx, vec);
            }
        }
    }

    if (threadIdx.x < n_peer) {
        sem.Save(&semaphores[blockIdx.x * n_peer + threadIdx.x]);
    }
}

template<class T, int vec_size, int block_dim, bool aligned, class Peers, class Relaxed>
__global__ void AllreduceResidualBiasRMSnormKernel_Simple_v3(T*                                             buf,
                                                             T*                                             res,
                                                             const T*                                       bias,
                                                             const T*                                       weights,
                                                             Array<T*, 8>                                   chns,
                                                             mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* semaphores,
                                                             int                                            rank,
                                                             Peers                                          peers,
                                                             int                                            slice,
                                                             int                                            count,
                                                             int                                            vdim,
                                                             float                                          inv_dim,
                                                             float                                          eps,
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
    }

    if (threadIdx.x < peers) {
        if (relaxed) {
            sem.Signal(cuda::memory_order_relaxed);
            sem.Wait(cuda::memory_order_relaxed);
        }
        else {
            sem.Signal(cuda::memory_order_release);
            sem.Wait(cuda::memory_order_acquire);
        }
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
        if (relaxed) {  // It seems that fence is not needed on NVLink devices
            sem.Signal(cuda::memory_order_relaxed);
            sem.Wait(cuda::memory_order_relaxed);
        }
        else {
            sem.Signal(cuda::memory_order_release);
            sem.Wait(cuda::memory_order_acquire);
        }
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

    if (threadIdx.x < peers) {
        sem.Save(&semaphores[blockIdx.x * peers + threadIdx.x]);
    }
}

void CustomComm::AllreduceResidualBiasRMSnorm(void*        hidden,
                                              void*        residual,
                                              const void*  bias,
                                              const void*  weights,
                                              float        eps,
                                              int          dim,
                                              int          token_num,
                                              DataType     dtype,
                                              cudaStream_t stream)
{

    using T = half;

#if 1

    if (sizeof(T) * token_num * dim <= 1 << 30 && 0) {

        Array<LLPacket*, 8> outgoing;
        for (size_t i = 0; i < packet_chns_.size(); ++i) {
            outgoing[i] = (LLPacket*)packet_chns_[i].deviceHandle().dst_;
        }
        constexpr int vec_size      = sizeof(uint2) / sizeof(T);
        constexpr int ctas_per_peer = 1;

        const int vdim = dim / vec_size;

        const int slice1 = (token_num + world_size_ - 1) / world_size_;  // padded token slice
        const int count1 = token_num;                                    // actual token count

        const int slice0 = (token_num * vdim + world_size_ - 1) / world_size_;  // padded vec slice
        const int count0 = token_num * vdim;                                    // actual vec count

        // const int slice0 = slice1 * vdim;  // padded vec slice
        // const int count0 = count1 * vdim;  // actual vec count

        printf("slice0 %d, count0 %d, slice1 %d, count1 %d\n", slice0, count0, slice1, count1);

        const int peers = world_size_ - 1;

        constexpr int threads = 1024;
        const int     blocks  = ctas_per_peer * peers;

        AllreduceResidualBiasRMSnormKernel_LL<T, ctas_per_peer, vec_size, threads>
            <<<blocks, threads, 0, stream>>>((T*)hidden,
                                             (T*)residual,
                                             (const T*)bias,
                                             (const T*)weights,
                                             (LLPacket*)packet_buff_,
                                             outgoing,
                                             rank_,
                                             peers,
                                             slice0,
                                             count0,
                                             slice1,
                                             count1,
                                             vdim,
                                             1.f / dim,
                                             eps,
                                             flag_++);
    }
    else {
        auto&        peer_mems = registered_memories_.at(hidden);
        Array<T*, 8> peer_data{};
        for (size_t i = 0; i < peer_mems.size(); ++i) {
            peer_data[i] = (T*)peer_mems[i].data();
        }
        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        if constexpr (0) {
            const int slice = (token_num + world_size_ - 1) / world_size_ * (dim / vec_size);
            const int count = token_num * (dim / vec_size);
            AllreduceResidualBiasRMSnormKernel_Simple<<<32, 1024, 0, stream>>>((T*)hidden,
                                                                               (T*)residual,
                                                                               (const T*)bias,
                                                                               (const T*)weights,
                                                                               (Array<T*, 8>&)peer_data,
                                                                               device_semaphores_,
                                                                               rank_,
                                                                               world_size_ - 1,
                                                                               slice,
                                                                               count,
                                                                               1.f / dim,
                                                                               eps,
                                                                               constant<vec_size>{},
                                                                               constant<1024>{},
                                                                               std::true_type{});
        }
        else {
            const int     slice   = (token_num + world_size_ - 1) / world_size_;
            const int     count   = token_num;
            constexpr int threads = 1024;
            const int     blocks  = std::min(token_num, 32);
            AllreduceResidualBiasRMSnormKernel_Simple_v2<<<blocks, threads, 0, stream>>>((T*)hidden,
                                                                                         (T*)residual,
                                                                                         (const T*)bias,
                                                                                         (const T*)weights,
                                                                                         (Array<T*, 8>&)peer_data,
                                                                                         device_semaphores_,
                                                                                         rank_,
                                                                                         world_size_ - 1,
                                                                                         slice,
                                                                                         count,
                                                                                         dim / vec_size,
                                                                                         1.f / dim,
                                                                                         eps,
                                                                                         constant<vec_size>{},
                                                                                         constant<threads>{},
                                                                                         constant<true>{},
                                                                                         std::true_type{});
        }
    }
#else
    Comm::AllReduceSum((const T*)hidden, (T*)hidden, token_num * dim, stream);
    invokeBiasResidualRMSNorm((T*)residual, (T*)hidden, (const T*)weights, (const T*)bias, dim, token_num, eps, stream);
#endif
}

}  // namespace turbomind
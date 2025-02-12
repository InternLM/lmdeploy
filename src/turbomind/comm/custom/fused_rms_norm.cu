
#include <atomic>
#include <stdexcept>
#include <type_traits>

#include "cub/block/block_reduce.cuh"

#include "src/turbomind/comm/custom/custom_comm.h"

#include "src/turbomind/comm/custom/device_semaphore.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class T, int vec_size, int block_dim, bool aligned, class Relaxed>
__global__ void AllreduceResidualBiasRMSnormKernel_Simple_v2(T*                                             buf,
                                                             T*                                             res,
                                                             const T*                                       bias,
                                                             const T*                                       weights,
                                                             Array<T*, kMaxNearPeers>                       chns,
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

        for (int p = 0; p < peers - 1; ++p) {
            const int  peer = p + rank < peers ? p + rank : p + rank - peers;
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

        {  // last peer
            const int p    = peers - 1;
            const int peer = p + rank < peers ? p + rank : p + rank - peers;
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
            for (int ti = first + bi; ti < last; ti += block_num) {
                const int idx = (ti * vdim + di) * vec_size;
                Vec       vec;
                Load(vec, chn + idx);
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

template<class T, int vec_size, int block_dim, bool aligned, class Peers, class Relaxed>
__global__ void AllreduceResidualBiasRMSnormKernel_Simple_v3(T*                                             buf,
                                                             T*                                             res,
                                                             const T*                                       bias,
                                                             const T*                                       weights,
                                                             Array<T*, kMaxNearPeers>                       chns,
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

    auto invoke = [&](auto t) {
        using T                = decltype(t);
        const auto    near     = get_near((T*)hidden);
        const int     slice    = (token_num + world_size_ - 1) / world_size_;
        const int     count    = token_num;
        constexpr int threads  = 1024;
        const int     blocks   = std::min(token_num, 48);
        constexpr int vec_size = sizeof(uint4) / sizeof(T);
        AllreduceResidualBiasRMSnormKernel_Simple_v2<<<blocks, threads, 0, stream>>>((T*)hidden,
                                                                                     (T*)residual,
                                                                                     (const T*)bias,
                                                                                     (const T*)weights,
                                                                                     near,
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
                                                                                     constant<false>{},
                                                                                     std::true_type{});
    };

    const size_t elemsize = get_elem_size(dtype);
    const size_t bytesize = elemsize * token_num * dim;
    const int    vec_size = sizeof(uint4) / elemsize;

    if (dim % vec_size == 0 && bytesize > (1 << 20)) {
        switch (dtype) {
            case DataType::TYPE_FP16:
                return invoke(half{});
            case DataType::TYPE_BF16:
                return invoke(nv_bfloat16{});
            default:
                FT_CHECK(0);
        }
    };

    // fallback
    AllReduceSum(hidden, hidden, token_num * dim, dtype, stream);
    invokeResidualBiasRMSNorm(hidden, residual, weights, bias, dtype, dim, token_num, eps, stream);
}

}  // namespace turbomind
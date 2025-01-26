
#include <atomic>
#include <stdexcept>

#include "cub/block/block_reduce.cuh"

#include "src/turbomind/comm/common.h"

#include "src/turbomind/comm/custom/custom_comm.h"
#include "src/turbomind/comm/custom/device_semaphore.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"

namespace turbomind {

template<int vec_size, int block_dim, class T>
__global__ void AllreduceResidualBiasRMSnormKernel(T*                                             buf,
                                                   T*                                             res,
                                                   const T*                                       bias,
                                                   const T*                                       weights,
                                                   Array<T*, 8>                                   chns,
                                                   mscclpp::SmDevice2DeviceSemaphoreDeviceHandle* semaphores,
                                                   int                                            rank,
                                                   int                                            world_size,
                                                   size_t                                         count,
                                                   float                                          inv_dim,
                                                   float                                          eps)
{
    const int block_num  = gridDim.x;
    const int thread_num = blockDim.x * block_num;
    const int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int n_peer = world_size - 1;

    DeviceSemaphore sem;

    const int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id < n_peer) {
        sem.Load(&semaphores[blockIdx.x * n_peer + lane_id]);
    }
    __syncwarp();

    if (threadIdx.x < n_peer) {
        // It seems that fence is not needed on NVLink devices
        sem.Signal(cuda::memory_order_relaxed);
        sem.Wait(cuda::memory_order_relaxed);
    }
    __syncthreads();

    count /= vec_size * world_size;

    const int offset = rank * (int)count;

    using Vec = Array<T, vec_size>;

    using namespace ops;

    const int first = offset;
    const int last  = offset + (int)count;

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

    {
        const int p    = n_peer - 1;
        const int peer = p + rank < n_peer ? p + rank : p + rank - n_peer;
        auto      chn  = cvta_generic_to_global(chns[peer]);
        Vec       acc, tmp;
        for (int idx = first + thread_idx; idx < last; idx += thread_num) {
            Load(tmp, chn + idx * vec_size);
            Load(acc, buf + idx * vec_size);
            acc = acc + tmp;
            Vec r_vec;
            Vec x_vec;
            Load(r_vec, res + idx);
            r_vec = r_vec + acc;
            if (bias) {
                Ldg(x_vec, bias + threadIdx.x * vec_size);
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
            Ldg(x_vec, weights + threadIdx.x * vec_size);
            PRAGMA_UNROLL
            for (int i = 0; i < vec_size; ++i) {
                r_vec[i] = static_cast<T>(((float)r_vec[i] * sum)) * x_vec[i];
            }
            Store(buf + idx * vec_size, r_vec);
        }
    }

    __syncthreads();

    if (threadIdx.x < n_peer) {
        // It seems that fence is not needed on NVLink devices
        sem.Signal(cuda::memory_order_relaxed);
        sem.Wait(cuda::memory_order_relaxed);
    }

    __syncthreads();

    for (int p = 0; p < n_peer; ++p) {
        const int peer      = p + rank < n_peer ? p + rank : p + rank - n_peer;
        const int peer_rank = peer < rank ? peer : peer + 1;
        const int first     = (int)count * peer_rank;
        const int last      = first + (int)count;
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

    auto& peer_mems = registered_memories_.at(hidden);
    using T         = half;
    Array<T*, 8> peer_data{};
    for (size_t i = 0; i < peer_mems.size(); ++i) {
        peer_data[i] = (T*)peer_mems[i].data();
    }

    AllreduceResidualBiasRMSnormKernel<8, 1024><<<1, 1024, 0, stream>>>((T*)hidden,
                                                                        (T*)residual,
                                                                        (const T*)bias,
                                                                        (const T*)weights,
                                                                        (Array<T*, 8>&)peer_data,
                                                                        device_semaphores_,
                                                                        rank_,
                                                                        world_size_,
                                                                        (size_t)token_num * dim,
                                                                        1.f / dim,
                                                                        eps);
}

}  // namespace turbomind
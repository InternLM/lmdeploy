#include "quant_kernels.h"
#include "reduce_kernel_utils.cuh"
#include "src/turbomind/utils/cuda_type_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T>
__global__ void
int8_quant_kernel(const T* __restrict__ input, int8_t* __restrict__ out, float* scale, const int hidden_size)
{
    using T2             = typename TypeConverter<T>::Type;
    const int tid        = threadIdx.x;
    const int token_idx  = blockIdx.x;
    float     absmax_val = 0.0f;

    int16_t*  out_ptr   = (int16_t*)out;
    const T2* input_ptr = (const T2*)input;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        T2 val     = cuda_abs(input_ptr[token_idx * hidden_size + i]);
        T  val_max = cuda_max<T>(val);
        absmax_val = cuda_max(absmax_val, cuda_cast<float>(val_max));
    }

    const float      block_absmax_val_maybe = blockReduceMax(absmax_val);
    __shared__ float block_absmax_val;
    if (tid == 0) {
        block_absmax_val = block_absmax_val_maybe;
        scale[token_idx] = block_absmax_val / 127.0f;
    }
    __syncthreads();

    const float tmp_scale = 127.0f / block_absmax_val;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float2 val                           = cuda_cast<float2>(input_ptr[token_idx * hidden_size + i]);
        float2 tmp                           = val * tmp_scale;
        out_ptr[token_idx * hidden_size + i] = cuda_cast<int16_t>(tmp);
    }
}

template<>
__global__ void
int8_quant_kernel(const float* __restrict__ input, int8_t* __restrict__ out, float* scale, const int hidden_size)
{
    const int   tid        = threadIdx.x;
    const int   token_idx  = blockIdx.x;
    float       absmax_val = 0.0f;
    float const zero       = 0.0f;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        absmax_val = cuda_max(absmax_val, cuda_abs(input[token_idx * hidden_size + i]));
    }

    const float      block_absmax_val_maybe = blockReduceMax(absmax_val);
    __shared__ float block_absmax_val;
    if (tid == 0) {
        block_absmax_val = block_absmax_val_maybe;
        scale[token_idx] = block_absmax_val / 127.0f;
    }
    __syncthreads();

    const float tmp_scale = 127.0f / block_absmax_val;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        out[token_idx * hidden_size + i] = cuda_cast<int8_t>(input[token_idx * hidden_size + i] * tmp_scale);
    }
}

template<typename T>
void invokeI8Quant(const T* input, int8_t* out, float* scale, const int token_num, int hidden_size, cudaStream_t stream)
{
    if (sizeof(T) == 2) {
        FT_CHECK(hidden_size % 2 == 0);
        hidden_size /= 2;
    }
    dim3 grid(token_num);
    dim3 block(std::min(hidden_size, 1024));
    int8_quant_kernel<<<grid, block, 0, stream>>>(input, out, scale, hidden_size);
    sync_check_cuda_error();
}

#define INSTANTIATE_I8QUANT(T)                                                                                         \
    template void invokeI8Quant(                                                                                       \
        const T* input, int8_t* out, float* scale, const int token_num, const int hidden_size, cudaStream_t stream)

INSTANTIATE_I8QUANT(half);
#ifdef ENABLE_FP32
INSTANTIATE_I8QUANT(float);
#endif
#ifdef ENABLE_BF16
INSTANTIATE_I8QUANT(__nv_bfloat16);
#endif

}  // namespace turbomind

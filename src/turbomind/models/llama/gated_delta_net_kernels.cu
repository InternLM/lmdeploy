
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/models/llama/gated_delta_net_kernels.h"

#include <algorithm>
#include <cmath>

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

// =============================================================================
// Shared utility: block-level L2 norm inverse using warp + shared reduction
// Returns rsqrt(sum_sq + eps) visible to all threads in the block.
// smem must be at least ceil(blockDim.x/32) floats.
// SM70 specific optimization: bypass smem entirely if blockDim.x <= 32
// =============================================================================
__device__ __forceinline__ float block_l2_inv_norm(float partial_sq, float* smem, float eps = 1e-6f)
{
    // Warp reduce
    for (int mask = 16; mask > 0; mask >>= 1)
        partial_sq += __shfl_xor_sync(0xffffffff, partial_sq, mask);

    // Fast path: if the block is a single warp, we don't need shared memory
    if (blockDim.x <= 32) {
        return rsqrtf(partial_sq + eps);
    }

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane_id == 0)
        smem[warp_id] = partial_sq;
    __syncthreads();

    // First warp reduces across warps
    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem[lane_id] : 0.f;
        for (int mask = 16; mask > 0; mask >>= 1)
            val += __shfl_xor_sync(0xffffffff, val, mask);
        if (lane_id == 0)
            smem[0] = rsqrtf(val + eps);
    }
    __syncthreads();
    return smem[0];
}

// Helper to accumulate squares of a 16-bit type scalar or vector2 using float32 arithmetic
template<typename T>
__device__ __forceinline__ float sq_acc(T val)
{
    return (float)val * (float)val;
}

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ float sq_acc(half2 val)
{
    float2 fval = __half22float2(val);
    return fval.x * fval.x + fval.y * fval.y;
}
__device__ __forceinline__ float sq_acc(nv_bfloat162 val)
{
    float2 fval = __bfloat1622float2(val);
    return fval.x * fval.x + fval.y * fval.y;
}

__device__ __forceinline__ float2 to_float2(half2 v)
{
    return __half22float2(v);
}
__device__ __forceinline__ float2 to_float2(nv_bfloat162 v)
{
    return __bfloat1622float2(v);
}
__device__ __forceinline__ half2 to_vec2(float2 v, half)
{
    return __float22half2_rn(v);
}
__device__ __forceinline__ nv_bfloat162 to_vec2(float2 v, nv_bfloat16)
{
    return __float22bfloat162_rn(v);
}
#endif

// =============================================================================
// Causal Conv1d — Decode (seq_len == 1)
// =============================================================================
template<typename T>
__global__ void causal_conv1d_decode_kernel(
    T* out, const T* in, const T* weight, const T* bias, T* conv_states, int conv_dim, int d_conv)
{
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.y * conv_dim;
    if (idx >= total)
        return;

    const int b = idx / conv_dim;
    const int c = idx % conv_dim;

    T* state = conv_states + (b * conv_dim + c) * d_conv;

#pragma unroll
    for (int d = 0; d < d_conv - 1; ++d)
        state[d] = state[d + 1];
    state[d_conv - 1] = in[b * conv_dim + c];

    const T* w   = weight + c * d_conv;
    float    acc = 0.0f;
#pragma unroll
    for (int d = 0; d < d_conv; ++d)
        acc += static_cast<float>(state[d]) * static_cast<float>(w[d]);
    if (bias)
        acc += static_cast<float>(bias[c]);
    out[b * conv_dim + c] = static_cast<T>(acc / (1.0f + expf(-acc)));
}

// =============================================================================
// Causal Conv1d — Prefill (seq_len > 1)
// =============================================================================
template<typename T>
__global__ void causal_conv1d_prefill_kernel(
    T* out, const T* in, const T* weight, const T* bias, T* conv_states, int conv_dim, int seq_len, int d_conv)
{
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.y * conv_dim * seq_len;
    if (tid >= total)
        return;

    const int b   = tid / (conv_dim * seq_len);
    const int rem = tid % (conv_dim * seq_len);
    const int c   = rem / seq_len;
    const int t   = rem % seq_len;

    const T* w   = weight + c * d_conv;
    float    acc = 0.0f;
#pragma unroll
    for (int d = 0; d < d_conv; ++d) {
        int   src_t = t - (d_conv - 1 - d);
        float val   = 0.0f;
        if (src_t >= 0)
            val = static_cast<float>(in[(b * conv_dim + c) * seq_len + src_t]);
        acc += val * static_cast<float>(w[d]);
    }
    if (bias)
        acc += static_cast<float>(bias[c]);
    out[(b * conv_dim + c) * seq_len + t] = static_cast<T>(acc / (1.0f + expf(-acc)));

    if (conv_states && t >= seq_len - d_conv) {
        int state_idx                                        = d_conv - (seq_len - t);
        conv_states[(b * conv_dim + c) * d_conv + state_idx] = in[(b * conv_dim + c) * seq_len + t];
    }
}

template<typename T>
void invokeCausalConv1d(T*           out,
                        const T*     in,
                        const T*     weight,
                        const T*     bias,
                        T*           conv_states,
                        int          batch_size,
                        int          conv_dim,
                        int          seq_len,
                        int          d_conv,
                        cudaStream_t stream)
{
    if (seq_len == 1) {
        const int n       = batch_size * conv_dim;
        const int threads = 256;
        const int blocks  = (n + threads - 1) / threads;
        dim3      grid(blocks, batch_size);
        causal_conv1d_decode_kernel<<<grid, threads, 0, stream>>>(out, in, weight, bias, conv_states, conv_dim, d_conv);
    }
    else {
        const int n       = batch_size * conv_dim * seq_len;
        const int threads = 256;
        const int blocks  = (n + threads - 1) / threads;
        dim3      grid(blocks, batch_size);
        causal_conv1d_prefill_kernel<<<grid, threads, 0, stream>>>(
            out, in, weight, bias, conv_states, conv_dim, seq_len, d_conv);
    }
}

// =============================================================================
// Recurrent Gated Delta Rule (decode, seq_len == 1)
//
// Optimizations vs. original:
//  - Reads Q/K/V directly from the packed qkv_in buffer (stride = conv_dim),
//    eliminating three cudaMemcpy2DAsync strided-copy passes.
//  - L2-normalizes Q and K in-kernel using a shared-memory block reduction,
//    removing the separate invokeL2Norm kernel launches.
//  - GQA: maps v_head -> k_head = v_head / ratio inside the kernel,
//    removing the invokeRepeatInterleave allocation.
// =============================================================================
template<typename T>
__global__ void recurrent_delta_rule_kernel(T*       v_out,
                                            const T* qkv_in,
                                            const T* beta_in,
                                            const T* g_in,
                                            T*       state,
                                            int      num_v_heads,
                                            int      num_k_heads,
                                            int      key_head_dim,
                                            int      value_head_dim,
                                            int      k_dim_total)
{
    const int bh    = blockIdx.x;
    const int b     = bh / num_v_heads;
    const int h     = bh % num_v_heads;
    const int ratio = num_v_heads / num_k_heads;
    const int kh    = h / ratio;

    const int state_size = key_head_dim * value_head_dim;
    const int conv_dim   = 2 * k_dim_total + num_v_heads * value_head_dim;

    // Pointers into packed qkv for this batch element & head
    const T* q_ptr = qkv_in + b * conv_dim + kh * key_head_dim;
    const T* k_ptr = qkv_in + b * conv_dim + k_dim_total + kh * key_head_dim;
    const T* v_ptr = qkv_in + b * conv_dim + 2 * k_dim_total + h * value_head_dim;
    T*       s_ptr = state + (b * num_v_heads + h) * state_size;
    T*       o_ptr = v_out + (b * num_v_heads + h) * value_head_dim;

    const float beta_val = static_cast<float>(beta_in[b * num_v_heads + h]);
    const float decay    = expf(static_cast<float>(g_in[b * num_v_heads + h]));

    // Shared memory for block reductions (one slot per warp)
    __shared__ float smem[32];

    // --- In-kernel L2-normalize Q (Vectorized) ---
    float q_sq = 0.f;
    if (key_head_dim % 2 == 0) {
        using T2           = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;
        const T2* q_ptr_v2 = reinterpret_cast<const T2*>(q_ptr);
        for (int kd = threadIdx.x; kd < key_head_dim / 2; kd += blockDim.x)
            q_sq += sq_acc(q_ptr_v2[kd]);
    }
    else {
        for (int kd = threadIdx.x; kd < key_head_dim; kd += blockDim.x)
            q_sq += sq_acc(q_ptr[kd]);
    }
    const float q_inv_norm = block_l2_inv_norm(q_sq, smem);

    // --- In-kernel L2-normalize K (Vectorized) ---
    float k_sq = 0.f;
    if (key_head_dim % 2 == 0) {
        using T2           = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;
        const T2* k_ptr_v2 = reinterpret_cast<const T2*>(k_ptr);
        for (int kd = threadIdx.x; kd < key_head_dim / 2; kd += blockDim.x)
            k_sq += sq_acc(k_ptr_v2[kd]);
    }
    else {
        for (int kd = threadIdx.x; kd < key_head_dim; kd += blockDim.x)
            k_sq += sq_acc(k_ptr[kd]);
    }
    const float k_inv_norm = block_l2_inv_norm(k_sq, smem);

    // --- Step 1: S *= decay ---
    for (int idx = threadIdx.x; idx < state_size; idx += blockDim.x)
        s_ptr[idx] = static_cast<T>((float)s_ptr[idx] * decay);
    __syncthreads();

    // --- Step 2: delta rule update (each thread owns a slice of vd) ---
    for (int vd = threadIdx.x; vd < value_head_dim; vd += blockDim.x) {
        float kv_mem = 0.f;
        for (int kd = 0; kd < key_head_dim; ++kd)
            kv_mem += (float)s_ptr[kd * value_head_dim + vd] * ((float)k_ptr[kd] * k_inv_norm);

        const float delta = ((float)v_ptr[vd] - kv_mem) * beta_val;

        for (int kd = 0; kd < key_head_dim; ++kd)
            s_ptr[kd * value_head_dim + vd] =
                static_cast<T>((float)s_ptr[kd * value_head_dim + vd] + (float)k_ptr[kd] * k_inv_norm * delta);
    }
    __syncthreads();

    // --- Step 3: output = (S^T @ q) * scale ---
    const float scale = rsqrtf((float)key_head_dim);
    for (int vd = threadIdx.x; vd < value_head_dim; vd += blockDim.x) {
        float o = 0.f;
        for (int kd = 0; kd < key_head_dim; ++kd)
            o += (float)s_ptr[kd * value_head_dim + vd] * ((float)q_ptr[kd] * q_inv_norm);
        o_ptr[vd] = static_cast<T>(o * scale);
    }
}

template<typename T>
void invokeRecurrentGatedDeltaRule(T*           v_out,
                                   const T*     qkv_in,
                                   const T*     beta,
                                   const T*     g,
                                   T*           recurrent_state,
                                   int          batch_size,
                                   int          num_v_heads,
                                   int          num_k_heads,
                                   int          key_head_dim,
                                   int          value_head_dim,
                                   int          k_dim_total,
                                   cudaStream_t stream)
{
    const int num_blocks = batch_size * num_v_heads;
    if (num_blocks == 0)
        return;
    const int    threads = std::min(256, value_head_dim);
    const size_t smem_sz = ((threads + 31) / 32) * sizeof(float);
    recurrent_delta_rule_kernel<<<num_blocks, threads, smem_sz, stream>>>(
        v_out, qkv_in, beta, g, recurrent_state, num_v_heads, num_k_heads, key_head_dim, value_head_dim, k_dim_total);
}

// =============================================================================
// Single-Launch Prefill Gated Delta Rule (seq_len > 1)
//
// Optimizations vs. original invokeSerialGatedDeltaRule:
//  - The entire sequence is processed inside ONE kernel launch.
//    The original code called invokeRecurrentGatedDeltaRule in a host-side
//    for-loop, causing O(seq_len) kernel dispatches and CPU-GPU synchronization
//    round-trips that starve the GPU for every timestep.
//  - L2Norm, GQA handling, and packed qkv access are also fused in-kernel
//    (same improvements as the recurrent decode kernel above).
// =============================================================================
template<typename T>
__global__ void gated_delta_rule_prefill_kernel(T*       v_out,
                                                const T* qkv_in,
                                                const T* beta_in,
                                                const T* g_in,
                                                T*       state,
                                                int      seq_len,
                                                int      num_v_heads,
                                                int      num_k_heads,
                                                int      key_head_dim,
                                                int      value_head_dim,
                                                int      k_dim_total)
{
    // One block per v_head. Threads are distributed over value_head_dim.
    const int h     = blockIdx.x;
    const int ratio = num_v_heads / num_k_heads;
    const int kh    = h / ratio;  // GQA: k_head for this v_head

    const int state_size = key_head_dim * value_head_dim;
    const int conv_dim   = 2 * k_dim_total + num_v_heads * value_head_dim;
    const int v_dim      = num_v_heads * value_head_dim;

    T*          s_ptr = state + h * state_size;
    const float scale = rsqrtf((float)key_head_dim);

    __shared__ float smem[32];

    for (int t = 0; t < seq_len; ++t) {
        // Pointers into packed qkv (row-major per token)
        const T* q_ptr = qkv_in + t * conv_dim + kh * key_head_dim;
        const T* k_ptr = qkv_in + t * conv_dim + k_dim_total + kh * key_head_dim;
        const T* v_ptr = qkv_in + t * conv_dim + 2 * k_dim_total + h * value_head_dim;
        T*       o_ptr = v_out + t * v_dim + h * value_head_dim;

        const float beta_val = (float)beta_in[t * num_v_heads + h];
        const float decay    = expf((float)g_in[t * num_v_heads + h]);

        // --- In-kernel L2-normalize Q (Vectorized) ---
        float q_sq = 0.f;
        if (key_head_dim % 2 == 0) {
            using T2           = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;
            const T2* q_ptr_v2 = reinterpret_cast<const T2*>(q_ptr);
            for (int kd = threadIdx.x; kd < key_head_dim / 2; kd += blockDim.x)
                q_sq += sq_acc(q_ptr_v2[kd]);
        }
        else {
            for (int kd = threadIdx.x; kd < key_head_dim; kd += blockDim.x)
                q_sq += sq_acc(q_ptr[kd]);
        }
        const float q_inv_norm = block_l2_inv_norm(q_sq, smem);

        // --- In-kernel L2-normalize K (Vectorized) ---
        float k_sq = 0.f;
        if (key_head_dim % 2 == 0) {
            using T2           = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;
            const T2* k_ptr_v2 = reinterpret_cast<const T2*>(k_ptr);
            for (int kd = threadIdx.x; kd < key_head_dim / 2; kd += blockDim.x)
                k_sq += sq_acc(k_ptr_v2[kd]);
        }
        else {
            for (int kd = threadIdx.x; kd < key_head_dim; kd += blockDim.x)
                k_sq += sq_acc(k_ptr[kd]);
        }
        const float k_inv_norm = block_l2_inv_norm(k_sq, smem);

        // --- Step 1: S *= decay ---
        for (int idx = threadIdx.x; idx < state_size; idx += blockDim.x)
            s_ptr[idx] = static_cast<T>((float)s_ptr[idx] * decay);
        __syncthreads();

        // --- Step 2: delta rule update ---
        for (int vd = threadIdx.x; vd < value_head_dim; vd += blockDim.x) {
            float kv_mem = 0.f;
            for (int kd = 0; kd < key_head_dim; ++kd)
                kv_mem += (float)s_ptr[kd * value_head_dim + vd] * ((float)k_ptr[kd] * k_inv_norm);

            const float delta = ((float)v_ptr[vd] - kv_mem) * beta_val;

            for (int kd = 0; kd < key_head_dim; ++kd)
                s_ptr[kd * value_head_dim + vd] =
                    static_cast<T>((float)s_ptr[kd * value_head_dim + vd] + (float)k_ptr[kd] * k_inv_norm * delta);
        }
        __syncthreads();

        // --- Step 3: output = (S^T @ q) * scale ---
        for (int vd = threadIdx.x; vd < value_head_dim; vd += blockDim.x) {
            float o = 0.f;
            for (int kd = 0; kd < key_head_dim; ++kd)
                o += (float)s_ptr[kd * value_head_dim + vd] * ((float)q_ptr[kd] * q_inv_norm);
            o_ptr[vd] = static_cast<T>(o * scale);
        }
        __syncthreads();  // Ensure state write-back is visible before next step's decay
    }
    // State is updated in-place in global memory throughout the loop
}

template<typename T>
void invokeGatedDeltaRulePrefill(T*           v_out,
                                 const T*     qkv_in,
                                 const T*     beta,
                                 const T*     g,
                                 T*           recurrent_state,
                                 int          seq_len,
                                 int          num_v_heads,
                                 int          num_k_heads,
                                 int          key_head_dim,
                                 int          value_head_dim,
                                 int          k_dim_total,
                                 cudaStream_t stream)
{
    if (num_v_heads == 0 || seq_len == 0)
        return;
    // One block per v_head; threads cover value_head_dim
    const int    threads = std::min(256, value_head_dim);
    const size_t smem_sz = ((threads + 31) / 32) * sizeof(float);
    gated_delta_rule_prefill_kernel<<<num_v_heads, threads, smem_sz, stream>>>(v_out,
                                                                               qkv_in,
                                                                               beta,
                                                                               g,
                                                                               recurrent_state,
                                                                               seq_len,
                                                                               num_v_heads,
                                                                               num_k_heads,
                                                                               key_head_dim,
                                                                               value_head_dim,
                                                                               k_dim_total);
}

// =============================================================================
// Compute beta = sigmoid(b) and g = -exp(A_log) * softplus(a + dt_bias)
// =============================================================================
template<typename T>
__global__ void compute_beta_g_kernel(
    T* beta_out, T* g_out, const T* b_in, const T* a_in, const T* A_log, const T* dt_bias, int total, int num_v_heads)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
#if defined(__CUDA_ARCH__)
    if (total % 2 == 0 && num_v_heads % 2 == 0) {
        if (tid < total / 2) {
            using T2          = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;
            T2*       beta_v2 = reinterpret_cast<T2*>(beta_out);
            T2*       g_v2    = reinterpret_cast<T2*>(g_out);
            const T2* b_v2    = reinterpret_cast<const T2*>(b_in);
            const T2* a_v2    = reinterpret_cast<const T2*>(a_in);

            float2 b_val = to_float2(b_v2[tid]);
            float2 a_val = to_float2(a_v2[tid]);

            int h0 = (tid * 2) % num_v_heads;
            int h1 = (tid * 2 + 1) % num_v_heads;

            float Al0 = static_cast<float>(A_log[h0]);
            float dt0 = static_cast<float>(dt_bias[h0]);
            float Al1 = static_cast<float>(A_log[h1]);
            float dt1 = static_cast<float>(dt_bias[h1]);

            float beta0  = 1.0f / (1.0f + expf(-b_val.x));
            float sum0   = a_val.x + dt0;
            float sp0    = sum0 > 20.0f ? sum0 : logf(1.0f + expf(sum0));
            float g_val0 = -expf(Al0) * sp0;

            float beta1  = 1.0f / (1.0f + expf(-b_val.y));
            float sum1   = a_val.y + dt1;
            float sp1    = sum1 > 20.0f ? sum1 : logf(1.0f + expf(sum1));
            float g_val1 = -expf(Al1) * sp1;

            beta_v2[tid] = to_vec2(make_float2(beta0, beta1), T{});
            g_v2[tid]    = to_vec2(make_float2(g_val0, g_val1), T{});
        }
    }
    else
#endif
    {
        if (tid >= total)
            return;

        const int h           = tid % num_v_heads;
        float     b_val       = static_cast<float>(b_in[tid]);
        float     a_val       = static_cast<float>(a_in[tid]);
        float     A_log_val   = static_cast<float>(A_log[h]);
        float     dt_bias_val = static_cast<float>(dt_bias[h]);

        float beta  = 1.0f / (1.0f + expf(-b_val));
        float sum   = a_val + dt_bias_val;
        float sp    = sum > 20.0f ? sum : logf(1.0f + expf(sum));
        float g_val = -expf(A_log_val) * sp;

        beta_out[tid] = static_cast<T>(beta);
        g_out[tid]    = static_cast<T>(g_val);
    }
}

template<typename T>
void invokeComputeBetaG(T*           beta_out,
                        T*           g_out,
                        const T*     b_in,
                        const T*     a_in,
                        const T*     A_log,
                        const T*     dt_bias,
                        int          total,
                        int          num_v_heads,
                        cudaStream_t stream)
{
    const int threads = 256;
    if (total % 2 == 0 && num_v_heads % 2 == 0) {
        const int blocks = (total / 2 + threads - 1) / threads;
        compute_beta_g_kernel<<<blocks, threads, 0, stream>>>(
            beta_out, g_out, b_in, a_in, A_log, dt_bias, total, num_v_heads);
    }
    else {
        const int blocks = (total + threads - 1) / threads;
        compute_beta_g_kernel<<<blocks, threads, 0, stream>>>(
            beta_out, g_out, b_in, a_in, A_log, dt_bias, total, num_v_heads);
    }
}

template<class T>
__global__ void compute_beta_g_kernel_v2(T*       beta_out,
                                         T*       g_out,
                                         const T* b_in,
                                         int      b_stride,
                                         const T* a_in,
                                         int      a_stride,
                                         const T* A_log,
                                         const T* dt_bias,
                                         int      total,
                                         int      num_v_heads)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total)
        return;

    const int hi = idx % num_v_heads;
    const int ti = idx / num_v_heads;

    float b_val       = static_cast<float>(b_in[ti * b_stride + hi]);
    float a_val       = static_cast<float>(a_in[ti * a_stride + hi]);
    float A_log_val   = static_cast<float>(A_log[hi]);
    float dt_bias_val = static_cast<float>(dt_bias[hi]);

    float beta  = 1.0f / (1.0f + expf(-b_val));
    float sum   = a_val + dt_bias_val;
    float sp    = sum > 20.0f ? sum : logf(1.0f + expf(sum));
    float g_val = -expf(A_log_val) * sp;

    beta_out[idx] = static_cast<T>(beta);
    g_out[idx]    = static_cast<T>(g_val);
}

void ComputeBetaG_v2(Ref<Tensor>   beta_out_,
                     Ref<Tensor>   g_out_,
                     const Tensor& b_in,
                     const Tensor& a_in,
                     const Tensor& A_log,
                     const Tensor& dt_bias,
                     cudaStream_t  stream)
{

    auto& beta_out = beta_out_.get();
    auto& g_out    = g_out_.get();

    const int threads = 256;
    const int blocks  = cdiv<ssize_t>(beta_out.size(), threads);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        compute_beta_g_kernel_v2<<<blocks, threads, 0, stream>>>(beta_out.data<T>(),
                                                                 g_out.data<T>(),
                                                                 b_in.data<T>(),
                                                                 b_in.stride(0),
                                                                 a_in.data<T>(),
                                                                 a_in.stride(0),
                                                                 A_log.data<T>(),
                                                                 dt_bias.data<T>(),
                                                                 beta_out.size(),
                                                                 A_log.size());
    };

    TM_DISPATCH_PRIMARY_DTYPES(beta_out.dtype(), invoke);
}

// =============================================================================
// RMSNorm * SiLU-Gate (fused output normalization)
// =============================================================================
template<typename T>
__global__ void rms_norm_gated_kernel(
    T* hidden, const T* gate, const T* weight, float eps, int N, int head_dim, int gate_stride, int num_heads)
{
    const int row = blockIdx.x;
    if (row >= N)
        return;

    T*        h         = hidden + row * head_dim;
    const int token_idx = row / num_heads;
    const int head_idx  = row % num_heads;
    const T*  g         = gate + token_idx * gate_stride + head_idx * head_dim;

    __shared__ float smem[32];
    float            sum_sq = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = static_cast<float>(h[d]);
        sum_sq += val * val;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
    if ((threadIdx.x & 31) == 0)
        smem[threadIdx.x >> 5] = sum_sq;
    __syncthreads();
    if (threadIdx.x >> 5 == 0) {
        sum_sq = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[threadIdx.x] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
        if (threadIdx.x == 0)
            smem[0] = sum_sq;
    }
    __syncthreads();
    sum_sq = smem[0];

    float inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float h_val  = static_cast<float>(h[d]) * inv_rms * static_cast<float>(weight[d]);
        float g_val  = static_cast<float>(g[d]);
        float silu_g = g_val / (1.0f + expf(-g_val));
        h[d]         = static_cast<T>(h_val * silu_g);
    }
}

template<typename T>
void invokeRMSNormGated(T*           hidden,
                        const T*     gate,
                        const T*     weight,
                        float        eps,
                        int          N,
                        int          head_dim,
                        int          gate_stride,
                        int          num_heads,
                        cudaStream_t stream)
{
    const int threads = std::min(256, head_dim);
    rms_norm_gated_kernel<<<N, threads, 0, stream>>>(hidden, gate, weight, eps, N, head_dim, gate_stride, num_heads);
}

// =============================================================================
// Fused Conv1d + SiLU for row-major layout
// =============================================================================
template<typename T>
__global__ void fused_conv1d_decode_kernel(
    T* out, const T* in, const T* weight, const T* bias, T* state, int conv_dim, int d_conv, int in_stride)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim)
        return;

    T* s = state + c * d_conv;
#pragma unroll
    for (int d = 0; d < d_conv - 1; ++d)
        s[d] = s[d + 1];
    s[d_conv - 1] = in[c];

    const T* w   = weight + c * d_conv;
    float    acc = 0.0f;
#pragma unroll
    for (int d = 0; d < d_conv; ++d)
        acc += static_cast<float>(s[d]) * static_cast<float>(w[d]);
    if (bias)
        acc += static_cast<float>(bias[c]);
    out[c] = static_cast<T>(acc / (1.0f + expf(-acc)));
}

template<typename T>
__global__ void fused_conv1d_prefill_kernel(
    T* out, const T* in, const T* weight, const T* bias, T* state, int conv_dim, int seq_len, int d_conv, int in_stride)
{
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * conv_dim;
    if (tid >= total)
        return;

    const int t = tid / conv_dim;
    const int c = tid % conv_dim;

    const T* w   = weight + c * d_conv;
    float    acc = 0.0f;
#pragma unroll
    for (int d = 0; d < d_conv; ++d) {
        int   src_t = t - (d_conv - 1 - d);
        float val   = 0.0f;
        if (src_t >= 0)
            val = static_cast<float>(in[src_t * in_stride + c]);
        acc += val * static_cast<float>(w[d]);
    }
    if (bias)
        acc += static_cast<float>(bias[c]);
    out[t * conv_dim + c] = static_cast<T>(acc / (1.0f + expf(-acc)));

    if (state && t >= seq_len - d_conv) {
        int state_idx                 = d_conv - (seq_len - t);
        state[c * d_conv + state_idx] = in[t * in_stride + c];
    }
}

template<typename T>
void invokeFusedConv1dSiLU(T*           out,
                           const T*     in,
                           const T*     weight,
                           const T*     bias,
                           T*           conv_states,
                           int          batch_size,
                           int          conv_dim,
                           int          seq_len,
                           int          d_conv,
                           int          in_stride,
                           cudaStream_t stream)
{
    if (seq_len == 1) {
        const int threads = 256;
        const int blocks  = (conv_dim + threads - 1) / threads;
        fused_conv1d_decode_kernel<<<blocks, threads, 0, stream>>>(
            out, in, weight, bias, conv_states, conv_dim, d_conv, in_stride);
    }
    else {
        const int total   = seq_len * conv_dim;
        const int threads = 256;
        const int blocks  = (total + threads - 1) / threads;
        fused_conv1d_prefill_kernel<<<blocks, threads, 0, stream>>>(
            out, in, weight, bias, conv_states, conv_dim, seq_len, d_conv, in_stride);
    }
}

// =============================================================================
// Element-wise SiLU
// =============================================================================
template<typename T>
__global__ void silu_kernel(T* data, int n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
#if defined(__CUDA_ARCH__)
    if (n % 2 == 0) {
        if (tid < n / 2) {
            using T2       = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;
            T2*    data_v2 = reinterpret_cast<T2*>(data);
            float2 fval    = to_float2(data_v2[tid]);
            fval.x         = fval.x / (1.0f + expf(-fval.x));
            fval.y         = fval.y / (1.0f + expf(-fval.y));
            data_v2[tid]   = to_vec2(fval, T{});
        }
    }
    else
#endif
    {
        if (tid >= n)
            return;
        float x   = static_cast<float>(data[tid]);
        data[tid] = static_cast<T>(x / (1.0f + expf(-x)));
    }
}

template<typename T>
void invokeSiLU(T* data, int n, cudaStream_t stream)
{
    const int threads = 256;
    if (n % 2 == 0) {
        const int blocks = (n / 2 + threads - 1) / threads;
        silu_kernel<<<blocks, threads, 0, stream>>>(data, n);
    }
    else {
        const int blocks = (n + threads - 1) / threads;
        silu_kernel<<<blocks, threads, 0, stream>>>(data, n);
    }
}

// =============================================================================
// Explicit instantiations
// =============================================================================

#define INSTANTIATE(T)                                                                                                 \
    template void invokeCausalConv1d(T*, const T*, const T*, const T*, T*, int, int, int, int, cudaStream_t);          \
    template void invokeFusedConv1dSiLU(T*, const T*, const T*, const T*, T*, int, int, int, int, int, cudaStream_t);  \
    template void invokeRecurrentGatedDeltaRule(                                                                       \
        T*, const T*, const T*, const T*, T*, int, int, int, int, int, int, cudaStream_t);                             \
    template void invokeGatedDeltaRulePrefill(                                                                         \
        T*, const T*, const T*, const T*, T*, int, int, int, int, int, int, cudaStream_t);                             \
    template void invokeComputeBetaG(T*, T*, const T*, const T*, const T*, const T*, int, int, cudaStream_t);          \
    template void invokeRMSNormGated(T*, const T*, const T*, float, int, int, int, int, cudaStream_t);                 \
    template void invokeSiLU(T*, int, cudaStream_t);

INSTANTIATE(half)
INSTANTIATE(nv_bfloat16)

#undef INSTANTIATE

}  // namespace turbomind

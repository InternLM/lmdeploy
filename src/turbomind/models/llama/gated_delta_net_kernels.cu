#include "src/turbomind/models/llama/gated_delta_net_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <algorithm>
#include <cmath>

namespace turbomind {

// =============================================================================
// Causal Conv1d — Decode (seq_len == 1)
// =============================================================================
template<typename T>
__global__ void causal_conv1d_decode_kernel(
    T* out, const T* in, const T* weight, const T* bias, T* conv_states, int conv_dim, int d_conv)
{
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.y * conv_dim;  // batch_size stored in gridDim.y
    if (idx >= total)
        return;

    const int b = idx / conv_dim;
    const int c = idx % conv_dim;

    T* state = conv_states + (b * conv_dim + c) * d_conv;

    // Shift state left, insert new value
    for (int d = 0; d < d_conv - 1; ++d) {
        state[d] = state[d + 1];
    }
    state[d_conv - 1] = in[b * conv_dim + c];

    // Depthwise conv
    const T* w   = weight + c * d_conv;
    float    acc = 0.0f;
    for (int d = 0; d < d_conv; ++d) {
        acc += static_cast<float>(state[d]) * static_cast<float>(w[d]);
    }
    if (bias)
        acc += static_cast<float>(bias[c]);

    // SiLU
    out[b * conv_dim + c] = static_cast<T>(acc / (1.0f + expf(-acc)));
}

// =============================================================================
// Causal Conv1d — Prefill (seq_len > 1)
// =============================================================================
template<typename T>
__global__ void causal_conv1d_prefill_kernel(
    T* out, const T* in, const T* weight, const T* bias, T* conv_states, int conv_dim, int seq_len, int d_conv)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // gridDim.y = batch_size;  total = batch * conv_dim * seq_len
    const int total = gridDim.y * conv_dim * seq_len;
    if (tid >= total)
        return;

    const int b   = tid / (conv_dim * seq_len);
    const int rem = tid % (conv_dim * seq_len);
    const int c   = rem / seq_len;
    const int t   = rem % seq_len;

    const T* w   = weight + c * d_conv;
    float    acc = 0.0f;

    for (int d = 0; d < d_conv; ++d) {
        int   src_t = t - (d_conv - 1 - d);
        float val   = 0.0f;
        if (src_t >= 0) {
            val = static_cast<float>(in[(b * conv_dim + c) * seq_len + src_t]);
        }
        acc += val * static_cast<float>(w[d]);
    }
    if (bias)
        acc += static_cast<float>(bias[c]);

    // SiLU
    out[(b * conv_dim + c) * seq_len + t] = static_cast<T>(acc / (1.0f + expf(-acc)));

    // Save last d_conv inputs to conv_states for subsequent decode
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
// =============================================================================
template<typename T>
__global__ void recurrent_delta_rule_kernel(T*       v_out,
                                            const T* q,
                                            const T* k,
                                            const T* v,
                                            const T* beta,
                                            const T* g,
                                            T*       state,
                                            int      num_v_heads,
                                            int      key_head_dim,
                                            int      value_head_dim)
{
    const int bh         = blockIdx.x;  // batch * num_v_heads
    const int b          = bh / num_v_heads;
    const int h          = bh % num_v_heads;
    const int state_size = key_head_dim * value_head_dim;

    const T* q_ptr = q + (b * num_v_heads + h) * key_head_dim;
    const T* k_ptr = k + (b * num_v_heads + h) * key_head_dim;
    const T* v_ptr = v + (b * num_v_heads + h) * value_head_dim;
    T*       s_ptr = state + (b * num_v_heads + h) * state_size;
    T*       o_ptr = v_out + (b * num_v_heads + h) * value_head_dim;

    const float beta_val = static_cast<float>(beta[b * num_v_heads + h]);
    const float decay    = expf(static_cast<float>(g[b * num_v_heads + h]));

    // Step 1: S *= exp(g)
    for (int idx = threadIdx.x; idx < state_size; idx += blockDim.x) {
        s_ptr[idx] = static_cast<T>(static_cast<float>(s_ptr[idx]) * decay);
    }
    __syncthreads();

    // Step 2: kv_mem, delta, state update (each thread handles one v dimension)
    for (int vd = threadIdx.x; vd < value_head_dim; vd += blockDim.x) {
        float kv_mem = 0.0f;
        for (int kd = 0; kd < key_head_dim; ++kd) {
            kv_mem += static_cast<float>(s_ptr[kd * value_head_dim + vd]) * static_cast<float>(k_ptr[kd]);
        }
        float delta = (static_cast<float>(v_ptr[vd]) - kv_mem) * beta_val;
        for (int kd = 0; kd < key_head_dim; ++kd) {
            float s = static_cast<float>(s_ptr[kd * value_head_dim + vd]);
            s += static_cast<float>(k_ptr[kd]) * delta;
            s_ptr[kd * value_head_dim + vd] = static_cast<T>(s);
        }
    }
    __syncthreads();

    // Step 3: o[vd] = scale * sum_kd(S[kd, vd] * q[kd])
    // HF reference: query = query * (1 / sqrt(key_head_dim)) applied before delta rule
    const float scale = rsqrtf(static_cast<float>(key_head_dim));
    for (int vd = threadIdx.x; vd < value_head_dim; vd += blockDim.x) {
        float o = 0.0f;
        for (int kd = 0; kd < key_head_dim; ++kd) {
            o += static_cast<float>(s_ptr[kd * value_head_dim + vd]) * static_cast<float>(q_ptr[kd]);
        }
        o_ptr[vd] = static_cast<T>(o * scale);
    }
}

template<typename T>
void invokeRecurrentGatedDeltaRule(T*           v_out,
                                   const T*     q,
                                   const T*     k,
                                   const T*     v,
                                   const T*     beta,
                                   const T*     g,
                                   T*           recurrent_state,
                                   int          batch_size,
                                   int          num_v_heads,
                                   int          key_head_dim,
                                   int          value_head_dim,
                                   cudaStream_t stream)
{
    const int num_blocks = batch_size * num_v_heads;
    if (num_blocks == 0)
        return;
    const int threads = std::min(256, value_head_dim);
    recurrent_delta_rule_kernel<<<num_blocks, threads, 0, stream>>>(
        v_out, q, k, v, beta, g, recurrent_state, num_v_heads, key_head_dim, value_head_dim);
}

// =============================================================================
// Serial Gated Delta Rule (prefill, seq_len > 1)
// =============================================================================
template<typename T>
void invokeSerialGatedDeltaRule(T*           v_out,
                                const T*     q,
                                const T*     k,
                                const T*     v,
                                const T*     beta,
                                const T*     g,
                                T*           recurrent_state,
                                int          batch_size,
                                int          seq_len,
                                int          num_v_heads,
                                int          key_head_dim,
                                int          value_head_dim,
                                cudaStream_t stream)
{
    // Layout: (batch, seq_len, num_v_heads, dim)
    // but our data comes as (token_num, num_v_heads * dim) where token_num = batch * seq_len
    // Step through each timestep
    const int q_step  = num_v_heads * key_head_dim;
    const int v_step  = num_v_heads * value_head_dim;
    const int bg_step = num_v_heads;

    for (int t = 0; t < seq_len; ++t) {
        invokeRecurrentGatedDeltaRule(v_out + t * v_step,
                                      q + t * q_step,
                                      k + t * q_step,
                                      v + t * v_step,
                                      beta + t * bg_step,
                                      g + t * bg_step,
                                      recurrent_state,
                                      batch_size,
                                      num_v_heads,
                                      key_head_dim,
                                      value_head_dim,
                                      stream);
    }
}

// =============================================================================
// Compute beta = sigmoid(b) and g = -exp(A_log) * softplus(a + dt_bias)
// =============================================================================
template<typename T>
__global__ void compute_beta_g_kernel(
    T* beta_out, T* g_out, const T* b_in, const T* a_in, const T* A_log, const T* dt_bias, int total, int num_v_heads)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total)
        return;

    const int h           = tid % num_v_heads;
    float     b_val       = static_cast<float>(b_in[tid]);
    float     a_val       = static_cast<float>(a_in[tid]);
    float     A_log_val   = static_cast<float>(A_log[h]);
    float     dt_bias_val = static_cast<float>(dt_bias[h]);

    // beta = sigmoid(b)
    float beta = 1.0f / (1.0f + expf(-b_val));

    // g = -exp(A_log) * softplus(a + dt_bias)
    float sum   = a_val + dt_bias_val;
    float sp    = sum > 20.0f ? sum : logf(1.0f + expf(sum));
    float g_val = -expf(A_log_val) * sp;

    beta_out[tid] = static_cast<T>(beta);
    g_out[tid]    = static_cast<T>(g_val);
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
    const int blocks  = (total + threads - 1) / threads;
    compute_beta_g_kernel<<<blocks, threads, 0, stream>>>(
        beta_out, g_out, b_in, a_in, A_log, dt_bias, total, num_v_heads);
}

// =============================================================================
// L2 Norm along last dimension
// =============================================================================
template<typename T>
__global__ void l2_norm_kernel(T* x, int outer, int head_dim)
{
    const int idx = blockIdx.x;
    if (idx >= outer)
        return;

    T* row = x + idx * head_dim;

    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = static_cast<float>(row[d]);
        sum_sq += val * val;
    }

    // Warp reduction
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
    }
    __shared__ float shared[32];
    int              warp_id = threadIdx.x / 32;
    int              lane_id = threadIdx.x % 32;
    if (lane_id == 0)
        shared[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + 31) / 32) ? shared[lane_id] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
        }
        if (lane_id == 0)
            shared[0] = sum_sq;
    }
    __syncthreads();
    sum_sq = shared[0];

    float inv_norm = rsqrtf(sum_sq + 1e-6f);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        row[d] = static_cast<T>(static_cast<float>(row[d]) * inv_norm);
    }
}

template<typename T>
void invokeL2Norm(T* x, int outer, int head_dim, cudaStream_t stream)
{
    const int threads = std::min(256, head_dim);
    l2_norm_kernel<<<outer, threads, 0, stream>>>(x, outer, head_dim);
}

// =============================================================================
// Repeat-interleave from num_k_heads to num_v_heads
// =============================================================================
template<typename T>
__global__ void
repeat_interleave_kernel(T* dst, const T* src, int total, int num_k_heads, int num_v_heads, int head_dim)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total)
        return;

    const int d      = tid % head_dim;
    const int v_head = (tid / head_dim) % num_v_heads;
    const int outer  = tid / (num_v_heads * head_dim);

    const int ratio  = num_v_heads / num_k_heads;
    const int k_head = v_head / ratio;

    dst[tid] = src[(outer * num_k_heads + k_head) * head_dim + d];
}

template<typename T>
void invokeRepeatInterleave(
    T* dst, const T* src, int total, int num_k_heads, int num_v_heads, int head_dim, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    repeat_interleave_kernel<<<blocks, threads, 0, stream>>>(dst, src, total, num_k_heads, num_v_heads, head_dim);
}

// =============================================================================
// Fused RMSNorm * SiLU-Gate
// =============================================================================
template<typename T>
__global__ void rms_norm_gated_kernel(T* hidden, const T* gate, const T* weight, float eps, int N, int head_dim)
{
    const int row = blockIdx.x;
    if (row >= N)
        return;

    T*       h = hidden + row * head_dim;
    const T* g = gate + row * head_dim;

    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = static_cast<float>(h[d]);
        sum_sq += val * val;
    }

    // Warp reduction
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
    }
    __shared__ float shared[32];
    int              warp_id = threadIdx.x / 32;
    int              lane_id = threadIdx.x % 32;
    if (lane_id == 0)
        shared[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + 31) / 32) ? shared[lane_id] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
        }
        if (lane_id == 0)
            shared[0] = sum_sq;
    }
    __syncthreads();
    sum_sq = shared[0];

    float inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float h_val  = static_cast<float>(h[d]) * inv_rms * static_cast<float>(weight[d]);
        float g_val  = static_cast<float>(g[d]);
        float silu_g = g_val / (1.0f + expf(-g_val));
        h[d]         = static_cast<T>(h_val * silu_g);
    }
}

template<typename T>
void invokeRMSNormGated(T* hidden, const T* gate, const T* weight, float eps, int N, int head_dim, cudaStream_t stream)
{
    const int threads = std::min(256, head_dim);
    rms_norm_gated_kernel<<<N, threads, 0, stream>>>(hidden, gate, weight, eps, N, head_dim);
}

// =============================================================================
// Element-wise SiLU
// =============================================================================
template<typename T>
__global__ void silu_kernel(T* data, int n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    float x   = static_cast<float>(data[tid]);
    data[tid] = static_cast<T>(x / (1.0f + expf(-x)));
}

template<typename T>
void invokeSiLU(T* data, int n, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    silu_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

// =============================================================================
// Fused Conv1d + SiLU for row-major layout
// =============================================================================
// Input/output: (seq_len, conv_dim) — row-major, with batch_size=1 per call.
// For each output out[t, c], compute:
//   acc = sum_{d=0}^{d_conv-1} in[t - (d_conv-1-d), c] * weight[c, d]
//   out[t, c] = silu(acc + bias[c])
// where in[t, c] is zero if t < 0 (padding for causal boundary).

// Decode path: seq_len == 1
template<typename T>
__global__ void fused_conv1d_decode_kernel(T*       out,     // (1, conv_dim)
                                           const T* in,      // (1, conv_dim)
                                           const T* weight,  // (conv_dim, d_conv)
                                           const T* bias,
                                           T*       state,  // (conv_dim, d_conv)
                                           int      conv_dim,
                                           int      d_conv)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim)
        return;

    // Update state: shift left, insert new value
    T* s = state + c * d_conv;
    for (int d = 0; d < d_conv - 1; ++d) {
        s[d] = s[d + 1];
    }
    s[d_conv - 1] = in[c];  // in[0, c] in row-major

    // Depthwise conv
    const T* w   = weight + c * d_conv;
    float    acc = 0.0f;
    for (int d = 0; d < d_conv; ++d) {
        acc += static_cast<float>(s[d]) * static_cast<float>(w[d]);
    }
    if (bias)
        acc += static_cast<float>(bias[c]);

    out[c] = static_cast<T>(acc / (1.0f + expf(-acc)));
}

// Prefill path: seq_len > 1
// Each thread handles one (t, c) pair.
template<typename T>
__global__ void fused_conv1d_prefill_kernel(T*       out,     // (seq_len, conv_dim) row-major
                                            const T* in,      // (seq_len, conv_dim) row-major
                                            const T* weight,  // (conv_dim, d_conv)
                                            const T* bias,
                                            T*       state,  // (conv_dim, d_conv) or NULL
                                            int      conv_dim,
                                            int      seq_len,
                                            int      d_conv)
{
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * conv_dim;
    if (tid >= total)
        return;

    // Row-major: tid = t * conv_dim + c
    const int t = tid / conv_dim;
    const int c = tid % conv_dim;

    const T* w   = weight + c * d_conv;
    float    acc = 0.0f;

    for (int d = 0; d < d_conv; ++d) {
        int   src_t = t - (d_conv - 1 - d);  // causal: look backward
        float val   = 0.0f;
        if (src_t >= 0) {
            // Row-major: in[src_t, c] = in[src_t * conv_dim + c]
            val = static_cast<float>(in[src_t * conv_dim + c]);
        }
        acc += val * static_cast<float>(w[d]);
    }

    if (bias)
        acc += static_cast<float>(bias[c]);

    // SiLU
    out[t * conv_dim + c] = static_cast<T>(acc / (1.0f + expf(-acc)));

    // Save last d_conv inputs to state for subsequent decode
    if (state && t >= seq_len - d_conv) {
        int state_idx                 = d_conv - (seq_len - t);
        state[c * d_conv + state_idx] = in[t * conv_dim + c];
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
                           cudaStream_t stream)
{
    // This function processes one sequence at a time (batch_size is for the state offset).
    // The caller should invoke this per-request.
    if (seq_len == 1) {
        const int threads = 256;
        const int blocks  = (conv_dim + threads - 1) / threads;
        fused_conv1d_decode_kernel<<<blocks, threads, 0, stream>>>(
            out, in, weight, bias, conv_states, conv_dim, d_conv);
    }
    else {
        const int total   = seq_len * conv_dim;
        const int threads = 256;
        const int blocks  = (total + threads - 1) / threads;
        fused_conv1d_prefill_kernel<<<blocks, threads, 0, stream>>>(
            out, in, weight, bias, conv_states, conv_dim, seq_len, d_conv);
    }
}

// =============================================================================
// Explicit instantiations
// =============================================================================

#define INSTANTIATE(T)                                                                                                 \
    template void invokeCausalConv1d(T*, const T*, const T*, const T*, T*, int, int, int, int, cudaStream_t);          \
    template void invokeFusedConv1dSiLU(T*, const T*, const T*, const T*, T*, int, int, int, int, cudaStream_t);       \
    template void invokeRecurrentGatedDeltaRule(                                                                       \
        T*, const T*, const T*, const T*, const T*, const T*, T*, int, int, int, int, cudaStream_t);                   \
    template void invokeSerialGatedDeltaRule(                                                                          \
        T*, const T*, const T*, const T*, const T*, const T*, T*, int, int, int, int, int, cudaStream_t);              \
    template void invokeComputeBetaG(T*, T*, const T*, const T*, const T*, const T*, int, int, cudaStream_t);          \
    template void invokeL2Norm(T*, int, int, cudaStream_t);                                                            \
    template void invokeRepeatInterleave(T*, const T*, int, int, int, int, cudaStream_t);                              \
    template void invokeRMSNormGated(T*, const T*, const T*, float, int, int, cudaStream_t);                           \
    template void invokeSiLU(T*, int, cudaStream_t);

INSTANTIATE(half)
INSTANTIATE(nv_bfloat16)

#undef INSTANTIATE

}  // namespace turbomind

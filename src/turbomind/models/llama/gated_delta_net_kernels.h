#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

// Fused Conv1d + SiLU — unified batched launcher (row-major layout).
//
// Processes all requests in a single kernel launch.  Decode (seq_len == 1)
// and prefill (seq_len > 1) requests may be mixed freely within the batch.
//
// out:             (total_tokens, conv_dim)       row-major output
// in:              (total_tokens, in_stride)      non-contiguous slice of all_proj
// weight:          (d_conv, conv_dim)
// bias:            (conv_dim) or empty Tensor
// conv_state_ptrs: device array[batch_size] of per-request state pointers
// q_offsets:       device int[batch_size+1] cumulative token offsets
// k_offsets:       device int[batch_size+1] cumulative key (history+input) offsets
//
// Optional prefix-cache (pass nullptr for all three to disable): when capturing linear-attention
// prefix blocks, pass the same flat snapshot metadata as recurrent (local_end per si, batch
// offsets over full batch_size+1); ring state is written to staged_conv_ptrs[si] during the main
// forward (no segment replay).
void invokeFusedConv1dSiLU(Ref<Tensor>           out,
                           const Tensor&         in,
                           const Tensor&         weight,
                           const Tensor&         bias,
                           const Buffer_<void*>& conv_state_ptrs,
                           const Buffer_<int>&   q_offsets,
                           const Buffer_<int>&   k_offsets,
                           int                   batch_size,
                           int                   state_layer_offset,
                           int                   sm_count,
                           int*                  work_counter,
                           cudaStream_t          stream,
                           const Buffer_<int>*   snap_batch_offsets    = nullptr,
                           const Buffer_<int>*   snap_local_end_tokens = nullptr,
                           const Buffer_<void*>* staged_conv_ptrs      = nullptr);

// All three recurrent-rule launchers share the same trailing parameters for
// interface consistency:
//   sm_count      — multiprocessor count, queried once by the caller at init
//   work_counter  — device int* (1 element), owned by caller; v3 uses it for
//                   atomic workload claiming, v2/chunked ignore it
//   stream        — CUDA stream
//
// v2: standard one-block-per-(b,h) grid launch; sm_count and work_counter ignored.
void invokeGatedDeltaRuleBatched_v2(Ref<Tensor>           v_out,
                                    const Tensor&         qkv_in,
                                    const Tensor&         beta,
                                    const Tensor&         g,
                                    const Buffer_<void*>& state_ptrs,
                                    const Buffer_<int>&   q_offsets,
                                    int                   batch_size,
                                    int                   num_k_heads,
                                    int                   state_layer_offset,
                                    DataType              state_dtype,
                                    int                   sm_count,
                                    int*                  work_counter,
                                    cudaStream_t          stream);

// v3: persistent decode kernel, seq_len == 1 only.
// Launches min(total_work, blocks_per_sm * sm_count) blocks; each block claims
// work items atomically via work_counter (zeroed via cudaMemsetAsync per launch).
// state_dtype controls state precision: kFloat32 → S=float, otherwise S=T.
void invokeGatedDeltaRuleBatched_v3(Ref<Tensor>           v_out,
                                    const Tensor&         qkv_in,
                                    const Tensor&         beta,
                                    const Tensor&         g,
                                    const Buffer_<void*>& state_ptrs,
                                    const Buffer_<int>&   q_offsets,
                                    int                   batch_size,
                                    int                   num_k_heads,
                                    int                   state_layer_offset,
                                    DataType              state_dtype,
                                    int                   sm_count,
                                    int*                  work_counter,
                                    cudaStream_t          stream);

// =============================================================================
// Chunked Gated Delta Rule — for accelerating prefill
//
// Processes sequences in chunks of size C (default 64), parallelizing
// intra-chunk computation while maintaining sequential inter-chunk state
// updates. Reduces sequential depth from L to L/C.
//
// Same tensor layouts as invokeGatedDeltaRuleBatched_v2.
// sm_count and work_counter accepted for interface parity; ignored internally.
//
// Optional prefix-cache export (Gated Delta Net linear snapshots): when the three
// optional buffers are non-null, after each token whose local index within this prefill
// matches `snap_local_end_tokens[si]`, the recurrent state for this head is written to
// `staged_recurrent_ptrs[si] + h*D*D` (same swizzled layout as `state_ptrs`).
// `snap_batch_offsets` has length batch_size+1 and indexes the flat arrays.
void invokeChunkedGatedDeltaRuleBatched(Ref<Tensor>           v_out,
                                        const Tensor&         qkv_in,
                                        const Tensor&         beta,
                                        const Tensor&         g,
                                        const Buffer_<void*>& state_ptrs,
                                        const Buffer_<int>&   q_offsets,
                                        int                   batch_size,
                                        int                   num_k_heads,
                                        int                   state_layer_offset,
                                        DataType              state_dtype,
                                        int                   sm_count,
                                        int*                  work_counter,
                                        cudaStream_t          stream,
                                        const Buffer_<int>*   snap_batch_offsets    = nullptr,
                                        const Buffer_<int>*   snap_local_end_tokens = nullptr,
                                        const Buffer_<void*>* staged_recurrent_ptrs = nullptr);

// =============================================================================
// Helper kernels
// =============================================================================

void ComputeBetaG_v2(Ref<Tensor>   beta_out_,
                     Ref<Tensor>   g_out_,
                     const Tensor& b_in,
                     const Tensor& a_in,
                     const Tensor& A_log,
                     const Tensor& dt_bias,
                     cudaStream_t  stream);

// RMSNorm * SiLU-gate (fused output normalization)
void invokeRMSNormGated(Ref<Tensor> hidden, const Tensor& gate, const Tensor& weight, float eps, cudaStream_t stream);

}  // namespace turbomind

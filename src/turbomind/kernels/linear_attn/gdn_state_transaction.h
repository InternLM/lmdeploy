// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/gemm/types.h"

#include <cuda_runtime.h>

namespace turbomind::linear_attn::delta_rule {

using core::Buffer_;
using core::Tensor;

struct TransitionJournal {
    Tensor raw_conv;
    Tensor key;
    Tensor value;
    Tensor log_decay;
    Tensor beta;
};

void invokeBuildGdnStateStoreMask(bool*        suppress_state_store,
                                  const bool*  finished,
                                  const bool*  speculative_row,
                                  int          request_count,
                                  cudaStream_t stream);

void invokeCaptureGdnTransitions(const Tensor&       raw_projection,
                                 const Tensor&       normalized_key,
                                 const Tensor&       value,
                                 const Tensor&       log_decay,
                                 const Tensor&       beta,
                                 const Buffer_<int>& q_offsets,
                                 const Buffer_<int>& speculative_request_indices,
                                 int                 gdn_layer,
                                 int                 verify_positions,
                                 TransitionJournal   journal,
                                 cudaStream_t        stream);

void invokeCommitAcceptedConvState(const Tensor&         raw_conv,
                                   const Buffer_<void*>& conv_state_ptrs,
                                   const Buffer_<int>&   speculative_request_indices,
                                   const Buffer_<int>&   entry_sequence_length,
                                   const Buffer_<int>&   accept_len,
                                   const Buffer_<int>&   conv_state_offsets,
                                   int                   conv_dim,
                                   int                   d_conv,
                                   cudaStream_t          stream);

struct AcceptedPrefixArguments {
    Tensor key;
    Tensor value;
    Tensor log_decay;
    Tensor beta;
    Tensor recurrent_state_ptrs;
    Tensor request_indices;
    Tensor accept_len;
    int    layer_count{};
    int    speculative_count{};
    int    position_count{};
    int    hq{};
    int    hv{};
    int    num_head_groups{};
    int    layers_per_block{};
    int    heads_per_block{};
    int    sm_count{};
};

void invokeCommitAcceptedRecurrentState(
    const AcceptedPrefixArguments& args, DataType state_dtype, cudaStream_t stream);

}  // namespace turbomind::linear_attn::delta_rule

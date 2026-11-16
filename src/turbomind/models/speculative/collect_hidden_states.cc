// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/collect_hidden_states.h"

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/padded_row_allgather.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

CollectHiddenStates::CollectHiddenStates(const EngineParam& engine,
                                         const Context&     context,
                                         int                phases,
                                         int                capture_width,
                                         int                hidden_units,
                                         DataType           data_type):
    capture_width_{capture_width},
    hidden_units_{hidden_units},
    capacity_{cdiv(engine.max_forward_token_num, engine.attn_cp_size * engine.attn_tp_size)},
    attn_dp_rank_{engine.attn_dp_rank},
    model_tp_group_{context.comm.d_tp_group},
    model_tp_rank_{engine.model_tp_rank},
    model_tp_size_{engine.attn_cp_size * engine.attn_tp_size},
    d_comm_{context.comm.d_comm},
    data_(phases)
{
    Allocator symmetric_allocator;
    if (model_tp_size_ > 1) {
        symmetric_allocator = GetSymmAllocator(context.comm.d_comm);
    }
    for (Data& data : data_) {
        data.captured = Tensor{{capacity_, capture_width_}, data_type, kDEVICE};
        if (model_tp_size_ > 1) {
            data.gathered_padded = {
                {model_tp_size_ * capacity_, hidden_units_}, data_type, symmetric_allocator};
        }
    }
}

void CollectHiddenStates::Begin(int phase, const std::vector<int>& local_token_nums)
{
    Data& data = data_.at(phase);

    const int global_rank   = d_comm_ ? d_comm_->rank(0) : 0;
    const int global_size   = d_comm_ ? d_comm_->n_ranks(0) : 1;
    const int model_tp_size = d_comm_ ? d_comm_->n_ranks(model_tp_group_) : 1;
    data.owned     = comm::ComputeTokenOwnership(global_rank, global_size, model_tp_size, local_token_nums.data());
    data.token_num = local_token_nums[attn_dp_rank_];
}

void CollectHiddenStates::SeedWarmup(int phase, cudaStream_t stream)
{
    Data&     data = data_.at(phase);
    const int rows  = data.owned.row_count();
    if (rows > 0) {
        TM_CUDA_CHECK(cudaMemsetAsync(
            data.captured.raw_data(), 0, byte_size(data.captured.dtype(), size_t(rows) * capture_width_), stream));
    }
}

Tensor CollectHiddenStates::Gather(int phase, const Tensor& local_buffer, cudaStream_t stream)
{
    Data&     data = data_.at(phase);
    const int rows  = data.owned.row_count();
    if (model_tp_size_ == 1) {
        return local_buffer.slice({0, 0}, {rows, local_buffer.shape(1)});
    }
    return PaddedRowAllGather(local_buffer,
                              data.gathered_padded,
                              data.token_num,
                              rows,
                              model_tp_rank_,
                              model_tp_size_,
                              *d_comm_,
                              model_tp_group_,
                              stream);
}

}  // namespace turbomind

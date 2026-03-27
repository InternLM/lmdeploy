// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/nccl/nccl_comm.h"

#include "src/turbomind/comm/nccl/deep_ep/deep_ep.hpp"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/gemm/moe_ep_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstdio>
#include <numeric>

namespace turbomind::comm {

void NcclCommImpl::InitializeEp(const EpConfig& config)
{
    TM_LOG_DEBUG("[NCCLEP][%d] Initialize", h_comm_->rank());

    // Check NCCL version
    int version{};
    ncclGetVersion(&version);
    TM_CHECK_GE(version, NCCL_VERSION(2, 29, 7));
    ep_config_ = config;

    const int num_rdma_bytes = config.num_nodes > 1 ? int(1e9) : 0;
    const int num_ll_rdma_bytes =
        config.ll_max_tokens_per_rank > 0 ?
            deep_ep ::get_low_latency_rdma_size_hint(
                config.ll_max_tokens_per_rank, config.hidden, h_comm_->n_ranks(), config.num_experts) :
            0;

    const int num_local_experts = config.num_experts / h_comm_->n_ranks();
    const int num_sms           = 24;
    const int qps_per_rank      = (config.num_nodes == 1) ? num_local_experts : std::max(num_sms, num_local_experts);

    buffer_ = std::make_unique<deep_ep::Buffer>(  //
        h_comm_->rank(),
        h_comm_->n_ranks(),
        int(2e9),
        num_rdma_bytes,
        num_ll_rdma_bytes,
        true,
        false,
        false,
        qps_per_rank,
        h_comm_);
}

void NcclCommImpl::Dispatch(const EpDispatchInput& input, EpDispatchOutput& output, int group)
{
    TM_CHECK_EQ(group, 0);
    TM_CHECK_NE(input.mode, EpMode::kNull);

    if (input.mode == EpMode::kLowLatency) {
        auto [packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range] =
            buffer_->low_latency_dispatch(input.x,
                                          input.topk_idx,
                                          std::nullopt,
                                          std::nullopt,
                                          ep_config_.ll_max_tokens_per_rank,
                                          ep_config_.num_experts,
                                          false,
                                          false,
                                          false);
        sync_check_cuda_error();

        const int num_local_experts = ep_config_.num_experts / h_comm_->n_ranks();

        auto st = core::Context::stream().handle();

        // Compute offsets
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr,
                                      temp_storage_bytes,
                                      packed_recv_count.data<int>(),
                                      output.offsets.data() + 1,
                                      num_local_experts,
                                      st);
        Buffer_<uint8_t> temp_storage(temp_storage_bytes, kDEVICE);
        cub::DeviceScan::InclusiveSum(temp_storage.raw_data(),
                                      temp_storage_bytes,
                                      packed_recv_count.data<int>(),
                                      output.offsets.data() + 1,
                                      num_local_experts,
                                      st);
        sync_check_cuda_error();

        // Compute f2n, f2E
        invokeMoeLLDispatchPostprocess(output.out_x,
                                       output.f2n.data(),
                                       output.f2E.data(),
                                       output.offsets.data(),
                                       buffer_->moe_recv_counter,
                                       buffer_->moe_recv_counter_mapped,
                                       packed_recv_x,
                                       st);
        sync_check_cuda_error();

        // Generate output
        output.handle        = {packed_recv_src_info, packed_recv_layout_range, output.offsets};
        output.out_token_num = output.out_expert_token_num = *buffer_->moe_recv_counter;
    }
    else {
        auto [num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank] =
            buffer_->get_dispatch_layout(input.topk_idx, ep_config_.num_experts);
        sync_check_cuda_error();

        if (buffer_->get_num_rdma_ranks() > 1) {
            // internode dispatch
        }
        else {
            auto config      = buffer_->get_dispatch_config();
            auto [recv_x,
                  recv_x_scales,
                  recv_topk_idx,
                  recv_topk_weights,
                  num_recv_tokens_per_expert_list,
                  num_recv_tokens_per_expert,
                  rank_prefix_matrix,
                  channel_prefix_matrix,
                  recv_channel_prefix_matrix,
                  recv_src_idx,
                  send_head] = buffer_->intranode_dispatch(input.x,
                                                           std::nullopt,
                                                           input.topk_idx,
                                                           input.topk_weights,
                                                           num_tokens_per_rank,
                                                           is_token_in_rank,
                                                           num_tokens_per_expert,
                                                           0,
                                                           std::nullopt,
                                                           std::nullopt,
                                                           1,
                                                           0,
                                                           config);
            sync_check_cuda_error();

            // Generate output
            output.handle           = {rank_prefix_matrix,
                                       channel_prefix_matrix,
                                       recv_channel_prefix_matrix,
                                       recv_src_idx,
                                       is_token_in_rank,
                                       send_head};
            output.out_x            = recv_x;
            output.out_topk_weights = recv_topk_weights.value();
            output.out_token_num    = recv_x.shape(0);
            output.out_expert_token_num =
                std::accumulate(num_recv_tokens_per_expert_list.begin(), num_recv_tokens_per_expert_list.end(), 0);

            const int num_local_experts = num_recv_tokens_per_expert_list.size();
            const int topk              = input.topk_idx.shape(1);
            const int num_recv_tokens   = recv_x.shape(0);
            auto      st                = core::Context::stream().handle();

            // Compute offsets
            size_t temp_storage_bytes = 0;
            cub::DeviceScan::InclusiveSum(nullptr,
                                          temp_storage_bytes,
                                          num_recv_tokens_per_expert.data<int>(),
                                          output.offsets.data() + 1,
                                          num_local_experts,
                                          st);
            Buffer_<uint8_t> temp_storage(temp_storage_bytes, kDEVICE);
            cub::DeviceScan::InclusiveSum(temp_storage.raw_data(),
                                          temp_storage_bytes,
                                          num_recv_tokens_per_expert.data<int>(),
                                          output.offsets.data() + 1,
                                          num_local_experts,
                                          st);
            sync_check_cuda_error();

            // Compute f2n, f2E, en2f
            turbomind::invokeMoeRoutingMapEp(output.f2n.data(),
                                             output.f2E.data(),
                                             output.en2f.data(),
                                             output.offsets.data(),
                                             recv_topk_idx->data_or((int64_t*)nullptr),
                                             num_recv_tokens,
                                             topk,
                                             num_local_experts,
                                             st);
            sync_check_cuda_error();
        }
    }
}

void NcclCommImpl::Combine(const EpCombineInput& input, EpCombineOutput& output, int group)
{
    TM_CHECK_EQ(group, 0);
    TM_CHECK_NE(input.mode, EpMode::kNull);

    if (input.mode == EpMode::kLowLatency) {
        const int   num_local_experts = ep_config_.num_experts / h_comm_->n_ranks();
        const auto& offsets           = input.handle[2];
        const int   num_max_tokens    = ep_config_.ll_max_tokens_per_rank * h_comm_->n_ranks();
        auto        sparse_x = Tensor({num_local_experts, num_max_tokens, ep_config_.hidden}, input.x.dtype(), kDEVICE);

        // convert dense input to sparse
        auto st = core::Context::stream().handle();
        invokeMoeLLCombinePreprocess(sparse_x, input.x, offsets.data<int>(), st);
        sync_check_cuda_error();

        auto& packed_recv_src_info     = input.handle[0];
        auto& packed_recv_layout_range = input.handle[1];
        auto [combined_x]              = buffer_->low_latency_combine(sparse_x,
                                                         input.topk_idx.value(),
                                                         input.topk_weights.value(),
                                                         packed_recv_src_info,
                                                         packed_recv_layout_range,
                                                         std::nullopt,
                                                         ep_config_.ll_max_tokens_per_rank,
                                                         ep_config_.num_experts,
                                                         false,
                                                         false,
                                                         std::nullopt);
        sync_check_cuda_error();

        // Generate output
        output.out_x = combined_x;
    }
    else {
        if (buffer_->get_num_rdma_ranks() > 1) {
            // internode combine
        }
        else {
            // intranode combine
            auto config = buffer_->get_combine_config();
            TM_CHECK(input.handle.size() == 6);
            auto rank_prefix_matrix    = input.handle[0];
            auto channel_prefix_matrix = input.handle[2];
            auto src_idx               = input.handle[3];
            auto send_head             = input.handle[5];

            auto [recv_x, recv_topk_weights] = buffer_->intranode_combine(input.x,
                                                                          input.topk_weights,
                                                                          std::nullopt,
                                                                          std::nullopt,
                                                                          src_idx,
                                                                          rank_prefix_matrix,
                                                                          channel_prefix_matrix,
                                                                          send_head,
                                                                          config);
            sync_check_cuda_error();
            output.out_x = recv_x;
        }
    }
}

}  // namespace turbomind::comm

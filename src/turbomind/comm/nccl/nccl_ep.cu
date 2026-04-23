// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/nccl/nccl_comm.h"

#include "3rdparty/deep_ep/deep_ep.hpp"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/gemm/moe_ep_utils.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/kernels/quantization.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstdio>

namespace turbomind::comm {

void NcclCommImpl::InitializeEp(const EpConfig& config)
{
    TM_LOG_DEBUG("[NCCLEP][{}] Initialize", h_comm_->rank());

    // Check NCCL version
    int version{};
    ncclGetVersion(&version);
    TM_CHECK_GE(version, NCCL_VERSION(2, 29, 7));
    ep_config_ = config;

    const int num_rdma_bytes    = config.num_nodes > 1 ? int(1e9) : 0;
    const int num_ll_rdma_bytes = [&]() -> int {
        if (config.ll_max_tokens_per_rank > 0) {
            return deep_ep::get_low_latency_rdma_size_hint(
                config.ll_max_tokens_per_rank, config.hidden, h_comm_->n_ranks(), config.num_experts);
        }
        return 0;
    }();

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

    temp_storage_bytes_ = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes_, (int*)nullptr, (int*)nullptr, num_local_experts, 0);
    temp_storage_ = core::Buffer_<uint8_t>(temp_storage_bytes_, kDEVICE);
}

void NcclCommImpl::Dispatch(const EpDispatchInput& input, EpDispatchOutput& output, int group)
{
    TM_CHECK_EQ(group, 0);
    TM_CHECK(input.mode != EpMode::kNull);

    const int num_local_experts = ep_config_.num_experts / h_comm_->n_ranks();
    auto      st                = core::Context::stream().handle();

    if (input.mode == EpMode::kLowLatency) {
        auto [packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range] =
            buffer_->low_latency_dispatch(input.x,
                                          input.topk_idx,
                                          std::nullopt,
                                          std::nullopt,
                                          ep_config_.ll_max_tokens_per_rank,
                                          ep_config_.num_experts,
                                          input.use_fp8,
                                          false,
                                          false);
        sync_check_cuda_error();
        // Compute offsets
        cub::DeviceScan::InclusiveSum(temp_storage_.raw_data(),
                                      temp_storage_bytes_,
                                      packed_recv_count.data<int>(),
                                      output.offsets.data() + 1,
                                      num_local_experts,
                                      st);
        sync_check_cuda_error();

        // Compute f2n, f2E (f2n points into the flattened sparse packed_recv_x)
        invokeMoeLLDispatchPostprocess(output.f2n.data(), output.f2E.data(), output.offsets.data(), packed_recv_x, st);
        sync_check_cuda_error();

        // Expose the sparse buffer as a flat 2D view; downstream linear gathers via f2n.
        Tensor sparse_out_x = packed_recv_x.view({-1, packed_recv_x.shape().back()});
        TM_CHECK_EQ(sparse_out_x.shape(0), input.num_worst_tokens);

        // Reorder sparse scales into [H/128, E*max_T] sparse layout, writing only the
        // valid prefix of each expert; gaps stay uninitialized and are never read.
        if (input.use_fp8) {
            const int num_groups = packed_recv_x_scales->shape(2);
            Tensor    out_scales{{num_groups, input.num_worst_tokens}, kFloat32, kDEVICE};
            invokeMoeLLDispatchScalesLayoutConvert(out_scales, packed_recv_x_scales.value(), packed_recv_count, st);
            sync_check_cuda_error();
            if (input.output_scales) {
                output.out_x        = sparse_out_x;
                output.out_x_scales = out_scales;
            }
            else {
                const int* total_ptr = output.offsets.data() + num_local_experts;
                Tensor     indices{output.f2n.slice(0, input.num_worst_tokens)};
                DequantizeSymm(output.out_x, sparse_out_x, out_scales, indices, total_ptr, st);
                sync_check_cuda_error();
            }
        }
        else {
            output.out_x = sparse_out_x;
        }

        // Generate output
        output.handle = {packed_recv_src_info, packed_recv_layout_range, output.offsets};

        if (input.zero_copy) {
            output.rdma = buffer_->get_next_low_latency_combine_buffer(
                ep_config_.ll_max_tokens_per_rank, ep_config_.hidden, ep_config_.num_experts);
        }
    }
    else {
        auto [num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank] =
            buffer_->get_dispatch_layout(input.topk_idx, ep_config_.num_experts);
        sync_check_cuda_error();

        auto Postprocess = [&](Tensor&                recv_x,
                               std::optional<Tensor>& recv_x_scales,
                               Tensor&                recv_topk_weights,
                               Tensor&                recv_topk_idx,
                               Tensor&                num_recv_tokens_per_expert,
                               const int*             recv_token_num_ptr) {
            // Compute offsets
            cub::DeviceScan::InclusiveSum(temp_storage_.raw_data(),
                                          temp_storage_bytes_,
                                          num_recv_tokens_per_expert.data<int>(),
                                          output.offsets.data() + 1,
                                          num_local_experts,
                                          st);
            sync_check_cuda_error();

            if (input.use_fp8) {
                auto&  scales_t = recv_x_scales.value();
                Tensor x_scales = Tensor{{scales_t.shape(1), scales_t.shape(0)}, scales_t.dtype(), scales_t.device()};
                if (scales_t.shape(0) > 0) {
                    invokeTransposeAxis01(
                        x_scales.data<float>(), scales_t.data<float>(), scales_t.shape(0), scales_t.shape(1), 1, st);
                }
                if (input.output_scales) {
                    output.out_x        = recv_x;
                    output.out_x_scales = x_scales;
                }
                else {
                    DequantizeSymm(output.out_x, recv_x, x_scales, Tensor{}, recv_token_num_ptr, st);
                }
            }
            else {
                output.out_x = recv_x;
            }
            const int topk = input.topk_idx.shape(1);

            output.out_topk_weights        = recv_topk_weights;
            output.num_distinct_tokens_ptr = recv_token_num_ptr;

            // Build the recv-token -> expert-token routing map. The device-side limit here
            // must be the real recv-token count, because `recv_topk_idx` is token-major.
            // `offsets.back()` is the real flattened expert-token total, not the distinct
            // received-token total in `recv_x/recv_topk_*`.
            turbomind::invokeMoeRoutingMapEp(output.f2n.data(),
                                             output.f2E.data(),
                                             output.en2f.data(),
                                             output.offsets.data(),
                                             recv_topk_idx.data_or((int64_t*)nullptr),
                                             recv_token_num_ptr,
                                             recv_x.shape(0),
                                             topk,
                                             num_local_experts,
                                             st);
            sync_check_cuda_error();
        };

        if (buffer_->get_num_rdma_ranks() > 1) {
            // internode dispatch
            Tensor                x = input.x;
            std::optional<Tensor> x_scales;
            if (input.use_fp8) {
                x        = {};
                x_scales = Tensor{};
                QuantizeSymm(x, x_scales.value(), input.x, core::Context::stream().handle());
                x_scales = x_scales->transpose(0, 1);
            }
            auto config          = buffer_->get_dispatch_config();
            auto [recv_x,
                  recv_x_scales,
                  recv_topk_idx,
                  recv_topk_weights,
                  num_recv_tokens_per_expert_list,
                  num_recv_tokens_per_expert,
                  rdma_channel_prefix_matrix,
                  gbl_channel_prefix_matrix,
                  recv_rdma_channel_prefix_matrix,
                  recv_rdma_rank_prefix_sum,
                  recv_gbl_channel_prefix_matrix,
                  recv_gbl_rank_prefix_sum,
                  recv_src_meta,
                  send_rdma_head,
                  send_nvl_head] = buffer_->internode_dispatch(x,
                                                               x_scales,
                                                               input.topk_idx,
                                                               input.topk_weights,
                                                               num_tokens_per_rank,
                                                               num_tokens_per_rdma_rank,
                                                               is_token_in_rank,
                                                               num_tokens_per_expert,
                                                               0,
                                                               0,
                                                               std::nullopt,
                                                               std::nullopt,
                                                               std::nullopt,
                                                               std::nullopt,
                                                               1,
                                                               input.num_worst_tokens,
                                                               config);
            sync_check_cuda_error();

            // Generate output
            output.handle = {is_token_in_rank,
                             rdma_channel_prefix_matrix,
                             gbl_channel_prefix_matrix,
                             recv_rdma_channel_prefix_matrix.value(),
                             recv_rdma_rank_prefix_sum,
                             recv_gbl_channel_prefix_matrix.value(),
                             recv_gbl_rank_prefix_sum,
                             recv_src_meta.value(),
                             send_rdma_head.value(),
                             send_nvl_head.value()};

            const int* recv_token_num_ptr = recv_gbl_rank_prefix_sum.data<int>() + h_comm_->n_ranks() - 1;
            Postprocess(recv_x,  //
                        recv_x_scales,
                        recv_topk_weights.value(),
                        recv_topk_idx.value(),
                        num_recv_tokens_per_expert,
                        recv_token_num_ptr);
        }
        else {
            // intranode dispatch
            Tensor                x = input.x;
            std::optional<Tensor> x_scales;
            if (input.use_fp8) {
                x        = {};
                x_scales = Tensor{};
                QuantizeSymm(x, x_scales.value(), input.x, core::Context::stream().handle());
                x_scales = x_scales->transpose(0, 1);
            }
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
                  send_head] = buffer_->intranode_dispatch(x,
                                                           x_scales,
                                                           input.topk_idx,
                                                           input.topk_weights,
                                                           num_tokens_per_rank,
                                                           is_token_in_rank,
                                                           num_tokens_per_expert,
                                                           0,
                                                           std::nullopt,
                                                           std::nullopt,
                                                           1,
                                                           input.num_worst_tokens,
                                                           config);
            sync_check_cuda_error();

            // Generate output
            output.handle = {rank_prefix_matrix,
                             channel_prefix_matrix,
                             recv_channel_prefix_matrix,
                             recv_src_idx,
                             is_token_in_rank,
                             send_head};

            const int  nranks             = h_comm_->n_ranks();
            const int  rank               = h_comm_->rank();
            const int* recv_token_num_ptr = rank_prefix_matrix.data<int>() + (nranks - 1) * nranks + rank;
            Postprocess(recv_x,  //
                        recv_x_scales,
                        recv_topk_weights.value(),
                        recv_topk_idx.value(),
                        num_recv_tokens_per_expert,
                        recv_token_num_ptr);
        }
    }
}

void NcclCommImpl::Combine(const EpCombineInput& input, EpCombineOutput& output, int group)
{
    TM_CHECK_EQ(group, 0);
    TM_CHECK(input.mode != EpMode::kNull);

    if (input.mode == EpMode::kLowLatency) {
        const auto& offsets                  = input.handle[2];
        auto&       packed_recv_src_info     = input.handle[0];
        auto&       packed_recv_layout_range = input.handle[1];

        auto [combined_x] = buffer_->low_latency_combine(input.x,
                                                         offsets,
                                                         input.topk_idx.value(),
                                                         input.topk_weights.value(),
                                                         packed_recv_src_info,
                                                         packed_recv_layout_range,
                                                         std::nullopt,
                                                         ep_config_.ll_max_tokens_per_rank,
                                                         ep_config_.num_experts,
                                                         false,
                                                         input.zero_copy,
                                                         std::nullopt);
        sync_check_cuda_error();

        // Generate output
        output.out_x = combined_x;
    }
    else {
        if (buffer_->get_num_rdma_ranks() > 1) {
            // internode combine
            auto config = buffer_->get_combine_config();

            auto src_meta                   = input.handle[7];
            auto is_combined_token_in_rank  = input.handle[0];
            auto rdma_channel_prefix_matrix = input.handle[3];
            auto rdma_rank_prefix_sum       = input.handle[4];
            auto gbl_channel_prefix_matrix  = input.handle[5];
            auto combined_rdma_head         = input.handle[8];
            auto combined_nvl_head          = input.handle[9];

            auto [combined_x, combined_topk_weights] = buffer_->internode_combine(input.x,
                                                                                  std::nullopt,
                                                                                  std::nullopt,
                                                                                  std::nullopt,
                                                                                  src_meta,
                                                                                  is_combined_token_in_rank,
                                                                                  rdma_channel_prefix_matrix,
                                                                                  rdma_rank_prefix_sum,
                                                                                  gbl_channel_prefix_matrix,
                                                                                  combined_rdma_head,
                                                                                  combined_nvl_head,
                                                                                  config);
            sync_check_cuda_error();
            output.out_x = combined_x;
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
                                                                          std::nullopt,
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

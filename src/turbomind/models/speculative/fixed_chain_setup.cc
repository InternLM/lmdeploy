// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/fixed_chain_setup.h"

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/engine/request.h"

#include <algorithm>
#include <numeric>

namespace turbomind {

FixedChainSetup::FixedChainSetup(const Communicators& comm, int phases, int max_batch_size):
    comm_{comm}, limit_to_accept_len_host_{max_batch_size, kCPUpinned}, data_(phases)
{
    for (FixedChainPhaseData& data : data_) {
        data.draft_extension_q_offsets_host = {max_batch_size + 1, kCPUpinned};
        data.draft_extension_q_offsets      = {max_batch_size + 1, kDEVICE};
        data.draft_extension_k_offsets      = {max_batch_size + 1, kDEVICE};
        data.draft_extension_local_token_nums.assign(comm_.h_dp_group->n_ranks(), 0);
        data.limit_to_accept_len = {max_batch_size, kDEVICE};
    }
}

void FixedChainSetup::Setup(int phase, const Buffer_<Sequence*>& requests, core::BatchCopy& copy)
{
    FixedChainPhaseData& data = data_.at(phase);
    const int batch_size      = requests.size();

    Buffer_<int>& extension = data.draft_extension_q_offsets_host;
    extension[0]            = 0;

    data.refresh_decode  = {};
    data.refresh_prefill = {};

    const int decode_request_count =
        std::find_if(requests.begin(), requests.end(), [](const Sequence* request) {
            return request->submitted->input_len > 1;
        })
        - requests.begin();

    for (int b = 0; b < batch_size; ++b) {
        const SubmittedRow& row = *requests[b]->submitted;

        extension[b + 1] = extension[b] + static_cast<int>(row.is_extension_candidate());
        limit_to_accept_len_host_[b] = row.autoregres;

        auto& partition = b < decode_request_count ? data.refresh_decode : data.refresh_prefill;
        partition.request_count += 1;
        partition.query_count += row.input_len;
        partition.max_query_length = std::max(partition.max_query_length, row.input_len);
        partition.key_capacity_sum += row.key_capacity_end;
        partition.max_key_capacity = std::max(partition.max_key_capacity, row.key_capacity_end);
    }

    data.draft_extension_query_count = extension[batch_size];

    const int attn_dp_rank = comm_.h_dp_group->rank();
    std::fill(data.draft_extension_local_token_nums.begin(), data.draft_extension_local_token_nums.end(), 0);
    data.draft_extension_local_token_nums[attn_dp_rank] = data.draft_extension_query_count;

    if (comm_.h_dp_group->n_ranks() > 1) {
        comm::AllGather(comm_.h_dp_group, data.draft_extension_local_token_nums.data(), 1);
    }

    data.draft_extension_global_query_count =
        std::accumulate(data.draft_extension_local_token_nums.begin(),
                        data.draft_extension_local_token_nums.end(),
                        0);

    data.extension_decode                  = {};
    data.extension_decode.request_count    = batch_size;
    data.extension_decode.query_count      = data.draft_extension_query_count;
    data.extension_decode.max_query_length = data.draft_extension_query_count ? 1 : 0;

    for (int b = 0; b < batch_size; ++b) {
        if (extension[b + 1] == extension[b]) {
            continue;
        }
        const int capacity = requests[b]->submitted->cache_write_end;
        data.extension_decode.key_capacity_sum += capacity;
        data.extension_decode.max_key_capacity = std::max(data.extension_decode.max_key_capacity, capacity);
    }

    copy(extension, batch_size + 1, data.draft_extension_q_offsets);
    copy(limit_to_accept_len_host_, batch_size, data.limit_to_accept_len);
}

}  // namespace turbomind

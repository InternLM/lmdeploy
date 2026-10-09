// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/buffer.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

#include <vector>

namespace turbomind {

class Sequence;

namespace core {
class BatchCopy;
}

struct FixedChainPhaseData {
    Buffer_<int> draft_extension_q_offsets_host;
    Buffer_<int> draft_extension_q_offsets;
    Buffer_<int> draft_extension_k_offsets;
    int          draft_extension_query_count{};

    std::vector<int> draft_extension_local_token_nums;
    int              draft_extension_global_query_count{};

    AttentionForwardMetadata::Partition refresh_decode;
    AttentionForwardMetadata::Partition refresh_prefill;
    AttentionForwardMetadata::Partition extension_decode;

    Buffer_<bool> limit_to_accept_len;
};

class FixedChainSetup {
public:
    FixedChainSetup(const Communicators& comm, int phases, int max_batch_size);

    void Setup(int phase, const Buffer_<Sequence*>& requests, core::BatchCopy& copy);

    FixedChainPhaseData& phase_data(int phase)
    {
        return data_.at(phase);
    }

private:
    const Communicators& comm_;
    Buffer_<bool>        limit_to_accept_len_host_;
    std::vector<FixedChainPhaseData> data_;
};

}  // namespace turbomind

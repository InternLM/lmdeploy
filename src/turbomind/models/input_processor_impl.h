// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/models/input_processor.h"
#include "src/turbomind/models/vision_model.h"

namespace turbomind {

struct InputProcessor::Impl {

    Impl(const EngineParam& engine,
         int                hidden_units,
         DataType           data_type,
         int                phases,
         bool               speculative,
         int                max_verification_positions,
         bool               successor_embeddings);

    int  Add(Sequence& c);
    void Add(int phase, TensorMap& env);

    // Shared lifecycle entry: dispatches to the mode implementation.
    void Setup(int phase, TensorMap& env);
    void Prepare(int phase, TensorMap& env);

    // Ordinary mode (input_processor.cc): generating rows trim to one token,
    // autoregressive ids patch the packed row, and selected positions are
    // row-major.
    void SetupOrdinary(int phase, TensorMap& env);
    void PrepareOrdinary(int phase, TensorMap& env);

    // Composed mode (input_processor_speculative.cc): all query rows pack
    // verbatim, selected positions carry the verification geometry, and
    // target ids are gathered from the request's token row.
    void SetupSpeculative(int phase, TensorMap& env);
    void PrepareSpeculative(int phase, TensorMap& env);
    void BuildTargetInputs(int phase, TensorMap& env);

    // Stages input-embedding patches for the submitted rows; successor
    // staging is requested only by the composed mode.
    void StageEmbeddingPatches(int phase, const Buffer_<Sequence*>& rc, bool stage_successor);

    static void ApplyPatches(const Tensor&                  source,
                             const std::vector<EmbeddingPatch>& patches,
                             Tensor&                        destination,
                             BatchCopy&                     copy);

    void PatchEmbedding(int phase, Tensor& embeds, BatchCopy& copy, TensorMap& env);
    void PatchSuccessorEmbedding(int phase, Tensor& embeds, BatchCopy& copy, TensorMap& env);

    struct Data {
        Buffer_<int> input_ids;
        Buffer_<int> input_ids_offsets;
        int          input_token_num;

        Buffer_<int>  selected_token_pos;
        int           selected_token_count;
        Buffer_<bool> target_ids_from_row;

        Buffer_<int> target_key_lengths;

        Buffer_<int> autoreg_ids_pos;

        Tensor                          input_embeds_buf;
        std::vector<EmbeddingPatch> target_input_patches;
        std::vector<EmbeddingPatch> successor_input_patches;
    };

    const int  max_batch_size_;
    const int  max_forward_token_num_;
    const bool speculative_engine_;
    const bool successor_embeddings_;

    std::vector<Data> data_;

    Buffer_<int> input_ids_buf_;
    Buffer_<int> input_ids_offsets_buf_;

    Buffer_<int>  decode_token_pos_buf_;
    Buffer_<bool> target_ids_from_row_buf_;
};

}  // namespace turbomind

#pragma once

#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/nccl_utils.h"

namespace turbomind {

template<typename T>
class UnifiedDecoder {
private:
    void freeBuffer();

    const size_t       layer_num_;
    const size_t       hidden_units_;
    const float        rmsnorm_eps_;
    cudaStream_t const stream_;
    IAllocator* const  allocator_;
    const DataType     dtype_;
    bool               is_free_buffer_after_forward_{};

    int* cu_q_len_{};
    int* cu_k_len_{};

    int* h_cu_q_len_{};
    int* h_cu_k_len_{};

    std::unique_ptr<UnifiedAttentionLayer<T>> attn_layer_;
    std::unique_ptr<LlamaFfnLayer<T>>         ffn_layer_;

    cudaEvent_t ev_h_cu_x_{};

    using WeightType = LlamaDecoderLayerWeight<T>;

    void forwardSelfAttn(T*                             attn_io,
                         TensorMap*                     _outputs,
                         const TensorMap*               _inputs,
                         size_t                         token_num,
                         size_t                         batch_size,
                         int                            layer_id,
                         const LlamaAttentionWeight<T>* weight);

public:
    UnifiedDecoder(const ModelParam&     model,
                   const AttentionParam& attn,
                   const LoraParam&      lora,
                   const NcclParam&      tp,
                   const Context<T>&     ctx);

    void allocateBuffer(size_t max_batch_size);

    ~UnifiedDecoder();

    void forward(TensorMap* outputs, const TensorMap* inputs, const std::vector<WeightType*>* weights);
};

}  // namespace turbomind

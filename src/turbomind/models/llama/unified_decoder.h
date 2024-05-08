#pragma once

#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/nccl_utils.h"

namespace turbomind {

template<typename T>
class UnifiedDecoder {
protected:
    void freeBuffer();

    void
    initialize(const LlamaAttentionParams& attn_params, size_t kv_head_num, int cache_block_seq_len, int quant_policy);

    cudaStream_t     stream_;
    cublasMMWrapper* cublas_wrapper_;
    IAllocator*      allocator_;
    bool             is_free_buffer_after_forward_{};

    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t hidden_units_;
    float  rmsnorm_eps_;

    NcclParam tensor_para_;

    LoraParams lora_params_;

    int* cu_q_len_{};
    int* cu_k_len_{};

    int* h_cu_q_len_{};
    int* h_cu_k_len_{};

    UnifiedAttentionLayer<T>* attn_layer_{};
    LlamaFfnLayer<T>*         ffn_layer_{};

    const DataType dtype_;

    // bool need_causal_mask_{false};

    using WeightType = LlamaDecoderLayerWeight<T>;

    void forwardSelfAttn(T*                             attn_io,
                         TensorMap*                     _outputs,
                         const TensorMap*               _inputs,
                         size_t                         token_num,
                         size_t                         batch_size,
                         int                            layer_id,
                         const LlamaAttentionWeight<T>* weight);

public:
    UnifiedDecoder(size_t                      head_num,
                   size_t                      kv_head_num,
                   size_t                      size_per_head,
                   size_t                      inter_size,
                   size_t                      num_layer,
                   const LlamaAttentionParams& attn_params,
                   float                       rmsnorm_eps,
                   NcclParam                   tensor_para,
                   cudaStream_t                stream,
                   cublasMMWrapper*            cublas_wrapper,
                   IAllocator*                 allocator,
                   const LoraParams&           lora_params,
                   bool                        is_free_buffer_after_forward,
                   bool                        use_fmha,
                   int                         cache_block_seq_len,
                   int                         quant_policy):
        stream_(stream),
        cublas_wrapper_(cublas_wrapper),
        allocator_(allocator),
        lora_params_(lora_params),
        is_free_buffer_after_forward_(is_free_buffer_after_forward),
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        hidden_units_(head_num * size_per_head),
        num_layer_(num_layer),
        rmsnorm_eps_(rmsnorm_eps),
        tensor_para_(tensor_para),
        dtype_(getTensorType<T>())
    {
        initialize(attn_params, kv_head_num, cache_block_seq_len, quant_policy);
    }

    void allocateBuffer(size_t max_batch_size);

    ~UnifiedDecoder();

    void forward(TensorMap* outputs, const TensorMap* inputs, const std::vector<WeightType*>* weights);
};

}  // namespace turbomind

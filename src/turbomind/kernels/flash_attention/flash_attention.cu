#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "flash_attention.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

#define VERSION_SWITCH(VERSION, CONST_NAME, ...)                                                                       \
    [&] {                                                                                                              \
        if (VERSION == 2) {                                                                                            \
            constexpr static int CONST_NAME = 2;                                                                       \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else {                                                                                                         \
            constexpr static int CONST_NAME = 1;                                                                       \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

template<typename T>
FlashAttentionOp<T>::FlashAttentionOp(int batch_size, int head_num, int key_len, int seq_len, int size_per_head):
    batch_size_(batch_size), head_num_(head_num), key_len_(key_len), seq_len_(seq_len), size_per_head_(size_per_head)
{
#ifdef _MSC_VER
    op_version_ = 1;
#else
    op_version_ = std::is_same<float, typename std::decay<T>::type>::value ? 1 : 2;
    if (op_version_ == 2 && getSMVersion() < 80) {
        op_version_ = 1;
    }
#endif
}

template<typename T>
int FlashAttentionOp<T>::get_workspace_size() const
{
#ifdef _MSC_VER
    FlashAttentionOpImpl<T, 1> attention_op(batch_size_, head_num_, key_len_, seq_len_, size_per_head_);
    return attention_op.get_workspace_size();
#else
    return VERSION_SWITCH(op_version_, OP_VERSION, [&]() {
        FlashAttentionOpImpl<T, OP_VERSION> attention_op(batch_size_, head_num_, key_len_, seq_len_, size_per_head_);
        return attention_op.get_workspace_size();
    });
#endif
}

template<typename T>
void FlashAttentionOp<T>::operator()(Params& params, cudaStream_t st) const
{
#ifdef _MSC_VER
    FlashAttentionOpImpl<T, 1> attention_op(batch_size_, head_num_, key_len_, seq_len_, size_per_head_);
    return attention_op(params, st);
#else
    return VERSION_SWITCH(op_version_, OP_VERSION, [&]() {
        FlashAttentionOpImpl<T, OP_VERSION> attention_op(batch_size_, head_num_, key_len_, seq_len_, size_per_head_);
        return attention_op(params, st);
    });
#endif
}

template class FlashAttentionOp<float>;
template class FlashAttentionOp<half>;
#ifdef ENABLE_BF16
template class FlashAttentionOp<__nv_bfloat16>;
#endif

}  // namespace turbomind
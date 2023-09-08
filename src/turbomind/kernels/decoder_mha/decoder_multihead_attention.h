
#include "decoder_multihead_attention_params.h"

namespace turbomind {

template<typename T, int HeadDim>
void LaunchDecoderMultiheadAttention(const DecoderMultiHeadAttentionParams<T>& params);

}
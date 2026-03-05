// Copyright (c) OpenMMLab. All rights reserved.

#include "decoding.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/attention/registry.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class T>
void dispatchDecoding(const AttentionParams<T>& params)
{
    using namespace attention;

    const bool is_kv_int8     = params.quant_policy & QuantPolicy::kCacheKVInt8;
    const bool is_kv_int4     = params.quant_policy & QuantPolicy::kCacheKVInt4;
    const int  query_group_sz = params.num_heads / params.num_kv_heads;

    FT_CHECK(!(is_kv_int4 && is_kv_int8));

    int kv_quant = is_kv_int4 ? 4 : (is_kv_int8 ? 8 : 0);

    AttnDesc desc{};
    desc.mode           = AttnDesc::kDecoding;
    desc.head_dim       = params.size_per_head;
    desc.data_type      = data_type_v<T>;
    desc.kv_quant       = kv_quant;
    desc.query_group_sz = query_group_sz;

    auto& reg    = Registry::instance();
    auto* kernel = reg.Find(desc);

    TM_CHECK(kernel) << "No decoding kernel found: " + to_string(desc);

    kernel->Launch(&params, reg.sm_count());
}

template void dispatchDecoding(const AttentionParams<half>& params);
#if ENABLE_BF16
template void dispatchDecoding(const AttentionParams<nv_bfloat16>& params);
#endif

}  // namespace turbomind

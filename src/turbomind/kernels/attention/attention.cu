// Copyright (c) OpenMMLab. All rights reserved.

#include "attention.h"
#include "src/turbomind/kernels/attention/registry.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class T>
void dispatchAttention(const AttentionParams<T>& params)
{
    using namespace attention;

    auto& reg = Registry::instance();
    AttnDesc desc{};
    desc.mode     = AttnDesc::kPrefill;
    desc.head_dim = params.size_per_head;
    desc.is_bf16  = std::is_same_v<T, nv_bfloat16>;

    auto* kernel = reg.Find(desc);

    TM_CHECK(kernel) << "No attention kernel found: " + to_string(desc);

    kernel->Launch(&params);
}

template void dispatchAttention(const AttentionParams<half>& params);
#if ENABLE_BF16
template void dispatchAttention(const AttentionParams<nv_bfloat16>& params);
#endif

}  // namespace turbomind

// Copyright (c) OpenMMLab. All rights reserved.

#include "attention.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/attention/registry.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class T>
void dispatchAttention(const AttentionParams<T>& params)
{
    using namespace attention;

    auto&    reg = Registry::instance();
    AttnDesc desc{};
    desc.mode      = AttnDesc::kPrefill;
    desc.head_dim  = params.size_per_head;
    desc.data_type = data_type_v<T>;

    auto* kernel = reg.Find(desc);

    TM_CHECK(kernel) << "No attention kernel found: " + to_string(desc);

    kernel->Launch(&params, reg.sm_count());
}

template void dispatchAttention(const AttentionParams<half>& params);
#if ENABLE_BF16
template void dispatchAttention(const AttentionParams<nv_bfloat16>& params);
#endif

}  // namespace turbomind

// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <type_traits>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/attention/attention_template.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/decoding_template.h"
#include "src/turbomind/kernels/attention/kernel.h"
#include "src/turbomind/kernels/core/common.h"

namespace turbomind::attention {

template<class Tkv>
constexpr int kv_quant_from_type()
{
    if constexpr (std::is_same_v<Tkv, uint8_t>) {
        return 8;
    }
    else if constexpr (std::is_same_v<Tkv, uint4_t>) {
        return 4;
    }
    else {
        return 0;
    }
}

template<class K>
class KernelImpl: public Kernel {
    static constexpr bool kIsDecoding = std::is_same_v<typename K::CtaMap, DecodingCtaMap>;

public:
    KernelImpl()
    {
        desc_.mode     = kIsDecoding ? AttnDesc::kDecoding : AttnDesc::kPrefill;
        desc_.arch     = K::Arch::value;
        desc_.head_dim = K::kHeadDim;
        desc_.is_bf16  = std::is_same_v<typename K::T, nv_bfloat16>;

        if constexpr (kIsDecoding) {
            desc_.kv_quant = kv_quant_from_type<typename K::Tkv>();
            desc_.qh       = K::CTA_H;
        }
        else {
            desc_.kv_quant = 0;
            desc_.qh       = 1;
        }

        auto func               = &attention_kernel<K>;
        info_.dynamic_smem_size = sizeof(typename K::SharedStorage);

        cudaFuncGetAttributes(&info_.attr, func);

        if (info_.dynamic_smem_size > (48 << 10)) {
            cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, info_.dynamic_smem_size);
        }

        info_.num_warps = K::kWarpCount;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info_.max_active_ctas, func, info_.num_warps * WARP_SIZE, info_.dynamic_smem_size);

        info_.name = to_string(desc_);
    }

    bool Launch(const void* params) const override
    {
        const auto& p = *static_cast<const typename K::ParamType*>(params);
        if constexpr (kIsDecoding) {
            return invokeDecoding<K>(p);
        }
        else {
            invokeAttention<K>(p);
            return true;
        }
    }
};

}  // namespace turbomind::attention

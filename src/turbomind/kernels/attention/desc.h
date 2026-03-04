// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>
#include <sstream>
#include <string>

namespace turbomind::attention {

struct AttnDesc {
    enum Mode
    {
        kPrefill,
        kDecoding
    };
    Mode mode;
    int  head_dim;
    bool is_bf16;
    int  kv_quant;  // 0=none, 8=int8, 4=int4
    int  qh;        // query heads per CTA (1 for prefill)
};

inline std::string to_string(const AttnDesc& d)
{
    std::ostringstream ss;
    ss << (d.mode == AttnDesc::kPrefill ? "prefill" : "decode");
    ss << "_d" << d.head_dim;
    ss << "_" << (d.is_bf16 ? "bf16" : "f16");
    if (d.mode == AttnDesc::kDecoding) {
        if (d.kv_quant == 8)
            ss << "_kvint8";
        else if (d.kv_quant == 4)
            ss << "_kvint4";
        ss << "_qh" << d.qh;
    }
    return ss.str();
}

struct KernelDesc {
    AttnDesc::Mode mode;
    int            arch;      // 700, 750, 800
    int            head_dim;
    bool           is_bf16;
    int            kv_quant;  // 0=none, 8=int8, 4=int4
    int            qh;        // query heads per CTA (1 for prefill)
};

struct KernelInfo {
    int                dynamic_smem_size;
    int                max_active_ctas;
    int                num_warps;
    std::string        name;
    cudaFuncAttributes attr;
};

inline std::string to_string(const KernelDesc& d)
{
    std::ostringstream ss;
    ss << (d.mode == AttnDesc::kPrefill ? "prefill" : "decode");
    ss << "_sm" << d.arch / 10;
    ss << "_d" << d.head_dim;
    ss << "_" << (d.is_bf16 ? "bf16" : "f16");
    if (d.mode == AttnDesc::kDecoding) {
        if (d.kv_quant == 8)
            ss << "_kvint8";
        else if (d.kv_quant == 4)
            ss << "_kvint4";
        ss << "_qh" << d.qh;
    }
    return ss.str();
}

}  // namespace turbomind::attention

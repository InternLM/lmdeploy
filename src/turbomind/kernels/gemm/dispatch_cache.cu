// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <iostream>

static inline bool operator==(const int3& a, const int3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

static inline bool operator==(const int2& a, const int2& b)
{
    return a.x == b.x && a.y == b.y;
}

namespace turbomind::gemm {

static inline decltype(auto) as_tuple(const KernelDesc& d)
{
    return std::tie(d.arch,
                    d.op_class,
                    d.type_a,
                    d.type_b,
                    d.type_c,
                    d.order_a,
                    d.order_b,
                    d.order_c,
                    d.pack_a,
                    d.pack_b,
                    d.pack_u,
                    d.pack_v,
                    d.quant_a,
                    d.quant_b,
                    d.policy_a,
                    d.policy_b,
                    d.cta_tile,
                    d.mma_tile,
                    d.align,
                    d.c_tile,
                    d.stages,
                    d.split_k);
}

static inline bool operator==(const QuantDesc& a, const QuantDesc& b)
{
    return a.type == b.type && a.group_size == b.group_size;
}

static inline bool operator==(const KernelDesc& a, const KernelDesc& b)
{
    return as_tuple(a) == as_tuple(b);
}

template<class... Ts>
static inline void export_impl(std::ostream& os, const Ts&... ts)
{
    ((os << static_cast<int>(ts) << " "), ...);
}

template<class T>
static inline void import_value(std::istream& is, T& value)
{
    int token{};
    is >> token;
    value = static_cast<T>(token);
}

template<class... Ts>
static inline void import_impl(std::istream& is, Ts&... ts)
{
    (import_value(is, ts), ...);
}

void ExportDispatchCache(std::ostream& os, const std::vector<std::pair<GemmDesc, LaunchSpec>>& entries)
{
    for (const auto& [g, spec] : entries) {
        // GEMM desc
        export_impl(os,
                    g.type_a,
                    g.type_b,
                    g.type_c,
                    g.order_a,
                    g.order_b,
                    g.order_c,
                    g.pack_a,
                    g.pack_b,
                    g.pack_u,
                    g.pack_v,
                    g.quant_a.type,
                    g.quant_a.group_size,
                    g.quant_b.type,
                    g.quant_b.group_size,
                    g.epilogue,
                    g.m,
                    g.n,
                    g.k);
        // Kernel desc
        auto& k = spec.kernel->desc();
        export_impl(os,
                    k.arch,
                    k.op_class,
                    k.cta_tile.x,
                    k.cta_tile.y,
                    k.cta_tile.z,
                    k.mma_tile.x,
                    k.mma_tile.y,
                    k.mma_tile.z,
                    k.stages,
                    k.align.x,
                    k.align.y,
                    k.align.z,
                    k.policy_a,
                    k.policy_b,
                    k.c_tile.x,
                    k.c_tile.y,
                    k.split_k);
        // Runtime params
        export_impl(os, spec.splits, spec.swizzle);
        os << std::endl;
    }
}

void ImportDispatchCache(std::istream&                                 is,
                         std::vector<std::pair<GemmDesc, LaunchSpec>>& entries,
                         const std::vector<std::unique_ptr<Kernel>>&   kernels)
{
    std::string line;
    while (std::getline(is, line)) {
        std::cout << line << std::endl;
        std::stringstream ss(line);
        GemmDesc          g{};
        import_impl(ss,
                    g.type_a,
                    g.type_b,
                    g.type_c,
                    g.order_a,
                    g.order_b,
                    g.order_c,
                    g.pack_a,
                    g.pack_b,
                    g.pack_u,
                    g.pack_v,
                    g.quant_a.type,
                    g.quant_a.group_size,
                    g.quant_b.type,
                    g.quant_b.group_size,
                    g.epilogue,
                    g.m,
                    g.n,
                    g.k);
        KernelDesc k{};
        k.type_a  = g.type_a;
        k.type_b  = g.type_b;
        k.type_c  = g.type_c;
        k.pack_a  = g.pack_a;
        k.pack_b  = g.pack_b;
        k.pack_u  = g.pack_u;
        k.pack_v  = g.pack_v;
        k.order_a = g.order_a;
        k.order_b = g.order_b;
        k.order_c = g.order_c;
        k.quant_a = g.quant_a;
        k.quant_b = g.quant_b;
        import_impl(ss,
                    k.arch,
                    k.op_class,
                    k.cta_tile.x,
                    k.cta_tile.y,
                    k.cta_tile.z,
                    k.mma_tile.x,
                    k.mma_tile.y,
                    k.mma_tile.z,
                    k.stages,
                    k.align.x,
                    k.align.y,
                    k.align.z,
                    k.policy_a,
                    k.policy_b,
                    k.c_tile.x,
                    k.c_tile.y,
                    k.split_k);
        LaunchSpec spec{};
        import_impl(ss, spec.splits, spec.swizzle);
        for (const auto& p : kernels) {
            if (p->desc() == k) {
                spec.kernel = p.get();
                break;
            }
        }
        if (spec.kernel) {
            entries.emplace_back(g, spec);
        }
        else {
            std::cerr << "No kernel found for entry: " << line << "\n";
        }
    }
}

}  // namespace turbomind::gemm

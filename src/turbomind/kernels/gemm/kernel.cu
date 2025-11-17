// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>

#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

bool accept(Striding a, Striding b)
{
    if (a == Striding::kBlocked) {
        switch (b) {
            case Striding::kBlocked:
            case Striding::kFlat:
                return true;
            default:
                return false;
        }
    }
    else if (a == Striding::kIndexed) {
        switch (b) {
            case Striding::kFlat:
            case Striding::kBlocked:
            case Striding::kIndexed:
                return true;
            default:
                return false;
        }
    }
    else {
        return a == b;
    }
}

bool Kernel::is_feasible(const GemmDesc& desc) const noexcept
{
    constexpr bool debug = 0;

    if constexpr (debug)
        printf("S\n");

    // printf("%d %d\n", desc.arch, desc_.arch);

    if (!is_arch_compatible(desc_.arch, desc.arch)) {
        return false;
    }

    if constexpr (debug)
        printf("S0\n");

    if (std::tie(desc.order_a, desc.order_b, desc.order_c) != std::tie(desc_.order_a, desc_.order_b, desc_.order_c)) {
        return false;
    }

    if (desc.group_axis >= 0 && desc.group_axis != desc_.group_axis) {
        return false;
    }

    if (!(accept(desc_.striding_a, desc.striding_a)     //
          && accept(desc_.striding_b, desc.striding_b)  //
          && accept(desc_.striding_c, desc.striding_c))) {
        return false;
    }

    if constexpr (debug)
        printf("A\n");

    if (std::tie(desc.type_a, desc.type_b, desc.type_c) != std::tie(desc_.type_a, desc_.type_b, desc_.type_c)) {
        return false;
    }

    if constexpr (debug) {
        printf("B\n");
        printf("%X %X %X %X\n", desc.pack_a, desc_.pack_a, desc.pack_u, desc_.pack_u);
    }

    if (std::tie(desc.pack_a, desc.pack_u) != std::tie(desc_.pack_a, desc_.pack_u)) {
        return false;
    }

    if constexpr (debug) {
        printf("C\n");
        printf("%X %X %X %X\n", desc.pack_b, desc_.pack_b, desc.pack_v, desc_.pack_v);
    }

    if (std::tie(desc.pack_b, desc.pack_v) != std::tie(desc_.pack_b, desc_.pack_v)) {
        return false;
    }

    if constexpr (debug)
        printf("D\n");

    if (desc.quant_a.type != desc_.quant_a.type || desc.quant_a.group_size != desc_.quant_a.group_size) {
        return false;
    }

    if constexpr (debug)
        printf("E\n");

    if (desc.quant_b.type != desc_.quant_b.type || desc.quant_b.group_size != desc_.quant_b.group_size) {
        return false;
    }

    if constexpr (debug)
        printf("F\n");

    if (desc.m % desc_.align.x || desc.n % desc_.align.y || desc.k % desc_.align.z) {
        return false;
    }

    if constexpr (debug)
        printf("success\n");

    return true;
}

//  mm:     m * n * k,     m * k,     n * k,     m * n
// Bmm: b * m * n * k, b * m * k, b * n * k, b * m * n
// Gmm: S $ M * n * k, S $ M * k, S $ n * k, S $ M * n

std::string Kernel::GetName() const
{
    std::stringstream ss;

    ss << "sm" << desc_.arch / 10;
    ss << "_" << to_string(desc_.type_a);  //
    if (desc_.quant_a) {
        ss << to_string(desc_.quant_a);
    }
    ss << "_" << to_string(desc_.type_b);  //
    if (desc_.quant_b) {
        ss << to_string(desc_.quant_b);
    }
    ss << "_" << to_string(desc_.type_c);
    ss << "_"                                        //
       << (desc_.order_a == kColMajor ? 'n' : 't')   //
       << (desc_.order_b == kColMajor ? 'n' : 't')   //
       << (desc_.order_c == kColMajor ? 'n' : 't');  //
    ss << "_"                                        //
       << to_string(desc_.striding_a)                //
       << to_string(desc_.striding_b)                //
       << to_string(desc_.striding_c);
    ss << "_" << desc_.cta_tile.x << "x" << desc_.cta_tile.y << "x" << desc_.cta_tile.z  //
       << "_" << desc_.stages                                                            //
       << "_" << desc_.cluster_shape.x << "x" << desc_.cluster_shape.y                   //
       << "_" << to_string(desc_.op_class)                                               //
       << "_" << desc_.mma_tile.x << "x" << desc_.mma_tile.y << "x" << desc_.mma_tile.z;
    if (desc_.group_axis >= 0) {
        ss << "_"
           << "mn"[desc_.group_axis] << "group";
    }
    ss << "_c" << desc_.c_tile.x << "x" << desc_.c_tile.y                        //
       << "_a" << desc_.align.x << "x" << desc_.align.y << "x" << desc_.align.z  //
       << "_" << desc_.policy_a << desc_.policy_b;

    return ss.str();
}

class TransposedKernel: public Kernel {
public:
    explicit TransposedKernel(Kernel& kernel): kernel_(&kernel)
    {
        desc_ = kernel.desc();
        info_ = kernel.info();

        desc_.transpose = !desc_.transpose;
    }

    int Launch(const Operation&    operation,
               float               alpha,
               const void*         A,
               const MatrixLayout& Adesc,
               const void*         U,
               const MatrixLayout& Udesc,
               const void*         B,
               const MatrixLayout& Bdesc,
               const void*         V,
               const MatrixLayout& Vdesc,
               float               beta,
               const void*         C,
               const MatrixLayout& Cdesc,
               void*               D,
               const MatrixLayout& Ddesc,
               int                 swizzle,
               int                 splits,
               Workspace&          workspace,
               cudaStream_t        stream) override
    {
        return kernel_->Launch(transpose(operation),
                               alpha,
                               B,
                               transpose(Bdesc),
                               V,
                               transpose(Vdesc),
                               A,
                               transpose(Adesc),
                               U,
                               transpose(Udesc),
                               beta,
                               C,
                               transpose(Cdesc),
                               D,
                               transpose(Ddesc),
                               swizzle,
                               splits,
                               workspace,
                               stream);
    }

    bool is_feasible(const GemmDesc& desc) const noexcept override
    {
        return kernel_->is_feasible(desc);
    }

    int GetMaxSwizzle(const int4& shape) const override
    {
        return kernel_->GetMaxSwizzle(shape);
    }

    int GetMaxSplits(const int4& shape, int swizzle, size_t bsize, size_t psize) const override
    {
        return kernel_->GetMaxSplits(shape, swizzle, bsize, psize);
    }

private:
    Kernel* kernel_;
};

std::unique_ptr<Kernel> transpose(Kernel& kernel)
{
    return std::make_unique<TransposedKernel>(kernel);
}

template<class Op>
inline static bool cmp(const int3& a, const int3& b, Op op)
{
    return op(std::tie(a.x, a.y, a.z), std::tie(b.x, b.y, b.z));
}

std::vector<std::vector<LaunchSpec>> Cluster(const std::vector<LaunchSpec>& specs, const ClusteringParam& param)
{
    std::vector<const LaunchSpec*> ptrs;  // pointer into `specs`
    for (auto& s : specs) {
        ptrs.push_back(&s);
    }

    auto less = [&](const LaunchSpec* u, const LaunchSpec* v) {
        const auto& a = u->kernel->desc();
        const auto& b = v->kernel->desc();
        if (!cmp(a.cta_tile, b.cta_tile, std::equal_to<>{})) {
            return cmp(a.cta_tile, b.cta_tile, std::less<>{});
        }
        if (!cmp(a.mma_tile, b.mma_tile, std::equal_to<>{})) {
            return cmp(a.mma_tile, b.mma_tile, std::less<>{});
        }
        if (param.cache_policy) {
            const auto pa = std::tie(a.policy_a, a.policy_b);
            const auto pb = std::tie(b.policy_a, b.policy_b);
            if (pa != pb) {
                return pa < pb;
            }
        }
        if (param.max_active_ctas) {
            const auto& a = u->kernel->info();
            const auto& b = v->kernel->info();
            if (a.max_active_ctas != b.max_active_ctas) {
                return a.max_active_ctas < b.max_active_ctas;
            }
        }
        return u->splits < v->splits;
    };

    std::stable_sort(ptrs.begin(), ptrs.end(), less);

    if (ptrs.empty()) {
        return {};
    }
    std::vector<std::vector<LaunchSpec>> clusters{{*ptrs[0]}};

    auto equal = [&](const LaunchSpec* u, const LaunchSpec* v) {  //
        return !less(u, v) && !less(v, u);
    };
    int p = 0;
    for (size_t i = 1; i < ptrs.size(); ++i) {
        if (equal(ptrs[p], ptrs[i])) {
            clusters.back().push_back(*ptrs[i]);
        }
        else {
            clusters.push_back({*ptrs[i]});
            p = i;
        }
    }

    return clusters;
}

}  // namespace turbomind::gemm

// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/desc.h"

#include <memory>
#include <optional>
#include <vector>

namespace turbomind::gemm {

class DispatchCache {
public:
    DispatchCache(std::vector<Kernel*> kernels);

    ~DispatchCache();

    std::optional<LaunchSpec> LowerBound(const GemmDesc& desc) const;

    std::optional<LaunchSpec> Find(const GemmDesc& desc) const;

    bool Insert(const GemmDesc& desc, const LaunchSpec& spec);

    int Export(std::ostream& os) const;

    int Import(std::istream& is);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind::gemm

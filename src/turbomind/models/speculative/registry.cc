// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/registry.h"

#include "src/turbomind/core/check.h"

namespace turbomind {

SpeculativeModelRegistry& SpeculativeModelRegistry::Instance()
{
    static SpeculativeModelRegistry registry;
    return registry;
}

void SpeculativeModelRegistry::Register(std::string name, Factory factory)
{
    factories_.emplace(std::move(name), std::move(factory));
}

bool SpeculativeModelRegistry::Contains(std::string_view name) const
{
    return factories_.find(name) != factories_.end();
}

std::unique_ptr<SpeculativeModel>
SpeculativeModelRegistry::Create(std::string_view name, const SpeculativeModelArgs& args) const
{
    auto it = factories_.find(name);
    TM_CHECK(it != factories_.end()) << "unknown speculative method '" << name << "'";
    return it->second(args);
}

}  // namespace turbomind

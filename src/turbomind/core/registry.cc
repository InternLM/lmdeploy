// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/registry.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/module.h"

namespace turbomind::core {

ModuleRegistry& ModuleRegistry::instance()
{
    static ModuleRegistry reg;
    return reg;
}

void ModuleRegistry::register_type(const std::string& name, Factory factory)
{
    factories_[name] = std::move(factory);
}

std::unique_ptr<Module> ModuleRegistry::create(const std::string& type, const ModuleConfig& config) const
{
    auto it = factories_.find(type);
    if (it == factories_.end()) {
        return nullptr;
    }
    return it->second(config);
}

bool ModuleRegistry::has_type(const std::string& name) const
{
    return factories_.count(name) > 0;
}

}  // namespace turbomind::core

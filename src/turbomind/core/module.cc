// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/module.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/registry.h"

#include <sstream>

namespace turbomind::core {

// ======================================================================
// Module
// ======================================================================

Module::Module() = default;

Module::~Module() = default;

// ----- Type info -----

const char* Module::type() const
{
    return "Module";
}

// ----- Hierarchy (default implementations) -----

Module* Module::add_child(std::string /*name*/, std::unique_ptr<Module> /*child*/)
{
    return nullptr;
}

Module* Module::child(const std::string& /*name*/) const
{
    return nullptr;
}

void Module::for_each_child(std::function<void(const char*, Module*)> /*visitor*/) const
{
    // default: no-op
}

// ----- Parameters (default implementations) -----

Param Module::param(const std::string& /*name*/)
{
    return {};
}

void Module::for_each_param(std::function<void(const char*, Tensor&)> /*visitor*/)
{
    // default: no-op
}

// ----- Lifecycle -----

void Module::prepare()
{
    for_each_child([](const char* /*name*/, Module* child) {
        if (child)
            child->prepare();
    });
}

// ----- Registry-driven child creation -----

std::unique_ptr<Module> Module::create(const ModuleConfig& config)
{
    return ModuleRegistry::instance().create(std::string(config.module_type), config);
}

Module* Module::create_child(const std::string& name, const ModuleConfig& config)
{
    auto mod = create(config);
    if (!mod) {
        return nullptr;
    }
    return add_child(name, std::move(mod));
}

// ----- Lookup -----

Module* Module::get(const std::string& segment)
{
    auto* c = child(segment);
    TM_CHECK(c != nullptr) << "child '" << segment << "' not found in " << type();
    return c;
}

// ----- Verification -----

bool Module::verify(std::vector<std::string>& missing)
{
    // Recurse into children
    for_each_child([&](const char* /*name*/, Module* child) {
        if (child)
            child->verify(missing);
    });

    // Check parameters are initialized
    for_each_param([&](const char* name, Tensor& tensor) {
        if (!tensor) {
            missing.push_back(full_path() + "." + name);
        }
    });

    return missing.empty();
}

// ----- Utilities -----

std::string Module::full_path() const
{
    if (!parent_) {
        return name_;
    }
    std::string pp = parent_->full_path();
    if (pp.empty()) {
        return name_;
    }
    return pp + "." + name_;
}

// ======================================================================
// ModuleList
// ======================================================================

Module* ModuleList::add_child(std::string name, std::unique_ptr<Module> child)
{
    TM_CHECK(child != nullptr);
    TM_CHECK(child->parent_ == nullptr) << "module already has a parent";

    // Parse index before moving name.
    int index = -1;
    {
        std::istringstream iss(name);
        iss >> index;
        if (!iss.eof()) {
            index = -1;
        }
    }

    child->parent_ = this;
    child->name_   = name;

    Module* raw = child.get();
    items_.emplace_back(std::move(name), std::move(child));

    if (index >= 0) {
        if (index >= static_cast<int>(indexed_.size())) {
            indexed_.resize(index + 1, nullptr);
        }
        indexed_[index] = raw;
    }

    return raw;
}

Module* ModuleList::child(const std::string& name) const
{
    for (auto& [n, c] : items_) {
        if (n == name) {
            return c.get();
        }
    }
    return nullptr;
}

void ModuleList::for_each_child(std::function<void(const char*, Module*)> visitor) const
{
    for (auto& [name, c] : items_) {
        visitor(name.c_str(), c.get());
    }
}

int ModuleList::size() const
{
    int n = 0;
    for (auto* p : indexed_) {
        if (p) {
            ++n;
        }
    }
    return n;
}

// ======================================================================
// ModuleList registry
// ======================================================================

TM_MODULE_REGISTER(ModuleList, ModuleListConfig);

}  // namespace turbomind::core

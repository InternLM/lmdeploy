// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace turbomind::core {

// Forward declarations — full definitions in module.h.
struct ModuleConfig;
class Module;

/// Module type registry. Maps type name strings to factory functions.
class ModuleRegistry {
public:
    using Factory = std::function<std::unique_ptr<Module>(const ModuleConfig&)>;

    static ModuleRegistry& instance();

    /// Register a factory under the given type name.
    /// Duplicate names overwrite silently.
    void register_type(const std::string& name, Factory factory);

    /// Convenience overload: derive the factory lambda from the concrete types.
    /// `CfgT` defaults to `ModuleConfig` so callers that accept the base config
    /// need not specify it explicitly.
    template<typename T, typename CfgT = ModuleConfig>
    void register_type(const std::string& name)
    {
        register_type(name, [](const ModuleConfig& cfg) -> std::unique_ptr<Module> {
            return std::make_unique<T>(static_cast<const CfgT&>(cfg));
        });
    }

    /// Create a module instance by type name and typed config.
    /// Returns nullptr if type name is not registered.
    std::unique_ptr<Module> create(const std::string& type, const ModuleConfig& config) const;

    /// Check if a type name is registered.
    bool has_type(const std::string& name) const;

private:
    ModuleRegistry() = default;
    std::map<std::string, Factory> factories_;
};

}  // namespace turbomind::core

#define TM_MODULE_REGISTER(ModuleClass, ConfigType)                                                                    \
    namespace {                                                                                                        \
    static const bool _tm_module_registered_ = [] {                                                                    \
        ::turbomind::core::ModuleRegistry::instance().register_type<ModuleClass, ConfigType>(#ModuleClass);            \
        return true;                                                                                                   \
    }();                                                                                                               \
    }


#include "src/turbomind/core/module.h"
#include "src/turbomind/core/check.h"
#include <optional>

namespace turbomind::core {

Module::Module(): parent_{} {}

Module::~Module()
{
    if (parent_) {
        parent_->remove_module(*this);
        parent_ = {};
    }
}

void Module::register_module(std::string name, Module& module, std::optional<int> index)
{
    module.parent_ = this;
    if (index) {
        name += ".";
        name += std::to_string(*index);
    }
    // std::cout << "register Module " << name << " " << &module << ", parent " << this << "\n";
    modules_.emplace_back(std::move(name), &module);
}

void Module::register_parameter(std::string name, Tensor& param)
{
    // std::cout << "register Parameter " << name << " " << &param << " " << param.layout() << "\n";
    params_.emplace_back(std::move(name), &param);
}

void Module::remove_module(Module& module)
{
    for (auto it = modules_.begin(); it != modules_.end(); ++it) {
        if (it->second == &module) {
            // std::cout << "erase " << it->first << " " << &module << " from " << this << "\n";
            modules_.erase(it);
            return;
        }
    }
    TM_CHECK(0) << "module " << &module << " not found";
}

void Module::remove_parameter(Tensor& param)
{
    for (auto it = params_.begin(); it != params_.end(); ++it) {
        if (it->second == &param) {
            params_.erase(it);
            return;
        }
    }
    TM_CHECK(0) << "param " << &param << " not found";
}

std::unordered_map<std::string, Tensor*> Module::get_parameters() const
{
    std::unordered_map<std::string, Tensor*> m;
    get_parameters_impl({}, m);
    return m;
}

void Module::get_parameters_impl(std::string prefix, std::unordered_map<std::string, Tensor*>& m) const
{
    if (!prefix.empty()) {
        prefix += ".";
    }
    for (const auto& [k, v] : params_) {
        m.emplace(prefix + k, v);
    }
    for (const auto& [k, v] : modules_) {
        v->get_parameters_impl(prefix + k, m);
    }
}

}  // namespace turbomind::core

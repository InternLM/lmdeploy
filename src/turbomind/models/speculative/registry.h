// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/models/speculative/speculative_model.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace turbomind {

class SpeculativeModelRegistry {
public:
    using Factory = std::function<std::unique_ptr<SpeculativeModel>(const SpeculativeModelArgs&)>;

    static SpeculativeModelRegistry& Instance();

    void Register(std::string name, Factory factory);
    bool Contains(std::string_view name) const;

    std::unique_ptr<SpeculativeModel> Create(std::string_view name, const SpeculativeModelArgs& args) const;

private:
    std::map<std::string, Factory, std::less<>> factories_;
};

}  // namespace turbomind

#define TM_REGISTER_SPECULATIVE_MODEL(name, ModelClass)                                                                \
    namespace {                                                                                                        \
    static const bool _tm_speculative_registered_##ModelClass = [] {                                                   \
        ::turbomind::SpeculativeModelRegistry::Instance().Register(                                                    \
            name, [](const ::turbomind::SpeculativeModelArgs& args) {                                                  \
                return std::make_unique<ModelClass>(args);                                                             \
            });                                                                                                        \
        return true;                                                                                                   \
    }();                                                                                                               \
    }

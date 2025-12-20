// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdlib>
#include <sstream>
#include <string>
#include <type_traits>

#include "src/turbomind/utils/logger.h"

namespace turbomind {

template<class E>
auto GetEnv()
{
    static auto value = [] {
        bool is_set{};
        auto x  = E::init();
        using T = decltype(x);
        try {
            if (auto p = std::getenv(E::full_name)) {
                is_set = true;
                if constexpr (std::is_integral_v<T>) {
                    x = std::stoll(p);
                }
                else if constexpr (std::is_floating_point_v<T>) {
                    x = std::stod(p);
                }
                else if constexpr (std::is_same_v<T, std::string>) {
                    x = std::string{p};
                }
                else {
                    static_assert(!std::is_same_v<T, T>, "not implemented");
                }
            }
        }
        catch (...) {
        }
        if (is_set) {
            std::stringstream ss;
            ss << x;
            TM_LOG_INFO("[%s] %s=%s", E::prefix, E::name, ss.str().c_str());
        }
        return x;
    }();
    return value;
}

#define TM_ENV_VAR(prefix_, name_, init_)                                                                              \
    struct prefix_##_##name_ {                                                                                         \
        static auto init()                                                                                             \
        {                                                                                                              \
            return init_;                                                                                              \
        }                                                                                                              \
        static constexpr auto prefix    = #prefix_;                                                                    \
        static constexpr auto name      = #name_;                                                                      \
        static constexpr auto full_name = "TM_" #prefix_ "_" #name_;                                                   \
    }

}  // namespace turbomind

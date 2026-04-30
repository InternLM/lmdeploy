#pragma once

#include <string>
#include <string_view>

#include "src/turbomind/core/context.h"

namespace turbomind::core {

class Scope {
public:
    Scope(const char* name, const char* file, int line);
    Scope(const char* name, const char* file, int line, scope_type type);
    ~Scope();

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

    static std::string trace();
    static int         depth();
};

// Strip return type, parameter list, and template args from __PRETTY_FUNCTION__
// or __FUNCSIG__, leaving "QualifiedName()". For named scopes this is not called.
std::string StripFunctionSignature(std::string_view name);

}  // namespace turbomind::core

#define TM_SCOPE(name) ::turbomind::core::Scope _tm_scope_##__LINE__(name, __FILE__, __LINE__)

#define TM_FUNCTION_SCOPE()                                                                                            \
    ::turbomind::core::Scope _tm_func_scope_##__LINE__(                                                                \
        __PRETTY_FUNCTION__, __FILE__, __LINE__, ::turbomind::core::scope_type::function)

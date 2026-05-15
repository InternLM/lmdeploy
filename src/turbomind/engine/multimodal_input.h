#pragma once

#include "src/turbomind/core/core.h"

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace turbomind {
namespace multimodal {

struct Value;
using Array    = std::vector<Value>;
using Document = std::map<std::string, Value>;

struct Value {
    using Variant = std::variant<std::nullptr_t,  //
                                 bool,
                                 std::int64_t,
                                 double,
                                 Tensor,
                                 std::string,
                                 Array,
                                 Document>;
    Variant data;

    Value(): data(nullptr) {}
    Value(std::nullptr_t): data(nullptr) {}
    Value(bool value): data(value) {}

    template<typename T,
             std::enable_if_t<std::is_integral_v<std::decay_t<T>> && !std::is_same_v<std::decay_t<T>, bool>, int> = 0>
    Value(T value): data(static_cast<std::int64_t>(value))
    {
    }

    template<typename T, std::enable_if_t<std::is_floating_point_v<std::decay_t<T>>, int> = 0>
    Value(T value): data(static_cast<double>(value))
    {
    }

    Value(const char* value): data(std::string(value)) {}
    Value(std::string value): data(std::move(value)) {}
    Value(std::string_view value): data(std::string(value)) {}
    Value(Array value): data(std::move(value)) {}
    Value(Document value): data(std::move(value)) {}
    Value(Tensor value): data(std::move(value)) {}

    bool is_null() const
    {
        return std::holds_alternative<std::nullptr_t>(data);
    }

    template<typename T>
    bool is() const
    {
        return std::holds_alternative<T>(data);
    }

    template<typename T>
    const T& get() const
    {
        return std::get<T>(data);
    }

    template<typename T>
    T& get()
    {
        return std::get<T>(data);
    }
};

}  // namespace multimodal
}  // namespace turbomind

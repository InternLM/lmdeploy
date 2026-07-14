#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <type_traits>
#include <vector>

namespace turbomind::core {

template<template<class...> typename F, class SFINAE, class... Args>
struct is_detected: std::false_type {
};

template<template<class... Args> typename F, class... Args>
struct is_detected<F, std::void_t<F<Args...>>, Args...>: std::true_type {
};

template<class Archive, class T>
using save_t = decltype(save(std::declval<Archive&>(), std::declval<T>()));

template<class Archive, class T>
inline constexpr bool has_save_v = is_detected<save_t, void, Archive, T>::value;

template<class Archive, class T>
using load_t = decltype(load(std::declval<Archive&>(), std::declval<T>()));

template<class Archive, class T>
inline constexpr bool has_load_v = is_detected<load_t, void, Archive, T>::value;

template<class Archive, class T>
using serdes_t = decltype(serdes(std::declval<Archive&>(), std::declval<T>()));

template<class Archive, class T>
inline constexpr bool has_serdes_v = is_detected<serdes_t, void, Archive, T>::value;

template<typename T>
class ArrayWrapper {
public:
    ArrayWrapper(T* t, std::size_t size): t_{t}, size_{size}
    {
        static_assert(std::is_trivially_copyable_v<T>, "ArrayWrapper requires trivially copyable type");
    }

    T* data() const
    {
        return t_;
    }

    std::size_t count() const
    {
        return size_;
    }

    T* const          t_;
    const std::size_t size_;
};

template<typename T>
inline constexpr bool is_array_wrapper_v = std::false_type{};

template<typename T>
inline constexpr bool is_array_wrapper_v<ArrayWrapper<T>> = std::true_type{};

template<class Derived>
struct OutputArchive {
    static constexpr bool is_loading = false;

    template<class T>
    void operator&(T&& x)
    {
        if constexpr (has_save_v<Derived, T>) {
            save(*this, (T &&) x);
        }
        else if constexpr (has_serdes_v<Derived, T>) {
            serdes(*this, (T &&) x);
        }
        else {
            reinterpret_cast<Derived*>(this)->write((T &&) x);
        }
    }
};

template<class Derived>
struct InputArchive {
    static constexpr bool is_loading = true;

    template<class T>
    void operator&(T&& x)
    {
        if constexpr (has_load_v<Derived, T>) {
            load(*this, (T &&) x);
        }
        else if constexpr (has_serdes_v<Derived, T>) {
            serdes(*this, (T &&) x);
        }
        else {
            reinterpret_cast<Derived*>(this)->read((T &&) x);
        }
    }
};

struct BinarySizeArchive: OutputArchive<BinarySizeArchive> {
    size_t size_{};

    size_t size()
    {
        return size_;
    }

    template<class T>
    void write(const T& x)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        size_ += sizeof(x);
    }

    template<class T>
    void write(const ArrayWrapper<T>& arr)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        size_ += sizeof(T) * arr.count();
    }
};

struct BinaryOutputArchive: OutputArchive<BinaryOutputArchive> {

    ArrayWrapper<std::byte> external_;
    size_t                  ptr_;

    BinaryOutputArchive(ArrayWrapper<std::byte> external): external_{external}, ptr_{} {}

    template<class T>
    void write(const T& x)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        auto data = (const std::byte*)&x;
        TM_CHECK_LE(ptr_ + sizeof(T), external_.count());
        std::copy_n(data, sizeof(T), external_.data() + ptr_);
        ptr_ += sizeof(T);
    }

    template<class T>
    void write(const ArrayWrapper<T>& arr)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        auto data = (const std::byte*)arr.data();
        TM_CHECK_LE(ptr_ + sizeof(T) * arr.count(), external_.count());
        std::copy_n(data, sizeof(T) * arr.count(), external_.data() + ptr_);
        ptr_ += sizeof(T) * arr.count();
    }
};

struct BinaryInputArchive: InputArchive<BinaryInputArchive> {

    ArrayWrapper<std::byte> external_;
    size_t                  ptr_;

    BinaryInputArchive(ArrayWrapper<std::byte> external): external_{external}, ptr_{} {}

    template<class T>
    void read(T& x)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        TM_CHECK_LE(ptr_ + sizeof(T), external_.count());
        std::copy_n(external_.data() + ptr_, sizeof(T), (std::byte*)&x);
        ptr_ += sizeof(T);
    }

    template<class T>
    void read(ArrayWrapper<T>&& arr)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        TM_CHECK_LE(ptr_ + sizeof(T) * arr.count(), external_.count());
        std::copy_n(external_.data() + ptr_, sizeof(T) * arr.count(), (std::byte*)arr.data());
        ptr_ += sizeof(T) * arr.count();
    }
};

template<class Archive, class T>
void save(Archive& ar, const std::vector<T>& xs)
{
    // clang-format off
    ar & xs.size();
    if constexpr (std::is_trivially_copyable_v<T>) {
        ar & ArrayWrapper(xs.data(), xs.size());
    }
    else {
        for (const auto& x : xs) {
            ar & x;
        }
    }
    // clang-format on
}

template<class Archive, class T>
void load(Archive& ar, std::vector<T>& xs)
{
    // clang-format off
    decltype(xs.size()) size;
    ar & size;
    xs.resize(size);

    if constexpr (std::is_trivially_copyable_v<T>) {
        ar & ArrayWrapper(xs.data(), size);
    } else {
        for (size_t i = 0; i < size; ++i) {
            ar & xs[i];
        }
    }
    // clang-format on
}

template<class Archive>
void save(Archive& ar, const std::string& s)
{
    // clang-format off
    ar & s.size();
    ar & ArrayWrapper(s.data(), s.size());
    // clang-format on
}

template<class Archive>
void load(Archive& ar, std::string& s)
{
    // clang-format off
    decltype(s.size()) size;
    ar & size;
    s.resize(size);
    ar & ArrayWrapper(s.data(), size);
    // clang-format on
}

template<class Archive, class T>
void save(Archive& ar, const std::shared_ptr<T>& p)
{
    // clang-format off
    ar & (bool)p;
    if (p) {
        ar & (*p);
    }
    // clang-format on
}

template<class Archive, class T>
void load(Archive& ar, std::shared_ptr<T>& p)
{
    // clang-format off
    bool pred;
    ar & pred;
    if (pred) {
        p = std::make_shared<T>();
        ar & (*p);
    }
}

template<class Archive, class T, size_t N>
void serdes(Archive& ar, std::array<T, N>& xs)
{
    // clang-format off
    if constexpr (std::is_trivially_copyable_v<T>) {
        ar & ArrayWrapper(xs.data(), N);
    }
    else {
        for (size_t i = 0; i < N; ++i) {
            ar & xs[i];
        }
    }
    // clang-format on
}

template<class Archive, class... Ts>
void serdes(Archive& ar, std::tuple<Ts...>& tpl)
{
    std::apply([&](auto&... elems) { ((ar & elems), ...); }, tpl);
}

}  // namespace turbomind::core

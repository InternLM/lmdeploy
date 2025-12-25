// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/serdes.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind::comm {

enum class RedOp
{
    kSum,
    kMin,
    kMax,
};

typedef void (*copy_fn)(void* src, int n, void* dst, int offset);

typedef void (*reduce_fn)(void* src, int n, void* dst, int offset);

typedef void (*ser_fn)(void* data, int offset, int n, size_t& size, void* out);

typedef void (*des_fn)(void* data, int offset, int n, void* in, size_t size);

class HostCommImpl {
public:
    virtual ~HostCommImpl();

    virtual int rank() const = 0;

    virtual int n_ranks() const = 0;

    virtual bool is_same_process() const = 0;

    virtual std::shared_ptr<HostCommImpl> Split(int color, int key) = 0;

    virtual void Sync(bool blocking = false) = 0;

    virtual void Broadcast(void*    data,  //
                           int      count,
                           DataType dtype,
                           int      root,
                           copy_fn  copy,
                           ser_fn   ser = nullptr,
                           des_fn   des = nullptr) = 0;

    virtual void AllGather(void*    data,  //
                           int      count,
                           DataType dtype,
                           copy_fn  copy,
                           ser_fn   ser = nullptr,
                           des_fn   des = nullptr) = 0;

    virtual void AllReduce(void* data, int count, DataType dtype, RedOp red_op) = 0;
};

class HostComm {
public:
    HostComm() = default;

    /* implicit */ HostComm(std::shared_ptr<HostCommImpl> impl): impl_{std::move(impl)} {}

    HostCommImpl* operator->() const noexcept
    {
        return impl_.get();
    }

    operator HostCommImpl*() const noexcept
    {
        return impl_.get();
    }

private:
    std::shared_ptr<HostCommImpl> impl_;
};

namespace detail {
template<class T>
void copy_fn(void* src, int n, void* dst, int offset)
{
    std::copy_n((T*)src + offset, n, (T*)dst + offset);
}

template<class T>
void ser_fn(void* data, int offset, int n, size_t& size, void* out)
{
    if (out == nullptr) {
        size = 0;
        core::BinarySizeArchive sa;
        for (int i = 0; i < n; ++i) {
            sa&((T*)data)[offset + i];
        }
        size = sa.size();
    }
    else {
        core::BinaryOutputArchive oa(core::ArrayWrapper((std::byte*)out, size));
        for (int i = 0; i < n; ++i) {
            oa&((T*)data)[offset + i];
        }
    }
}

template<class T>
void des_fn(void* data, int offset, int n, void* in, size_t size)
{
    core::BinaryInputArchive ia(core::ArrayWrapper((std::byte*)in, size));
    for (int i = 0; i < n; ++i) {
        ia&((T*)data)[offset + i];
    }
}

}  // namespace detail

//////////////////////////////////////////////////////////////////////////////////
// Typed array interface
template<class T>
void Broadcast(HostCommImpl* comm, T* data, int n, int root)
{
    if constexpr (std::is_trivially_copyable_v<T>) {
        comm->Broadcast(data, sizeof(T) * n, data_type_v<uint8_t>, root, detail::copy_fn<uint8_t>);
    }
    else {
        if (comm->is_same_process()) {
            /// TODO: Constness should be considered
            comm->Broadcast(data, n, kNull, root, detail::copy_fn<T>);
        }
        else {
            comm->Broadcast(data, n, kNull, root, detail::copy_fn<T>, detail::ser_fn<T>, detail::des_fn<T>);
        }
    }
}

template<class T>
void AllGather(HostCommImpl* comm, T* data, int n)
{
    if constexpr (std::is_trivially_copyable_v<T>) {
        comm->AllGather(data, sizeof(T) * n, data_type_v<uint8_t>, detail::copy_fn<uint8_t>);
    }
    else {
        if (comm->is_same_process()) {
            /// TODO: Constness should be considered
            comm->AllGather(data, n, kNull, detail::copy_fn<T>);
        }
        else {
            comm->AllGather(data, n, kNull, detail::copy_fn<T>, detail::ser_fn<T>, detail::des_fn<T>);
        }
    }
}

template<class T>
void AllReduce(HostCommImpl* comm, T* data, int n, RedOp red_op)
{
    comm->AllReduce(data, n, data_type_v<T>, red_op);
}

//////////////////////////////////////////////////////////////////////////////////
// Typed value interface
template<class T>
void Broadcast(HostCommImpl* comm, T& value, int root)
{
    Broadcast(comm, &value, 1, root);
}

template<class T>
std::vector<T> AllGather(HostCommImpl* comm, const T& value)
{
    std::vector<T> ret(comm->n_ranks());
    ret.at(comm->rank()) = value;
    AllGather(comm, ret.data(), 1);
    return ret;
}

template<class T>
T AllReduce(HostCommImpl* comm, const T& value, RedOp red_op)
{
    T tmp = value;
    AllReduce(comm, &tmp, 1, red_op);
    return tmp;
}

class HostGroupId {
public:
    virtual ~HostGroupId() = default;

    virtual void Initialize()             = 0;
    virtual void Export(std::ostream& os) = 0;
    virtual void Import(std::istream& is) = 0;

    virtual HostComm CreateCommunicator(int n_ranks, int rank, int node_rank = 0) = 0;
};

std::unique_ptr<HostGroupId> CreateHostGroupId(const std::string& backend);

}  // namespace turbomind::comm

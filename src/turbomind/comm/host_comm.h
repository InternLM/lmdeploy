// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "src/turbomind/utils/Tensor.h"

namespace turbomind::comm {

enum class RedOp
{
    kSum,
    kMin,
    kMax,
};

typedef void (*copy_fn)(void* src, int n, void* dst, int offset);

typedef void (*reduce_fn)(void* src, int n, void* dst, int offset);

class HostCommImpl {
public:
    virtual ~HostCommImpl();

    virtual int rank() const = 0;

    virtual int n_ranks() const = 0;

    virtual bool is_same_process() const = 0;

    virtual std::shared_ptr<HostCommImpl> Split(int color, int key) = 0;

    virtual void Sync() = 0;

    virtual void Broadcast(void* data, int count, DataType dtype, int root, copy_fn copy) = 0;

    virtual void AllGather(void* data, int count, DataType dtype, copy_fn copy) = 0;

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

}  // namespace detail

//////////////////////////////////////////////////////////////////////////////////
// Typed array interface
template<class T>
void Broadcast(HostCommImpl* comm, T* data, int n, int root)
{
    if constexpr (std::is_trivially_copyable_v<T>) {
        comm->Broadcast((char*)data, sizeof(T) * n, TYPE_INT8, root, detail::copy_fn<char>);
    }
    else {
        if (comm->is_same_process()) {
            /// TODO: Constness should be considered
            comm->Broadcast(data, n, TYPE_INVALID, root, detail::copy_fn<T>);
        }
        else {
            throw std::runtime_error("not implemented");
        }
    }
}

template<class T>
void AllGather(HostCommImpl* comm, T* data, int n)
{
    if constexpr (std::is_trivially_copyable_v<T>) {
        comm->AllGather(data, sizeof(T) * n, TYPE_INT8, detail::copy_fn<char>);
    }
    else {
        if (comm->is_same_process()) {
            /// TODO: Constness should be considered
            comm->AllGather(data, n, TYPE_INVALID, detail::copy_fn<T>);
        }
        else {
            /// serialize data
            throw std::runtime_error("not implemented");
        }
    }
}

template<class T>
void AllReduce(HostCommImpl* comm, T* data, int n, RedOp red_op)
{
    comm->AllReduce(data, n, getTensorType<T>(), red_op);
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

    virtual HostComm CreateCommunicator(int n_ranks, int rank) = 0;
};

std::unique_ptr<HostGroupId> CreateHostGroupId(const std::string& backend);

}  // namespace turbomind::comm

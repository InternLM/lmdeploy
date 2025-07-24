// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "src/turbomind/comm/serialize.h"
#include "src/turbomind/core/data_type.h"
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

    virtual void Send(void* data, int count, DataType dtype, int dst) = 0;

    virtual void Recv(void* data, int count, DataType dtype, int src, copy_fn copy) = 0;
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
        comm->Broadcast(data, sizeof(T) * n, data_type_v<uint8_t>, root, detail::copy_fn<uint8_t>);
    }
    else {
        if (comm->is_same_process()) {
            /// TODO: Constness should be considered
            comm->Broadcast(data, n, kNull, root, detail::copy_fn<T>);
        }
        else {
            try {
                // buf may have different size on different ranks
                std::vector<char> buf;
                serialize(data, n, buf);
                size_t size = buf.size();
                Broadcast(comm, &size, 1, root);
                buf.resize(size);
                comm->Broadcast(buf.data(), buf.size(), data_type_v<uint8_t>, root, detail::copy_fn<char>);
                if (comm->rank() != root) {
                    // some field in data may be not shared by all rank
                    deserialize(data, n, buf);
                }
            }
            catch (const std::invalid_argument& e) {
                TM_LOG_ERROR("Broadcast failed: %s", e.what());
                throw;
            }
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
            try {
                // buf may have different size on different ranks
                std::vector<char> rbuf;
                for (int i = 0; i < n; ++i) {
                    std::vector<char> ibuf;
                    serialize(data + n * comm->rank() + i, 1, ibuf);
                    rbuf.insert(rbuf.end(), ibuf.begin(), ibuf.end());
                }
                int size = rbuf.size();
                comm->AllReduce(&size, 1, data_type_v<int>, RedOp::kMax);
                std::vector<char> buf(size * comm->n_ranks());
                std::memcpy(buf.data() + comm->rank() * size, rbuf.data(), rbuf.size());
                comm->AllGather(buf.data(), size, data_type_v<uint8_t>, detail::copy_fn<char>);
                for (int i = 0; i < comm->n_ranks(); ++i) {
                    if (i != comm->rank()) {
                        // some field in data may be not shared by all rank
                        deserialize(
                            data + n * i, n, std::vector<char>(buf.begin() + i * size, buf.begin() + (i + 1) * size));
                    }
                }
            }
            catch (const std::invalid_argument& e) {
                TM_LOG_ERROR("AllGather failed: %s", e.what());
                throw;
            }
        }
    }
}

template<class T>
void AllReduce(HostCommImpl* comm, T* data, int n, RedOp red_op)
{
    static_assert(std::is_trivially_copyable_v<T>, "AllReduce only supports trivially copyable types");
    comm->AllReduce(data, n, data_type_v<T>, red_op);
}

template<class T>
void Send(HostCommImpl* comm, T* data, int n, int dst)
{
    if constexpr (std::is_trivially_copyable_v<T>) {
        comm->Send(data, sizeof(T) * n, data_type_v<uint8_t>, dst);
    }
    else {
        if (comm->is_same_process()) {
            comm->Send(data, n, kNull, dst);
        }
        else {
            try {
                std::vector<char> buf;
                for (int i = 0; i < n; ++i) {
                    std::vector<char> ibuf;
                    serialize(data + i, 1, ibuf);
                    buf.insert(buf.end(), ibuf.begin(), ibuf.end());
                }
                uint64_t size = buf.size();
                comm->Send(&size, 1, data_type_v<uint64_t>, dst);
                comm->Send(buf.data(), buf.size(), data_type_v<uint8_t>, dst);
            }
            catch (const std::invalid_argument& e) {
                TM_CHECK(0) << "Send failed: " << e.what();
            }
        }
    }
}

template<class T>
void Recv(HostCommImpl* comm, T* data, int n, int src)
{
    if constexpr (std::is_trivially_copyable_v<T>) {
        comm->Recv(data, sizeof(T) * n, data_type_v<uint8_t>, src, detail::copy_fn<uint8_t>);
    }
    else {
        if (comm->is_same_process()) {
            comm->Recv(data, n, kNull, src, detail::copy_fn<T>);
        }
        else {
            try {
                uint64_t size;
                comm->Recv(&size, 1, data_type_v<uint64_t>, src, detail::copy_fn<int>);
                std::vector<char> buf(size);
                comm->Recv(buf.data(), size, data_type_v<uint8_t>, src, detail::copy_fn<char>);
                deserialize(data, n, buf);
            }
            catch (const std::invalid_argument& e) {
                TM_CHECK(0) << "Recv failed: " << e.what();
            }
        }
    }
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

template<class T>
void Send(HostCommImpl* comm, T& value, int dst)
{
    Send(comm, &value, 1, dst);
}

template<class T>
void Recv(HostCommImpl* comm, T& value, int src)
{
    Recv(comm, &value, 1, src);
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

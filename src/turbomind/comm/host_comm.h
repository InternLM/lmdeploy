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

enum class CommType
{
    kIntra,
    kInter,
    kHybrid,
};

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

    virtual void* query(CommType type) const
    {
        return nullptr;
    }

    template<typename T>
    T* as() const
    {
        return static_cast<T*>(query(T::kCommType));
    }

    virtual int rank() const = 0;

    virtual int n_ranks() const = 0;

    virtual bool is_same_process() const = 0;

    virtual std::shared_ptr<HostCommImpl> Split(int color, int key) = 0;

    virtual void Sync(bool blocking = false) = 0;

    virtual void Broadcast(void* data, int count, DataType dtype, int root, copy_fn copy) = 0;

    virtual void AllGather(void* data, int count, DataType dtype, copy_fn copy) = 0;

    virtual void AllReduce(void* data, int count, DataType dtype, RedOp red_op) = 0;
};

class IpcHostCommImpl: public HostCommImpl {
public:
    static constexpr CommType kCommType = CommType::kInter;

    virtual void* query(CommType type) const override
    {
        if (type == kCommType) {
            return const_cast<IpcHostCommImpl*>(this);
        }
        return HostCommImpl::query(type);
    }
};

class HybridHostCommImpl: public HostCommImpl {
public:
    static constexpr CommType kCommType = CommType::kHybrid;

    virtual IpcHostCommImpl* inter_comm() const = 0;

    virtual HostCommImpl* intra_comm() const = 0;

    virtual const std::unordered_map<int, int>& get_rank_to_intra() const = 0;

    virtual const std::vector<int>& get_rank_to_inter() const = 0;

    virtual void* query(CommType type) const override
    {
        if (type == kCommType) {
            return const_cast<HybridHostCommImpl*>(this);
        }
        return HostCommImpl::query(type);
    }
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
            auto process = [&](IpcHostCommImpl* ipc_comm, int root) {
                // serialize on root rank and deserialize on other ranks
                core::BinarySizeArchive sa;
                if (ipc_comm->rank() == root) {
                    for (int i = 0; i < n; ++i) {
                        sa& data[i];
                    }
                }
                size_t buf_size = sa.size();
                ipc_comm->Broadcast(&buf_size, 1, data_type_v<size_t>, root, detail::copy_fn<size_t>);

                core::BinaryOutputArchive oa(buf_size);
                if (ipc_comm->rank() == root) {
                    for (int i = 0; i < n; ++i) {
                        oa& data[i];
                    }
                    TM_CHECK_EQ(oa.bytes().size(), buf_size);
                }
                else {
                    oa.bytes().resize(buf_size);
                }
                ipc_comm->Broadcast(oa.bytes().data(), buf_size, data_type_v<uint8_t>, root, detail::copy_fn<uint8_t>);

                if (ipc_comm->rank() != root) {
                    core::BinaryInputArchive ia(std::move(oa.bytes()));
                    for (int i = 0; i < n; ++i) {
                        ia& data[i];
                    }
                }
            };
            if (auto hybrid_comm = comm->as<HybridHostCommImpl>(); hybrid_comm) {
                auto  inter_comm    = TM_CHECK_NOTNULL(hybrid_comm->inter_comm());
                auto  intra_comm    = TM_CHECK_NOTNULL(hybrid_comm->intra_comm());
                auto& rank_to_intra = hybrid_comm->get_rank_to_intra();
                auto& rank_to_inter = hybrid_comm->get_rank_to_inter();
                bool  root_node     = rank_to_intra.count(root) > 0;  // root on this node
                if (root_node) {
                    intra_comm->Broadcast(data, n, kNull, rank_to_intra.at(root), detail::copy_fn<T>);
                }
                if (intra_comm->rank() == 0) {
                    process(inter_comm, rank_to_inter[root]);
                }
                if (!root_node) {
                    intra_comm->Broadcast(data, n, kNull, 0, detail::copy_fn<T>);
                }
            }
            else {
                process(TM_CHECK_NOTNULL(comm->as<IpcHostCommImpl>()), root);
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
            // buf may have different size on different ranks
            core::BinarySizeArchive sa;
            for (int i = 0; i < n; ++i) {
                sa& data[comm->rank() * n + i];
            }

            std::vector<int> sizes(comm->n_ranks());
            sizes[comm->rank()] = sa.size();
            comm->AllGather(sizes.data(), 1, data_type_v<int>, detail::copy_fn<int>);
            auto max_size = *std::max_element(sizes.begin(), sizes.end());

            std::vector<std::byte> bytes;
            bytes.reserve(max_size * comm->n_ranks());
            auto buffer = core::ArrayWrapper(bytes.data(), max_size * comm->n_ranks());
            auto oa     = core::BinaryOutputExArchive(buffer).offset(comm->rank() * max_size);
            for (int i = 0; i < n; ++i) {
                oa& data[comm->rank() * n + i];
            }
            comm->AllGather(bytes.data(), max_size, data_type_v<uint8_t>, detail::copy_fn<uint8_t>);

            for (int i = 0; i < comm->n_ranks(); ++i) {
                if (i != comm->rank()) {
                    // some field in data may be not shared by all rank
                    auto buffer_i = core::ArrayWrapper<std::byte>(bytes.data() + i * max_size, sizes[i]);
                    core::BinaryInputExArchive ia(buffer_i);
                    for (int j = 0; j < n; ++j) {
                        ia& data[n * i + j];
                    }
                }
            }
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

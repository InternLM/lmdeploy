// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <atomic>
#include <deque>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include "src/turbomind/models/llama/Barrier.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind::comm {

enum class RedOp {
    kSum,
    kMin,
    kMax,
};

class HostComm {
public:
    struct State {
        explicit State(int n): channels(n * n), barrier(n) {}
        std::deque<std::atomic<void*>> channels;
        turbomind::Barrier             barrier;
    };

    typedef void (*copy_fn)(void* src, int n, void* dst, int offset);

    typedef void (*reduce_fn)(void* src, int n, void* accum);

    HostComm(int n_ranks, int rank, std::shared_ptr<State> state);

    int rank(int group = 0) const
    {
        return groups_.at(group).mapping.at(rank_);
    }

    int n_ranks(int group = 0) const
    {
        return groups_.at(group).ranks.size();
    }

    bool is_same_process(int group = 0) const
    {
        return true;
    }

    int Split(const std::vector<int>& ranks, int parent_group = 0);

    int Split(int color, int key, int parent_group = 0);

    void Sync(int group = 0);

    void Broadcast(void* data, int count, DataType dtype, int root, copy_fn copy_fn, int group);

    void Allgather(void* all_data, int count, DataType dtype, copy_fn copy_fn, int group);

    void Allreduce(const void* src, void* dst, int count, DataType dtype, RedOp red_op, int group);

private:
    struct Group {
        std::vector<int> ranks;    // group rank -> global rank
        std::vector<int> mapping;  // global rank -> group rank
    };

    std::atomic<void*>& channel(int from, int to)
    {
        return state_->channels[from * n_ranks_ + to];
    }

private:
    int n_ranks_;
    int rank_;

    std::shared_ptr<State> state_;
    std::vector<Group>     groups_;
};

namespace detail {

template<class T>
void copy_fn(void* src, int n, void* dst, int offset)
{
    std::copy_n((T*)src + offset, n, (T*)dst + offset);
}

}  // namespace detail

template<class T>
void Broadcast(HostComm& comm, T* data, int n, int root, int group = 0)
{
    if constexpr (std::is_trivially_copyable_v<T>) {
        comm.Broadcast((char*)data, sizeof(T) * n, TYPE_INT8, root, detail::copy_fn<char>, group);
    }
    else {
        if (comm.is_same_process(group)) {
            comm.Broadcast(data, n, TYPE_INVALID, root, detail::copy_fn<T>, group);
        }
        else {
            /// serialize data
            throw std::runtime_error("not implemented");
        }
    }
}

template<class T>
void Allgather(HostComm& comm, T* all_data, int n, int group = 0)
{
    if constexpr (std::is_trivially_copyable_v<T>) {
        comm.Allgather(all_data, sizeof(T) * n, TYPE_INT8, detail::copy_fn<char>, group);
    }
    else {
        if (comm.is_same_process(group)) {
            comm.Allgather(all_data, n, TYPE_INVALID, detail::copy_fn<T>, group);
        }
        else {
            /// serialize data
            throw std::runtime_error("not implemented");
        }
    }
}

template<class T>
void Allreduce(HostComm& comm, const T* src, T* dst, int n, RedOp red_op, int group = 0)
{
    comm.Allreduce(src, dst, n, getTensorType<T>(), red_op, group);
}

template<class T>
void Broadcast(HostComm& comm, T& value, int root, int group = 0)
{
    Broadcast(comm, &value, 1, root, group);
}

template<class T>
std::vector<T> Allgather(HostComm& comm, const T& value, int group = 0)
{
    std::vector<T> ret(comm.n_ranks(group));
    ret[comm.rank(group)] = value;
    Allgather(comm, ret.data(), 1, group);
    return ret;
}

template<class T>
T Allreduce(HostComm& comm, const T& value, RedOp red_op, int group = 0)
{
    T dst{};
    Allreduce(comm, &value, &dst, 1, red_op, group);
    return dst;
}

class HostGroupId {
public:
    virtual ~HostGroupId() = default;

    virtual void Initialize()             = 0;
    virtual void Export(std::ostream& os) = 0;
    virtual void Import(std::istream& is) = 0;

    virtual std::unique_ptr<HostComm> CreateHostCommunicator(int rank, int n_ranks) = 0;
};

std::unique_ptr<HostGroupId> CreateHostGroupId(const std::string& backend);

}  // namespace turbomind::comm

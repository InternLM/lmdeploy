// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <new>

#include "src/turbomind/comm/barrier.h"
#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/serdes.h"
namespace turbomind::comm {

struct ThreadCommImpl: public HostCommImpl {

    class State {
    public:
        explicit State(int n): n_{n}, channels_(n * n), barrier_{n} {}

        std::atomic<void*>& channel(int from, int to)
        {
            return channels_[from * n_ + to];
        }

        void sync()
        {
            barrier_.arrive_and_wait();
        }

    private:
        int                            n_;
        std::deque<std::atomic<void*>> channels_;
        Barrier                        barrier_;
    };

    std::shared_ptr<State> state_;

    int n_ranks_;
    int rank_;

    ThreadCommImpl(int n_ranks, std::shared_ptr<State> state, int rank):
        state_{std::move(state)}, n_ranks_{n_ranks}, rank_{rank}
    {
    }

    int rank() const override
    {
        return rank_;
    }

    int n_ranks() const override
    {
        return n_ranks_;
    }

    bool is_same_process() const override
    {
        return true;
    }

    std::atomic<void*>& channel(int from, int to)
    {
        return state_->channel(from, to);
    }

    std::shared_ptr<HostCommImpl> Split(int color, int key) override
    {
        TM_CHECK(color >= 0);

        auto ranks = comm::AllGather(this, std::make_tuple(color, key, rank_));

        auto same_color = [&](auto x) { return std::get<0>(x) == color; };
        ranks.erase(std::stable_partition(ranks.begin(), ranks.end(), same_color), ranks.end());

        std::stable_sort(ranks.begin(), ranks.end(), [](auto& a, auto& b) { return a < b; });

        std::shared_ptr<State> state;

        int rank = -1;
        for (int i = 0; i < ranks.size(); ++i) {
            if (std::get<2>(ranks[i]) == rank_) {
                rank = i;
            }
        }

        TM_CHECK_GE(rank, 0);

        if (rank == 0) {
            state = std::make_shared<State>(ranks.size());
        }

        auto states = comm::AllGather(this, state);
        if (rank != 0) {
            const int root = std::get<2>(ranks[0]);
            state          = states[root];
        }

        return std::make_shared<ThreadCommImpl>(ranks.size(), state, rank);
    }

    void Sync(bool blocking) override
    {
        if (n_ranks_ == 1) {
            return;
        }

        if (blocking) {
            state_->sync();
            return;
        }

        for (int r = 0; r < n_ranks_; ++r) {
            if (r != rank_) {
                auto& c = channel(rank_, r);
                void* expected{};
                while (!c.compare_exchange_weak(expected, (void*)1, std::memory_order_release)) {
                    expected = {};
                }
            }
        }
        for (int r = 0; r < n_ranks_; ++r) {
            if (r != rank_) {
                auto& c        = channel(r, rank_);
                void* expected = (void*)1;
                while (!c.compare_exchange_weak(expected, nullptr, std::memory_order_acquire)) {
                    expected = (void*)1;
                }
            }
        }
    }

    void Broadcast(void* data, int count, DataType dtype, int root, copy_fn copy, ser_fn ser, des_fn des) override
    {
        TM_CHECK(copy);
        if (n_ranks_ == 1) {
            return;
        }
        // transform root to global rank
        if (rank_ == root) {
            for (int r = 0; r < n_ranks_; ++r) {
                if (r != rank_) {
                    auto& c = channel(rank_, r);
                    void* expected{};
                    while (!c.compare_exchange_weak(expected, data, std::memory_order_release)) {
                        expected = {};
                    }
                }
            }
            for (int r = 0; r < n_ranks_; ++r) {
                if (r != rank_) {
                    auto& c = channel(rank_, r);
                    while (c.load(std::memory_order_relaxed)) {}
                }
            }
        }
        else {
            auto& c = channel(root, rank_);
            void* incoming{};
            while (!(incoming = c.load(std::memory_order_acquire))) {}
            copy(incoming, count, data, 0);
            c.store(nullptr, std::memory_order_relaxed);
        }
    }

    void AllGather(void* data, int count, DataType dtype, copy_fn copy, ser_fn ser, des_fn des) override
    {
        TM_CHECK(copy);
        if (n_ranks_ == 1) {
            return;
        }
        for (int r = 0; r < n_ranks_; ++r) {
            if (r != rank_) {
                auto& c = channel(rank_, r);
                void* expected{};
                while (!c.compare_exchange_weak(expected, data, std::memory_order_release)) {
                    expected = {};
                }
            }
        }
        for (int r = 0; r < n_ranks_; ++r) {
            if (r != rank_) {
                auto& c = channel(r, rank_);
                void* incoming{};
                while (!(incoming = c.load(std::memory_order_acquire))) {}
                copy(incoming, count, data, r * count);
                c.store(nullptr, std::memory_order_relaxed);
            }
        }
        for (int r = 0; r < n_ranks_; ++r) {
            if (r != rank_) {
                auto& c = channel(rank_, r);
                while (c.load(std::memory_order_relaxed)) {}
            }
        }
    }

    template<class T, RedOp op>
    static void reduce(void* src, int n, void* dst, int offset)
    {
        for (int i = 0; i < n; ++i) {
            auto& s = *((T*)src + offset + i);
            auto& a = *((T*)dst + offset + i);
            if constexpr (op == RedOp::kSum) {
                a += s;
            }
            else if constexpr (op == RedOp::kMin) {
                a = std::min(a, s);
            }
            else if constexpr (op == RedOp::kMax) {
                a = std::max(a, s);
            }
            else {
                static_assert(sizeof(T) != sizeof(T), "not implemented");
            }
        }
    }

    static reduce_fn get_reduce(DataType dtype, RedOp red_op)
    {
        auto dispatch_op = [&](auto t) -> reduce_fn {
            using T = decltype(t);
            switch (red_op) {
                case RedOp::kSum:
                    return reduce<T, RedOp::kSum>;
                case RedOp::kMax:
                    return reduce<T, RedOp::kMax>;
                case RedOp::kMin:
                    return reduce<T, RedOp::kMin>;
                default:
                    return {};
            }
        };
        auto dispatch = [&]() -> reduce_fn {
            switch (dtype) {
                case kInt32:
                    return dispatch_op(int32_t{});
                case kInt64:
                    return dispatch_op(int64_t{});
                case kUint32:
                    return dispatch_op(uint32_t{});
                case kUint64:
                    return dispatch_op(uint64_t{});
                default:
                    return {};
            }
        };
        if (auto fn = dispatch()) {
            return fn;
        }
        else {
            throw std::runtime_error("not implemented");
            return {};
        }
    }

    void AllReduce(void* data, int count, DataType dtype, RedOp red_op) override
    {
        const auto reduce    = get_reduce(dtype, red_op);
        const auto elem_size = byte_size(dtype);
        if (n_ranks_ == 1) {
            return;
        }
        std::unique_ptr<char[]> tmp((char*)::operator new[](elem_size* count));
        std::copy_n((char*)data, elem_size * count, tmp.get());
        for (int r = 0; r < n_ranks_; ++r) {
            if (r != rank_) {
                auto& c = channel(rank_, r);
                void* expected{};
                while (!c.compare_exchange_weak(expected, (void*)tmp.get(), std::memory_order_release)) {
                    expected = {};
                }
            }
        }
        for (int r = 0; r < n_ranks_; ++r) {
            if (r != rank_) {
                auto& c = channel(r, rank_);
                void* incoming{};
                while (!(incoming = c.load(std::memory_order_acquire))) {}
                reduce(incoming, count, data, 0);
                c.store(nullptr, std::memory_order_relaxed);
            }
        }
        for (int r = 0; r < n_ranks_; ++r) {
            if (r != rank_) {
                auto& c = channel(rank_, r);
                while (c.load(std::memory_order_relaxed)) {}
            }
        }
    }
};

class ThreadGroupId: public HostGroupId {
public:
    void Initialize() override
    {
        internal_ = std::make_shared<Internal>();
    }

    void Export(std::ostream& os) override
    {
        TM_CHECK((bool)internal_);  // `Initialize` must come befor `Export`

        const void* ptr = this;
        os.write((const char*)&ptr, sizeof(ptr));
    }

    void Import(std::istream& is) override
    {
        void* ptr{};
        is.read((char*)&ptr, sizeof(ptr));
        internal_ = reinterpret_cast<ThreadGroupId*>(ptr)->internal_;

        TM_CHECK((bool)internal_);
    }

    HostComm CreateCommunicator(int n_ranks, int rank, int node_rank = 0) override
    {
        auto init_shared_state = [&] {  //
            internal_->state = std::make_shared<ThreadCommImpl::State>(n_ranks);
        };

        TM_CHECK((bool)internal_);

        // One of the rank initialize the shared state
        std::call_once(internal_->flag, init_shared_state);

        TM_CHECK((bool)internal_->state);

        auto impl = std::make_shared<ThreadCommImpl>(n_ranks, internal_->state, rank);

        return std::static_pointer_cast<HostCommImpl>(impl);
    }

private:
    struct Internal {
        std::once_flag                         flag;
        std::shared_ptr<ThreadCommImpl::State> state;
    };

private:
    std::shared_ptr<Internal> internal_;
};

std::unique_ptr<HostGroupId> CreateThreadGroupId()
{
    return std::make_unique<ThreadGroupId>();
}

template<class Archive>
void save(Archive& ar, const std::shared_ptr<ThreadCommImpl::State>& p)
{
    TM_CHECK(false) << "should never be called";
}

template<class Archive>
void load(Archive& ar, std::shared_ptr<ThreadCommImpl::State>& p)
{
    TM_CHECK(false) << "should never be called";
}

}  // namespace turbomind::comm

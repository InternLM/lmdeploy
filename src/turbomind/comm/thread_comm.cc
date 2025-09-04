// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
namespace turbomind::comm {

struct ThreadCommImpl: public HostCommImpl {

    constexpr static int kMaxSplits = 32;

    class State {
    public:
        explicit State(int n): n_{n}, channels_(n * n * kMaxSplits) {}
        std::atomic<void*>& channel(int from, int to)
        {
            return channels_[from * n_ + to];
        }

        int next_offset()
        {
            std::lock_guard lock{mutex_};
            TM_CHECK(offset_ < channels_.size());
            offset_ += n_;
            return offset_;
        }

    private:
        std::mutex                     mutex_;
        int                            offset_{0};
        int                            n_;
        std::deque<std::atomic<void*>> channels_;
    };

    std::shared_ptr<State> state_;

    int rank_;  // global rank

    int offset_{0};

    std::vector<int> l2g_;
    std::vector<int> g2l_;

    ThreadCommImpl(int n_ranks, std::shared_ptr<State> state, int rank): state_{std::move(state)}, rank_{rank}
    {
        l2g_.resize(n_ranks);
        std::iota(l2g_.begin(), l2g_.end(), 0);
        g2l_ = l2g_;
    }

    ThreadCommImpl(std::vector<int> l2g, std::vector<int> g2l, std::shared_ptr<State> state, int rank):
        state_{std::move(state)}, rank_{rank}, l2g_{std::move(l2g)}, g2l_{std::move(g2l)}
    {
        int offset = (this->rank() == 0) ? state_->next_offset() : 0;
        comm::Broadcast(this, offset, 0);
        offset_ = offset;
    }

    int rank() const override
    {
        return g2l_.at(rank_);
    }

    int n_ranks() const override
    {
        return l2g_.size();
    }

    bool is_same_process() const override
    {
        return true;
    }

    std::atomic<void*>& channel(int from, int to)
    {
        return state_->channel(from + offset_, to);
    }

    std::shared_ptr<HostCommImpl> Split(int color, int key) override
    {
        TM_CHECK(color >= 0);
        TM_CHECK(g2l_[rank_] >= 0);

        // `g2l_[rank_]` imposes proper ordering when keys are equal
        auto vec = comm::AllGather(this, std::make_tuple(color, key, g2l_[rank_]));

        auto last = std::stable_partition(vec.begin(), vec.end(), [&](auto x) {  //
            return std::get<0>(x) == color;
        });
        vec.erase(last, vec.end());
        std::stable_sort(vec.begin(), vec.end(), [](auto& a, auto& b) {  //
            return a < b;
        });

        std::vector<int> l2g;
        std::vector<int> g2l(g2l_.size(), -1);

        for (size_t i = 0; i < vec.size(); ++i) {
            int r = l2g_.at(std::get<2>(vec[i]));
            l2g.push_back(r);
            g2l[r] = i;
        }

        return std::make_shared<ThreadCommImpl>(std::move(l2g), std::move(g2l), state_, rank_);
    }

    void Sync() override
    {
        if (n_ranks() == 1) {
            return;
        }
        for (const auto& r : l2g_) {
            if (r != rank_) {
                auto& c = channel(rank_, r);
                void* expected{};
                while (!c.compare_exchange_weak(expected, (void*)1, std::memory_order_release)) {
                    expected = {};
                }
            }
        }
        for (const auto& r : l2g_) {
            if (r != rank_) {
                auto& c        = channel(r, rank_);
                void* expected = (void*)1;
                while (!c.compare_exchange_weak(expected, nullptr, std::memory_order_acquire)) {
                    expected = (void*)1;
                }
            }
        }
    }

    void Broadcast(void* data, int count, DataType dtype, int root, copy_fn copy) override
    {
        TM_CHECK(copy);
        if (n_ranks() == 1) {
            return;
        }
        // transform root to global rank
        root = l2g_.at(root);
        if (rank_ == root) {
            for (const auto& r : l2g_) {
                if (r != rank_) {
                    auto& c = channel(rank_, r);
                    void* expected{};
                    while (!c.compare_exchange_weak(expected, data, std::memory_order_release)) {
                        expected = {};
                    }
                }
            }
            for (const auto& r : l2g_) {
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

    void AllGather(void* data, int count, DataType dtype, copy_fn copy) override
    {
        TM_CHECK(copy);
        if (n_ranks() == 1) {
            return;
        }
        for (const auto& r : l2g_) {
            if (r != rank_) {
                auto& c = channel(rank_, r);
                void* expected{};
                while (!c.compare_exchange_weak(expected, data, std::memory_order_release)) {
                    expected = {};
                }
            }
        }
        for (const auto& r : l2g_) {
            if (r != rank_) {
                auto& c = channel(r, rank_);
                void* incoming{};
                while (!(incoming = c.load(std::memory_order_acquire))) {}
                copy(incoming, count, data, g2l_[r] * count);
                c.store(nullptr, std::memory_order_relaxed);
            }
        }
        for (const auto& r : l2g_) {
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
        if (n_ranks() == 1) {
            return;
        }
        std::unique_ptr<char[]> tmp((char*)::operator new[](elem_size* count));
        std::copy_n((char*)data, elem_size * count, tmp.get());
        for (const auto& r : l2g_) {
            if (r != rank_) {
                auto& c = channel(rank_, r);
                void* expected{};
                while (!c.compare_exchange_weak(expected, (void*)tmp.get(), std::memory_order_release)) {
                    expected = {};
                }
            }
        }
        for (const auto& r : l2g_) {
            if (r != rank_) {
                auto& c = channel(r, rank_);
                void* incoming{};
                while (!(incoming = c.load(std::memory_order_acquire))) {}
                reduce(incoming, count, data, 0);
                c.store(nullptr, std::memory_order_relaxed);
            }
        }
        for (const auto& r : l2g_) {
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

    HostComm CreateCommunicator(int n_ranks, int rank) override
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

}  // namespace turbomind::comm

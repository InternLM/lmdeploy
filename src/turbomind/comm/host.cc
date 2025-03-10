

#include <algorithm>
#include <mutex>

#include "src/turbomind/comm/host.h"

#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::comm {

HostComm::HostComm(int n_ranks, int rank, std::shared_ptr<State> state): n_ranks_{n_ranks}, rank_{rank}, state_{state}
{
    Group g;
    g.ranks.resize(n_ranks);
    std::iota(g.ranks.begin(), g.ranks.end(), 0);
    g.mapping = g.ranks;  // identity mapping
    groups_.push_back(g);
}

int HostComm::Split(const std::vector<int>& ranks, int parent_group)
{
    const int index = groups_.size();

    auto& p = groups_.at(parent_group);
    auto& g = groups_.emplace_back();

    g.mapping.resize(n_ranks_, -1);
    for (size_t i = 0; i < ranks.size(); ++i) {
        int rank = p.ranks.at(ranks[i]);
        g.ranks.push_back(rank);
        g.mapping[rank] = i;
    }

    return index;
}

int HostComm::Split(int color, int key, int parent_group)
{
    auto& t = groups_.at(parent_group);

    FT_CHECK(color >= 0);
    FT_CHECK(t.mapping[rank_] >= 0);

    auto vec = comm::Allgather(*this, std::make_tuple(color, key, t.mapping[rank_]), parent_group);

    auto last = std::stable_partition(vec.begin(), vec.end(), [&](auto x) {  //
        return std::get<0>(x) == color;
    });
    vec.erase(last, vec.end());
    std::stable_sort(vec.begin(), vec.end(), [](auto& a, auto& b) {  //
        return a < b;
    });

    const int index = groups_.size();

    auto& g = groups_.emplace_back();  // ! reference invalidation
    auto& p = groups_.at(parent_group);

    g.mapping.resize(n_ranks_, -1);

    for (size_t i = 0; i < vec.size(); ++i) {
        int rank = p.ranks.at(std::get<2>(vec[i]));
        g.ranks.push_back(rank);
        g.mapping[rank] = i;
    }

    return index;
}

void HostComm::Sync(int group)
{
    const auto& g = groups_[group];
    if (g.ranks.size() == 1 || g.mapping[rank_] < 0) {
        return;
    }
    for (const auto& r : g.ranks) {
        if (r != rank_) {
            auto& c = channel(rank_, r);
            void* expected{};
            while (!c.compare_exchange_weak(expected, (void*)1, std::memory_order_release)) {
                expected = {};
            }
        }
    }
    for (const auto& r : g.ranks) {
        if (r != rank_) {
            auto& c        = channel(r, rank_);
            void* expected = (void*)1;
            while (!c.compare_exchange_weak(expected, nullptr, std::memory_order_acquire)) {
                expected = (void*)1;
            }
        }
    }
    // for (const auto& r : g.ranks) {
    //     if (r != rank_) {
    //         auto& c = channel(rank_, r);
    //         while (c.load(std::memory_order_relaxed)) {}
    //     }
    // }
}

void HostComm::Broadcast(void* data, int count, DataType, int root, copy_fn copy, int group)
{
    FT_CHECK(copy);
    const auto& g = groups_.at(group);
    if (g.ranks.size() == 1 || g.mapping[rank_] < 0) {
        return;
    }
    // transform root to abs rank
    root = g.ranks.at(root);
    if (rank_ == root) {
        for (const auto& r : g.ranks) {
            if (r != rank_) {
                auto& c = channel(rank_, r);
                void* expected{};
                while (!c.compare_exchange_weak(expected, data, std::memory_order_release)) {
                    expected = {};
                }
            }
        }
        for (const auto& r : g.ranks) {
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

void HostComm::Allgather(void* all_data, int count, DataType, copy_fn copy, int group)
{
    FT_CHECK(copy);
    const auto& g = groups_.at(group);
    if (g.ranks.size() == 1 || g.mapping[rank_] < 0) {
        return;
    }
    for (const auto& r : g.ranks) {
        if (r != rank_) {
            auto& c = channel(rank_, r);
            void* expected{};
            while (!c.compare_exchange_weak(expected, all_data, std::memory_order_release)) {
                expected = {};
            }
        }
    }
    for (const auto& r : g.ranks) {
        if (r != rank_) {
            auto& c = channel(r, rank_);
            void* incoming{};
            while (!(incoming = c.load(std::memory_order_acquire))) {}
            copy(incoming, count, all_data, g.mapping[r] * count);
            c.store(nullptr, std::memory_order_relaxed);
        }
    }
    for (const auto& r : g.ranks) {
        if (r != rank_) {
            auto& c = channel(rank_, r);
            while (c.load(std::memory_order_relaxed)) {}
        }
    }
}

namespace {

template<class T, RedOp op>
void reduce_fn(void* src, int n, void* accum)
{
    for (int i = 0; i < n; ++i) {
        auto& s = *((T*)src + i);
        auto& a = *((T*)accum + i);
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

HostComm::reduce_fn get_reduce_fn(DataType dtype, RedOp red_op)
{
    auto dispatch_op = [&](auto t) -> HostComm::reduce_fn {
        using T = decltype(t);
        switch (red_op) {
            case RedOp::kSum:
                return reduce_fn<T, RedOp::kSum>;
            case RedOp::kMax:
                return reduce_fn<T, RedOp::kMax>;
            case RedOp::kMin:
                return reduce_fn<T, RedOp::kMin>;
            default:
                return {};
        }
    };
    auto dispatch = [&]() -> HostComm::reduce_fn {
        switch (dtype) {
            case DataType::TYPE_INT32:
                return dispatch_op(int32_t{});
            case DataType::TYPE_INT64:
                return dispatch_op(int64_t{});
            case DataType::TYPE_UINT32:
                return dispatch_op(uint32_t{});
            case DataType::TYPE_UINT64:
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

}  // namespace

void HostComm::Allreduce(const void* src, void* dst, int count, DataType dtype, RedOp red_op, int group)
{
    FT_CHECK(src != dst);
    const auto  reduce_fn = get_reduce_fn(dtype, red_op);
    const auto  elem_size = get_elem_size(dtype);
    const auto& g         = groups_.at(group);
    if (g.mapping[rank_] < 0) {
        return;
    }
    std::copy_n((char*)src, elem_size * count, (char*)dst);
    if (g.ranks.size() == 1) {
        return;
    }
    for (const auto& r : g.ranks) {
        if (r != rank_) {
            auto& c = channel(rank_, r);
            void* expected{};
            while (!c.compare_exchange_weak(expected, (void*)src, std::memory_order_release)) {
                expected = {};
            }
        }
    }
    for (const auto& r : g.ranks) {
        if (r != rank_) {
            auto& c = channel(r, rank_);
            void* incoming{};
            while (!(incoming = c.load(std::memory_order_acquire))) {}
            reduce_fn(incoming, count, dst);
            c.store(nullptr, std::memory_order_relaxed);
        }
    }
    for (const auto& r : g.ranks) {
        if (r != rank_) {
            auto& c = channel(rank_, r);
            while (c.load(std::memory_order_relaxed)) {}
        }
    }
}

class ThreadGroupId: public HostGroupId {
public:
    void Initialize() override
    {
        internal_ = std::make_shared<Internal>();
    }

    void Export(std::ostream& os) override
    {
        FT_CHECK((bool)internal_);  // `Initialize` must come befor `Export`

        const void* ptr = this;
        os.write((const char*)&ptr, sizeof(ptr));
    }

    void Import(std::istream& is) override
    {
        void* ptr{};
        is.read((char*)&ptr, sizeof(ptr));
        internal_ = reinterpret_cast<ThreadGroupId*>(ptr)->internal_;

        FT_CHECK((bool)internal_);
    }

    std::unique_ptr<HostComm> CreateHostCommunicator(int rank, int n_ranks) override
    {
        auto init_shared_state = [&] {  //
            internal_->state = std::make_shared<HostComm::State>(n_ranks);
        };

        FT_CHECK((bool)internal_);

        // One of the rank initialize the shared state
        std::call_once(internal_->flag, init_shared_state);

        FT_CHECK((bool)internal_->state);

        return std::make_unique<HostComm>(n_ranks, rank, internal_->state);
    }

private:
    struct Internal {
        std::once_flag                   flag;
        std::shared_ptr<HostComm::State> state;
    };

private:
    std::shared_ptr<Internal> internal_;
};

std::unique_ptr<HostGroupId> CreateHostGroupId(const std::string& backend)
{
    return std::make_unique<ThreadGroupId>();
}

}  // namespace turbomind::comm
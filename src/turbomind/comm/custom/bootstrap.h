
#include <mutex>
#include <queue>

#include "mscclpp/core.hpp"

#include "src/turbomind/comm/barrier.h"
#include "src/turbomind/comm/comm.h"

namespace turbomind::comm {

class LocalBootstrap: public mscclpp::Bootstrap {
public:
    struct State {

        explicit State(int n): num(n), barrier(n), ptrs(n), queues(n * n)
        {
            for (int i = 0; i < n; ++i) {
                mutexes.emplace_back();
            }
        }

        using Queue = std::queue<std::vector<uint8_t>>;

        Queue& get_que(int from, int to)
        {
            return queues[from * num + to];
        }

        int num;

        comm::Barrier barrier;

        std::vector<void*>     ptrs;
        std::deque<std::mutex> mutexes;
        std::vector<Queue>     queues;
    };

    LocalBootstrap(int world_size, int rank, std::shared_ptr<State> state):
        world_size_{world_size}, rank_{rank}, state_{state}
    {
    }

    int getRank() override
    {
        return rank_;
    }

    int getNranks() override
    {
        return world_size_;
    }

    int getNranksPerNode() override
    {
        return world_size_;
    }

    void send(void* data, int size, int peer, int tag) override
    {
        // std::cerr << "send " << size << " " << rank_ << " -> " << peer << " " << tag << "\n";
        std::lock_guard lock{state_->mutexes[peer]};
        auto&           que = state_->get_que(rank_, peer);
        que.push(std::vector<uint8_t>((uint8_t*)data, (uint8_t*)data + size));
    }

    void recv(void* data, int size, int peer, int tag) override
    {
        // std::cerr << "recv " << size << " " << rank_ << " <- " << peer << " " << tag << "\n";
        auto& que = state_->get_que(peer, rank_);
        while (true) {
            {
                std::lock_guard lock{state_->mutexes[rank_]};
                if (!que.empty()) {
                    FT_CHECK(que.front().size() == (size_t)size);
                    std::copy_n(que.front().begin(), size, (uint8_t*)data);
                    que.pop();
                    return;
                }
            }
            std::this_thread::yield();
        }
    }

    void allGather(void* allData, int size) override
    {
        barrier();

        state_->ptrs[rank_] = allData;

        barrier();

        for (int i = 0; i < world_size_; ++i) {
            if (i == rank_) {
                continue;
            }
            const auto offset = i * (size_t)size;
            std::copy_n((uint8_t*)state_->ptrs[i] + offset, size, (uint8_t*)allData + offset);
        }

        barrier();
    }

    void barrier() override
    {
        state_->barrier.arrive_and_wait();
    }

private:
    int world_size_;
    int rank_;

    std::shared_ptr<State> state_;
};

}  // namespace turbomind::comm
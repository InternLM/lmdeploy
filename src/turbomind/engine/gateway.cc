// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>

#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/request_queue.h"

namespace turbomind {

Gateway::Gateway(int n_ranks, std::function<std::shared_ptr<void>()> ctx_factory):
    n_ranks_{n_ranks}, queues_{}, ctx_factory_{ctx_factory}
{
    queues_.reserve(n_ranks);
    for (int i = 0; i < n_ranks; ++i) {
        queues_.push_back(std::make_unique<RequestQueueV2>(this));
    }
    signal_thread_ = std::thread(&Gateway::signal_thread_entry, this);
}

void Gateway::shutdown()
{
    for (auto& q : queues_) {
        q->close();
    }

    signal_buffer_.close();
    signal_thread_.join();
}

void Gateway::signal_thread_entry() noexcept
{
    while (true) {
        bool                abort{};
        std::vector<Signal> signals = signal_buffer_.take_all(abort);
        if (abort) {
            break;
        }
        else {
            auto ctx = ctx_factory_();
            for (const auto& s : signals) {
                s();
            }
        }
    }
}

}  // namespace turbomind

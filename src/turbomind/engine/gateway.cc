// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>

#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/request_queue.h"

namespace turbomind {

Gateway::Gateway(std::function<std::shared_ptr<void>()> ctx_factory): request_queue_{this}, ctx_factory_{ctx_factory}
{
    signal_thread_ = std::thread(&Gateway::signal_thread_entry, this);
}

void Gateway::shutdown()
{
    request_queue_.close();
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

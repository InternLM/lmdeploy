// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

#include "src/turbomind/engine/request.h"

namespace turbomind {

class Gateway;

class RequestQueue {
public:
    RequestQueue(Gateway* gateway): gateway_{gateway} {}

    void push(std::vector<std::shared_ptr<Request>> reqs);

    void pop(std::vector<std::shared_ptr<Request>>& infer_reqs,
             std::vector<std::shared_ptr<Request>>& kill_reqs,
             unsigned                               max_infer_num,
             bool                                   blocking,
             bool&                                  abort);

    void cancel(std::shared_ptr<Request> r);

    void kill(std::shared_ptr<Request> r);

    void close();

private:
    Gateway* gateway_;

    std::queue<std::shared_ptr<Request>> queue_;

    std::vector<std::shared_ptr<Request>> kill_;

    std::mutex              mutex_;
    std::condition_variable cv_;

    bool closed_{false};
};

}  // namespace turbomind

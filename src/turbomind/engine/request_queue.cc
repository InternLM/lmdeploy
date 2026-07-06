// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/engine/request_queue.h"
#include "src/turbomind/engine/gateway.h"

#include "src/turbomind/engine/request.h"

namespace turbomind {

std::unique_ptr<RequestQueue> RequestQueue::create(SchedulePolicy schedule_policy)
{
    switch (schedule_policy) {
        case SchedulePolicy::kFifo:
            return std::make_unique<FifoRequestQueue>();
        case SchedulePolicy::kPriority:
            return std::make_unique<PriorityRequestQueue>();
        default:
            throw std::invalid_argument("unsupported schedule_policy "
                                        + std::to_string(static_cast<int>(schedule_policy)));
    }
}

}  // namespace turbomind

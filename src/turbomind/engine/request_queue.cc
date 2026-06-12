// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/engine/request_queue.h"
#include "src/turbomind/engine/gateway.h"

#include "src/turbomind/engine/request.h"

namespace turbomind {

std::unique_ptr<RequestQueue> RequestQueue::create(SchedulePolicy schedule_policy)
{
    if (schedule_policy == SchedulePolicy::kFifo) {
        return std::make_unique<FifoRequestQueue>();
    }
    return std::make_unique<PriorityRequestQueue>();
}

}  // namespace turbomind

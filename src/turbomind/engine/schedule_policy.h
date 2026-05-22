// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace turbomind {

enum class SchedulePolicy
{
    kFifo,
    kPriority,
};

inline SchedulePolicy parse_schedule_policy(std::string_view value)
{
    if (value == "fifo") {
        return SchedulePolicy::kFifo;
    }
    if (value == "priority") {
        return SchedulePolicy::kPriority;
    }
    throw std::invalid_argument{"invalid schedule_policy: " + std::string{value}};
}

}  // namespace turbomind

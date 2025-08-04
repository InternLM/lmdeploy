#pragma once

#include <ostream>
#include <cstdint>
#include <chrono>

namespace turbomind {

struct ScheduleMetrics {
    // sequences
    int total_seqs;   // the number of received sequence
    int active_seqs;  // the number of active sequence
    int waiting_seqs; // the number of waiting sequence

    // kv block usage
    int total_blocks;  // the number of kv blocks
    int cached_blocks; // the number of cached kv blocks
    int free_blocks;   // the number of free kv blocks
};

struct RequestMetrics {
    int64_t enque_time;     // when a request is enqued
    int64_t scheduled_time; // when a request is scheduled for inference

    static int64_t timestamp()
    {
        // Get current timestamp in milliseconds and microseconds since Unix epoch
        // system_clock uses wall-clock time (matches Python's time.time())
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }
};

inline std::ostream& operator<<(std::ostream& os, const ScheduleMetrics& m)
{
    os << "ScheduleMetrics { ";
    os << "total_seqs=" << m.total_seqs;
    os << ", active_seqs=" << m.active_seqs;
    os << ", waiting_seqs=" << m.waiting_seqs;
    os << ", total_blocks=" << m.total_blocks;
    os << ", cached_blocks=" << m.cached_blocks;
    os << ", free_blocks=" << m.free_blocks;
    os << " }";
    return os;
}

}  // namespace turbomind

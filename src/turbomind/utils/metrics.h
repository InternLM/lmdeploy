#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ostream>

namespace turbomind {

struct ScheduleMetrics {
    // sequences
    int total_seqs;    // the number of received sequence
    int active_seqs;   // the number of active sequence
    int waiting_seqs;  // the number of waiting sequence

    // kv block usage
    int total_blocks;   // the number of kv blocks
    int active_blocks;  // the number of active kv blocks
    int cached_blocks;  // the number of cached kv blocks
    int free_blocks;    // the number of free kv blocks
};

// Counters for Gated Delta Net / linear-attention prefix caching (process-wide atomics in SequenceManager).
struct LinearPrefixCacheStats {
    uint64_t publish_ok{};
    uint64_t publish_miss{};
    uint64_t publish_pool_exhausted{};
    uint64_t prefix_match_skipped_alpha{};
    uint64_t linear_restore{};
};

struct RequestMetrics {
    std::atomic<int64_t> enqueue_time{};    // when a request is enqued
    std::atomic<int64_t> scheduled_time{};  // when a request is scheduled for inference

    static int64_t timestamp()
    {
        // Get current timestamp in microseconds since Unix epoch
        // system_clock uses wall-clock time (matches Python's time.time())
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
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

inline std::ostream& operator<<(std::ostream& os, const LinearPrefixCacheStats& s)
{
    os << "LinearPrefixCacheStats { ";
    os << "publish_ok=" << s.publish_ok;
    os << ", publish_miss=" << s.publish_miss;
    os << ", publish_pool_exhausted=" << s.publish_pool_exhausted;
    os << ", prefix_match_skipped_alpha=" << s.prefix_match_skipped_alpha;
    os << ", linear_restore=" << s.linear_restore;
    os << " }";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const RequestMetrics& m)
{
    os << "RequestMetrics { ";
    os << "enqueue_time=" << m.enqueue_time.load(std::memory_order_relaxed);
    os << ", scheduled_time=" << m.scheduled_time.load(std::memory_order_relaxed);
    os << " }";
    return os;
}

}  // namespace turbomind

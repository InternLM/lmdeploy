#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ostream>

namespace turbomind {

struct ScheduleMetrics {
    // sequences
    int total_seqs{};    // the number of received sequences
    int active_seqs{};   // the number of active sequences
    int waiting_seqs{};  // the number of waiting sequences

    // Cache-object counts. The heterogeneous allocator has no fixed-size free-block pool.
    int64_t total_blocks{};   // the number of live cache objects
    int64_t active_blocks{};  // live cache objects used by active sequences
    int64_t cached_blocks{};  // live cache objects not used by active sequences
    int64_t free_blocks{};    // always zero for the heterogeneous allocator

    double cache_usage{};            // live cache-object bytes / cache region bytes
    double prefix_cache_hit_rate{};  // skipped prompt tokens / queried prompt tokens

    int64_t scheduler_tick{};  // monotonic scheduler progress counter
};

struct RequestMetrics {
    std::atomic<int64_t> enqueue_time{};    // when a request is enqued
    std::atomic<int64_t> scheduled_time{};  // when a request is scheduled for inference
    std::atomic<int64_t> cached_tokens{};   // prompt tokens skipped at first admission

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
    os << ", scheduler_tick=" << m.scheduler_tick;
    os << ", cache_usage=" << m.cache_usage;
    os << ", prefix_cache_hit_rate=" << m.prefix_cache_hit_rate;
    os << ", total_blocks=" << m.total_blocks;
    os << ", active_blocks=" << m.active_blocks;
    os << ", cached_blocks=" << m.cached_blocks;
    os << ", free_blocks=" << m.free_blocks;
    os << " }";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const RequestMetrics& m)
{
    os << "RequestMetrics { ";
    os << "enqueue_time=" << m.enqueue_time.load(std::memory_order_relaxed);
    os << ", scheduled_time=" << m.scheduled_time.load(std::memory_order_relaxed);
    os << ", cached_tokens=" << m.cached_tokens.load(std::memory_order_relaxed);
    os << " }";
    return os;
}

}  // namespace turbomind

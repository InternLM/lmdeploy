#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <ostream>
#include <vector>

namespace turbomind {

struct ScheduleMetrics {
    // sequences
    int total_seqs{};    // the number of received sequences
    int active_seqs{};   // the number of active sequences
    int waiting_seqs{};  // the number of waiting sequences

    double cache_usage{};            // live cache-object bytes / cache region bytes
    double prefix_cache_hit_rate{};  // skipped prompt tokens / queried prompt tokens

    int64_t scheduler_tick{};  // monotonic scheduler progress counter
};

struct RequestMetrics {
    explicit RequestMetrics(int speculative_tokens = 0):
        num_accepted_tokens_per_pos(static_cast<size_t>(speculative_tokens))
    {
    }

    std::atomic<int64_t> enqueue_time{};    // when a request is enqued
    std::atomic<int64_t> scheduled_time{};  // when a request is scheduled for inference
    std::atomic<int64_t> cached_tokens{};   // prompt tokens skipped at first admission

    std::mutex spec_mutex;

    int64_t num_drafts{};
    int64_t num_draft_tokens{};
    int64_t num_accepted_tokens{};

    std::vector<int64_t> num_accepted_tokens_per_pos;

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

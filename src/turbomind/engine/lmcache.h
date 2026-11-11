#pragma once

#include <memory>
#include <string>

namespace turbomind {

namespace comm {
class HostComm;
}

class CacheRegistry;
class ObjectAllocator;
class Scheduler;
struct Sequence;

// Engine-owned LMCache coordinator. Empty `addr` yields a no-op.
class LmCache {
public:
    static LmCache Create(const std::string& addr,
                          const std::string& model_name,
                          comm::HostComm&    tp_group,
                          const int&         is_warm_up,
                          int                logical_block_size,
                          int                phases);

    LmCache();
    ~LmCache();

    LmCache(const LmCache&) = delete;
    LmCache& operator=(const LmCache&) = delete;
    LmCache(LmCache&&) noexcept;
    LmCache& operator=(LmCache&&) noexcept;

    void Register(void* cache_region_base, const ObjectAllocator& allocator, const CacheRegistry& registry);
    void Bind(Scheduler& scheduler);

    void OnAccepted(Sequence& sequence);
    void PrepareSchedule();
    void OnScheduled();
    bool Schedulable(const Sequence& sequence) const;
    bool Fallback(Sequence& sequence);
    void StageStores(int phase, Sequence* const* sequences, int count);
    void OnBatchComplete(int phase);
    void Poll();
    // Local drain for Join(), after the engine loop has stopped. No collectives.
    void Drain();
    bool HasPendingReleases() const;
    int  chunk_size() const;
    bool Ready(const Sequence& sequence) const;
    // Cache work must finish before the engine releases a retiring sequence.
    bool CanRetire(const Sequence& sequence) const;
    void OnRetire(Sequence& sequence);

private:
    class Impl;
    explicit LmCache(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind

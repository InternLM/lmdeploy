#pragma once

#include "src/turbomind/lmcache/mq.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace turbomind::lmcache {

class LookupContext;
class StoreContext;
class RetrieveContext;

struct ConnectorConfig {
    std::string server_addr;
    std::string model_name;
    std::string session_id;  // shared by the engine's TP ranks
    int         world_size{1};
    int         worker_id{0};  // this rank; LOOKUP/QUERY/FREE/END keys still use worker_id=None
    int         request_timeout_ms{300'000};
};

struct CachePool {
    std::size_t part_bytes{};
    int         tokens_per_block{};
};

struct RegistrationConfig {
    void*                  base{};
    std::size_t            size{};
    std::size_t            storage_offset_bytes{};
    std::vector<CachePool> pools;
};

struct Request {
    std::string                            request_id;
    std::vector<std::int64_t>              token_ids;
    std::vector<std::vector<std::int64_t>> block_ids;
    std::int64_t                           start{};
    std::int64_t                           end{};
    std::int64_t                           skip_first_n_tokens{};
    std::string                            cache_salt;
};

// Per-rank MP DEALER. LOOKUP uses worker_id=None; STORE uses worker_id.
class Connector final {
public:
    explicit Connector(ConnectorConfig config);
    ~Connector();

    Connector(const Connector&) = delete;
    Connector& operator=(const Connector&) = delete;
    Connector(Connector&&)                 = delete;
    Connector& operator=(Connector&&) = delete;

    int          chunk_size() const noexcept;
    std::int64_t instance_id() const noexcept;
    bool         healthy() const noexcept;

    void         Register(const RegistrationConfig& registration);
    std::int64_t BlockId(int pool_index, const void* part_address) const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    const ConnectorConfig& config() const noexcept;
    std::string            MakeScopedRequestId(const std::string& request_id) const;
    RequestHandle          Submit(protocol::RequestType            request_type,
                                  ResponseFrames                   frames,
                                  MessageQueueClient::ResponseMode response_mode = MessageQueueClient::ResponseMode::kTrack);

    // Best-effort cleanup only. Include serialization in the exception boundary;
    // a failed notification leaves cleanup to the daemon's existing TTL.
    template<class... Args>
    void Notify(protocol::RequestType type, const Args&... args) noexcept
    {
        try {
            Submit(type, {protocol::Pack(args)...}, MessageQueueClient::ResponseMode::kIgnore);
        }
        catch (...) {
        }
    }

    friend class LookupContext;
    friend class RequestLease;
    friend class StoreContext;
    friend class RetrieveContext;
};

}  // namespace turbomind::lmcache

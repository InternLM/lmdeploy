#include "src/turbomind/lmcache/lookup.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/lmcache/protocol.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace turbomind::lmcache {
namespace {
using Clock = std::chrono::steady_clock;
using protocol::Pack;
using protocol::RequestType;
using protocol::Unpack;

constexpr auto kQueryBackoff = std::chrono::milliseconds(5);

protocol::IPCCacheServerKey MakeCacheKey(const ConnectorConfig& config, Request& request)
{
    protocol::IPCCacheServerKey key;
    key.model_name = config.model_name;
    key.world_size = config.world_size;
    key.worker_id  = std::nullopt;  // scheduler RPCs; STORE/RETRIEVE will use config.worker_id
    key.token_ids  = std::move(request.token_ids);
    key.start      = request.start;
    key.end        = request.end;
    key.request_id = request.request_id;
    key.cache_salt = request.cache_salt;
    return key;
}

}  // namespace

struct RequestLease::State {
    State(Connector& connector, protocol::IPCCacheServerKey key): connector{&connector}, key{std::move(key)} {}

    State(const State&) = delete;
    State& operator=(const State&) = delete;

    ~State()
    {
        ReleaseLocks();
        connector->Notify(RequestType::kEndSession, key.request_id);
    }

    void ReleaseLocks() noexcept
    {
        if (key.start < delegated_start) {
            const auto end = std::exchange(key.end, delegated_start);
            connector->Notify(RequestType::kFreeLookupLocks, key, std::int64_t{connector->config().world_size});
            key.end   = end;
            key.start = delegated_start;
        }
        key.start = std::max(key.start, delegated_end);
        if (key.start == key.end) {
            return;
        }
        connector->Notify(RequestType::kFreeLookupLocks, key, std::int64_t{connector->config().world_size});
        key.start = key.end;
    }

    Connector*                  connector;
    protocol::IPCCacheServerKey key;
    std::int64_t                delegated_start{};
    std::int64_t                delegated_end{};
};

RequestLease::RequestLease() = default;
RequestLease::RequestLease(Connector& connector, const std::string& request_id)
{
    protocol::IPCCacheServerKey key;
    key.request_id = connector.MakeScopedRequestId(request_id);
    state_         = std::make_unique<State>(connector, std::move(key));
}
RequestLease::RequestLease(Connector& connector, protocol::IPCCacheServerKey key):
    state_{std::make_unique<State>(connector, std::move(key))}
{
}

RequestLease::RequestLease(RequestLease&&) noexcept = default;
RequestLease& RequestLease::operator=(RequestLease&&) noexcept = default;
RequestLease::~RequestLease()                                  = default;

void RequestLease::ReleaseLocks() noexcept
{
    if (state_) {
        state_->ReleaseLocks();
    }
}

void RequestLease::DelegateLocks(std::int64_t start, std::int64_t end) noexcept
{
    state_->delegated_start = start;
    state_->delegated_end   = end;
}

struct LookupContext::State {
    enum class Stage
    {
        kWaitingLookup,
        kWaitingQuery,
    };

    Connector*                            connector{};
    protocol::IPCCacheServerKey           key;
    std::optional<RequestHandle>          rpc;
    std::chrono::steady_clock::time_point deadline;
    std::chrono::steady_clock::time_point next_query{};
    Stage                                 stage{Stage::kWaitingLookup};
    std::optional<LookupResult>           result;
    std::optional<RequestLease>           lease;
};

LookupContext::LookupContext(Connector& connector, Request request)
{
    request.request_id = connector.MakeScopedRequestId(request.request_id);
    state_             = std::make_unique<State>();
    state_->connector  = &connector;
    state_->key        = MakeCacheKey(connector.config(), request);
    state_->deadline   = Clock::now() + std::chrono::milliseconds(connector.config().request_timeout_ms);

    if (!connector.healthy()) {
        Finish(false, 0, "LMCache server is unhealthy; Lookup skipped");
        return;
    }

    state_->rpc.emplace(connector.Submit(
        RequestType::kLookup, {Pack(state_->key), Pack(static_cast<std::int64_t>(connector.config().world_size))}));
}

LookupContext::LookupContext(LookupContext&&) noexcept = default;
LookupContext& LookupContext::operator=(LookupContext&&) noexcept = default;
LookupContext::~LookupContext()                                   = default;

void LookupContext::Finish(bool success, std::int64_t matched_tokens, std::string error)
{
    state_->rpc.reset();
    const auto start = state_->key.start;
    if (success) {
        state_->key.end = start + matched_tokens;
        state_->key.token_ids.resize(state_->key.end);
        state_->lease.emplace(RequestLease{*state_->connector, std::move(state_->key)});
    }
    // Unknown outcomes acquire no lease and rely on the daemon's existing TTL.
    state_->result = LookupResult{success, start, matched_tokens, std::move(error)};
}

void LookupContext::SubmitQuery()
{
    const auto& request_id = state_->key.request_id;
    state_->rpc.emplace(state_->connector->Submit(RequestType::kQueryPrefetchStatus, {Pack(request_id)}));
    state_->stage = State::Stage::kWaitingQuery;
}

std::optional<LookupResult> LookupContext::Poll()
{
    auto& state = *state_;
    if (state.result) {
        return state.result;
    }
    if (!state.connector->healthy()) {
        Finish(false, 0, "LMCache server is unhealthy");
        return state.result;
    }
    if (Clock::now() >= state.deadline) {
        Finish(false, 0, "LMCache Lookup timed out");
        return state.result;
    }

    try {
        if (!state.rpc) {
            if (Clock::now() >= state.next_query) {
                SubmitQuery();
            }
            return std::nullopt;
        }
        if (!state.rpc->Ready()) {
            return std::nullopt;
        }
        auto response = state.rpc->Get();
        state.rpc.reset();
        if (state.stage == State::Stage::kWaitingLookup) {
            SubmitQuery();
            return std::nullopt;
        }

        const auto chunks = Unpack<std::optional<std::int64_t>>(response.at(0));
        if (!chunks) {
            state.next_query = Clock::now() + kQueryBackoff;
            return std::nullopt;
        }
        const int chunk_size = state.connector->chunk_size();
        if (*chunks < 0 || *chunks > (state.key.end - state.key.start) / chunk_size) {
            throw std::runtime_error("LMCache returned an invalid Lookup chunk count");
        }
        Finish(true, *chunks * chunk_size, {});
    }
    catch (const std::exception& e) {
        Finish(false, 0, e.what());
    }
    return state.result;
}

RequestLease LookupContext::TakeLease()
{
    TM_CHECK(state_ && state_->lease);
    auto lease = std::move(*state_->lease);
    state_->lease.reset();
    return lease;
}

}  // namespace turbomind::lmcache

#include "src/turbomind/lmcache/retrieve.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/lmcache/event.h"
#include "src/turbomind/lmcache/protocol.h"

#include <chrono>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace turbomind::lmcache {
namespace {
using Clock = std::chrono::steady_clock;
using protocol::Pack;
using protocol::RequestType;

protocol::IPCCacheServerKey MakeRetrieveKey(const ConnectorConfig& config, Request& request)
{
    protocol::IPCCacheServerKey key;
    key.model_name = config.model_name;
    key.world_size = config.world_size;
    key.worker_id  = config.worker_id;
    key.token_ids  = std::move(request.token_ids);
    key.start      = request.start;
    key.end        = request.end;
    key.request_id = request.request_id;
    key.cache_salt = request.cache_salt;
    return key;
}

cudaIpcEventHandle_t DecodeCudaEventHandle(const Bytes& bytes)
{
    if (bytes.size() != sizeof(cudaIpcEventHandle_t)) {
        throw std::runtime_error("LMCache transfer response contains an invalid CUDA event handle");
    }
    cudaIpcEventHandle_t handle{};
    std::memcpy(&handle, bytes.data(), sizeof(handle));
    return handle;
}

}  // namespace

struct RetrieveContext::State {
    enum class Stage
    {
        kPrepared,
        kWaitingReady,
        kSubmitted
    };

    State(Connector& connector, Request request, cudaStream_t stream):
        connector{&connector},
        key{MakeRetrieveKey(connector.config(), request)},
        block_ids{std::move(request.block_ids)},
        skip_first_n_tokens{request.skip_first_n_tokens},
        ready_event{CudaEvent::Create()}
    {
        key.request_id = connector.MakeScopedRequestId(key.request_id);
        ready_event.Record(stream);
    }

    ~State()
    {
        TM_CHECK(stage == Stage::kPrepared || result) << "activated RETRIEVE must be drained before destruction";
    }

    void Finish(bool success, std::string error)
    {
        if (!success && !error.empty()) {
            TM_LOG_WARN("LMCache RETRIEVE {}, worker={}: {}", key.request_id, connector->config().worker_id, error);
        }
        rpc.reset();
        done_event.reset();
        result = RetrieveResult{success, std::move(error)};
        // Cleanup is last and cannot throw. After submission the daemon owns
        // these locks; unconsumed groups from a failed transfer expire by TTL.
        if (stage == Stage::kWaitingReady) {
            connector->Notify(RequestType::kFreeLookupLocks, key, std::int64_t{connector->config().world_size});
        }
    }

    bool TrySubmit()
    {
        if (!connector->healthy() || Clock::now() >= deadline) {
            Finish(false, "unavailable before submission; recomputing locally");
            return false;
        }
        const auto status = ready_event.Query();
        if (status == cudaErrorNotReady) {
            return false;
        }
        if (status != cudaSuccess) {
            Finish(false, cudaGetErrorString(status));
            return false;
        }
        const auto ready = ready_event.handle();
        rpc.emplace(connector->Submit(RequestType::kRetrieve,
                                      {Pack(key),
                                       Pack(connector->instance_id()),
                                       Pack(block_ids),
                                       protocol::PackBinary(&ready, sizeof(ready)),
                                       Pack(skip_first_n_tokens)}));
        stage = Stage::kSubmitted;
        return true;
    }

    Connector*                             connector;
    protocol::IPCCacheServerKey            key;
    std::vector<std::vector<std::int64_t>> block_ids;
    std::int64_t                           skip_first_n_tokens;
    std::optional<RequestHandle>           rpc;
    CudaEvent                              ready_event;
    std::optional<CudaEvent>               done_event;
    std::optional<cudaIpcEventHandle_t>    done_handle;
    Clock::time_point                      deadline;
    std::optional<RetrieveResult>          result;
    Stage                                  stage{Stage::kPrepared};
    bool                                   success{};
    bool                                   uncertain{};
};

RetrieveContext::RetrieveContext(Connector& connector, Request request, cudaStream_t stream):
    state_{std::make_unique<State>(connector, std::move(request), stream)}
{
}
RetrieveContext::RetrieveContext(RetrieveContext&&) noexcept = default;
RetrieveContext& RetrieveContext::operator=(RetrieveContext&&) noexcept = default;
RetrieveContext::~RetrieveContext()                                     = default;

void RetrieveContext::Activate() noexcept
{
    state_->stage    = State::Stage::kWaitingReady;
    state_->deadline = Clock::now() + std::chrono::milliseconds(state_->connector->config().request_timeout_ms);
}

std::optional<RetrieveResult> RetrieveContext::Poll()
{
    auto& state = *state_;
    if (state.result || state.stage == State::Stage::kPrepared) {
        return state.result;
    }
    try {
        if (state.stage == State::Stage::kWaitingReady && !state.TrySubmit()) {
            return state.result;
        }
        if (state.rpc && state.rpc->Ready()) {
            auto rpc = std::move(*state.rpc);
            state.rpc.reset();
            auto       response = rpc.Get();
            const auto transfer = protocol::UnpackTransferResponse(response.at(0));
            state.success       = transfer.success;
            // An empty event is the daemon's guarantee that no device work ran.
            if (transfer.event_handle.empty()) {
                state.Finish(state.success, state.success ? std::string{} : "daemon rejected RETRIEVE");
                return state.result;
            }
            state.done_handle = DecodeCudaEventHandle(transfer.event_handle);
        }
        if (state.done_handle && !state.done_event) {
            state.done_event.emplace(CudaEvent::Open(*state.done_handle));
        }
        if (state.done_event) {
            const auto status = state.done_event->Query();
            if (status == cudaSuccess) {
                state.Finish(state.success, state.success ? std::string{} : "daemon rejected RETRIEVE");
                return state.result;
            }
            if (status != cudaErrorNotReady) {
                MarkUncertain(cudaGetErrorString(status));
            }
        }
    }
    catch (const std::exception& e) {
        if (state.stage == State::Stage::kWaitingReady) {
            state.Finish(false, e.what());
            return state.result;
        }
        MarkUncertain(e.what());
    }
    if (!state.connector->healthy()) {
        MarkUncertain("server is unhealthy");
    }
    else if (Clock::now() >= state.deadline) {
        MarkUncertain("timed out");
    }
    return state.result;
}

bool RetrieveContext::uncertain() const noexcept
{
    return state_->uncertain && !state_->result;
}

void RetrieveContext::Cancel()
{
    if (state_->stage != State::Stage::kSubmitted && !state_->result) {
        state_->Finish(false, {});
    }
}

void RetrieveContext::Drain()
{
    Cancel();
    while (!Poll()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void RetrieveContext::MarkUncertain(const std::string& error)
{
    if (!std::exchange(state_->uncertain, true)) {
        TM_LOG_WARN("LMCache RETRIEVE {}, worker={}: {}; retaining destinations until CUDA completion",
                    state_->key.request_id,
                    state_->connector->config().worker_id,
                    error);
    }
}

}  // namespace turbomind::lmcache

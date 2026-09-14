#include "src/turbomind/lmcache/store.h"

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

protocol::IPCCacheServerKey MakeStoreKey(const ConnectorConfig& config, Request& request)
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

struct StoreContext::State {
    State(Connector& connector, Request request, std::shared_ptr<CudaEvent> ready);
    ~State()
    {
        TM_CHECK(result) << "submitted STORE must be drained before destruction";
    }

    void Finish(bool success, std::string error);

    Connector*                            connector{};
    std::optional<RequestHandle>          rpc;
    std::shared_ptr<CudaEvent>            ready_event;
    std::optional<CudaEvent>              done_event;
    std::optional<cudaIpcEventHandle_t>   done_handle;
    std::chrono::steady_clock::time_point deadline;
    std::optional<StoreResult>            result;
    std::string                           request_id;
    bool                                  success{};
    bool                                  uncertain{};
};

std::shared_ptr<CudaEvent> StoreContext::RecordReady(cudaStream_t stream)
{
    auto event = std::make_shared<CudaEvent>(CudaEvent::Create());
    event->Record(stream);
    return event;
}

StoreContext::StoreContext(Connector& connector, Request request, cudaStream_t stream):
    StoreContext{connector, std::move(request), RecordReady(stream)}
{
}

StoreContext::StoreContext(Connector& connector, Request request, std::shared_ptr<CudaEvent> ready):
    state_{std::make_unique<State>(connector, std::move(request), std::move(ready))}
{
}

StoreContext::State::State(Connector& connector, Request request, std::shared_ptr<CudaEvent> ready):
    connector{&connector},
    ready_event{std::move(ready)},
    deadline{Clock::now() + std::chrono::milliseconds(connector.config().request_timeout_ms)},
    request_id{connector.MakeScopedRequestId(request.request_id)}
{
    request.request_id = request_id;

    if (!connector.healthy()) {
        Finish(false, "LMCache server is unhealthy; Store skipped");
        return;
    }

    const auto handle = ready_event->handle();
    // Submission is the final construction step. Earlier failures unwind
    // members without invoking State's completed-transfer check.
    rpc.emplace(connector.Submit(RequestType::kStore,
                                 {Pack(MakeStoreKey(connector.config(), request)),
                                  Pack(connector.instance_id()),
                                  Pack(request.block_ids),
                                  protocol::PackBinary(&handle, sizeof(handle))}));
}

StoreContext::StoreContext(StoreContext&&) noexcept = default;
StoreContext& StoreContext::operator=(StoreContext&&) noexcept = default;
StoreContext::~StoreContext()                                  = default;

void StoreContext::State::Finish(bool success, std::string error)
{
    if (!success) {
        TM_LOG_WARN("LMCache STORE {}, worker={}: {}", request_id, connector->config().worker_id, error);
    }
    rpc.reset();
    done_event.reset();
    result = StoreResult{success, std::move(error)};
}

std::optional<StoreResult> StoreContext::Poll()
{
    auto& state = *state_;
    if (state.result) {
        return state.result;
    }
    try {
        if (state.rpc && state.rpc->Ready()) {
            // Retain a late reply after a timeout. Transport completion alone
            // says nothing about outstanding CUDA reads, including on failure.
            auto rpc = std::move(*state.rpc);
            state.rpc.reset();
            auto       response = rpc.Get();
            const auto transfer = protocol::UnpackTransferResponse(response.at(0));
            state.success       = transfer.success;
            state.done_handle   = DecodeCudaEventHandle(transfer.event_handle);
        }
        if (state.done_handle && !state.done_event) {
            state.done_event.emplace(CudaEvent::Open(*state.done_handle));
        }
        if (state.done_event) {
            const auto status = state.done_event->Query();
            if (status == cudaSuccess) {
                state.Finish(state.success, state.success ? std::string{} : "LMCache daemon rejected STORE");
                return state.result;
            }
            if (status != cudaErrorNotReady) {
                MarkUncertain(cudaGetErrorString(status));
            }
        }
    }
    catch (const std::exception& e) {
        MarkUncertain(e.what());
    }
    if (!state.connector->healthy()) {
        MarkUncertain("LMCache server is unhealthy");
    }
    else if (Clock::now() >= state.deadline) {
        MarkUncertain("LMCache STORE timed out");
    }
    return state.result;
}

bool StoreContext::uncertain() const noexcept
{
    return state_->uncertain && !state_->result;
}

void StoreContext::Drain()
{
    while (!Poll()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void StoreContext::MarkUncertain(const std::string& error)
{
    if (!std::exchange(state_->uncertain, true)) {
        TM_LOG_WARN("LMCache STORE {}, worker={}: {}; retaining source buffers until CUDA completion",
                    state_->request_id,
                    state_->connector->config().worker_id,
                    error);
    }
}

}  // namespace turbomind::lmcache

#pragma once

#include "src/turbomind/lmcache/connector.h"

#include <cuda_runtime_api.h>

#include <memory>
#include <optional>
#include <string>

namespace turbomind::lmcache {

class CudaEvent;

struct StoreResult {
    bool        success{};
    std::string error;
};

// A terminal result means the daemon can no longer read the source buffers.
// The owner must drain submitted work before destroying this context.
class StoreContext final {
public:
    StoreContext(Connector& connector, Request request, cudaStream_t stream);
    StoreContext(Connector& connector, Request request, std::shared_ptr<CudaEvent> ready);
    StoreContext(StoreContext&&) noexcept;
    StoreContext& operator=(StoreContext&&) noexcept;
    ~StoreContext();

    StoreContext(const StoreContext&) = delete;
    StoreContext& operator=(const StoreContext&) = delete;

    std::optional<StoreResult> Poll();
    bool                       uncertain() const noexcept;

    // Blocking shutdown drain; normal engine progress uses Poll().
    void Drain();

    // All STOREs produced by a batch can share this event.
    static std::shared_ptr<CudaEvent> RecordReady(cudaStream_t stream);

private:
    void MarkUncertain(const std::string& error);

    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace turbomind::lmcache

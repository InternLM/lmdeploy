#pragma once

#include "src/turbomind/lmcache/connector.h"

#include <cuda_runtime_api.h>

#include <memory>
#include <optional>
#include <string>

namespace turbomind::lmcache {

struct RetrieveResult {
    bool        success{};
    std::string error;
};

// Owns one RETRIEVE. Poll waits for the engine-stream ready event, then the RPC,
// then the daemon completion event. The MP Retrieve handler does not wait on the
// ready handle, so destinations must not be exposed until that event completes.
class RetrieveContext final {
public:
    RetrieveContext(Connector& connector, Request request, cudaStream_t stream);
    RetrieveContext(RetrieveContext&&) noexcept;
    RetrieveContext& operator=(RetrieveContext&&) noexcept;
    ~RetrieveContext();

    RetrieveContext(const RetrieveContext&) = delete;
    RetrieveContext& operator=(const RetrieveContext&) = delete;

    // Preparation owns no read locks and submits no RPC. Activate only after
    // all TP workers have prepared, and the LOOKUP lease delegated this range.
    void                          Activate() noexcept;
    std::optional<RetrieveResult> Poll();
    bool                          uncertain() const noexcept;
    // Before submission cancellation is immediate; submitted writes must drain.
    void Cancel();
    void Drain();

private:
    void MarkUncertain(const std::string& error);

    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace turbomind::lmcache

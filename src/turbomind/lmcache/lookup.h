#pragma once

#include "src/turbomind/lmcache/connector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace turbomind::lmcache {

struct LookupResult {
    bool         success{};
    std::int64_t start{};
    std::int64_t matched_tokens{};
    std::string  error;
};

// Owns the matched read locks and the remote request session independently of
// the LOOKUP RPC. ReleaseLocks leaves the session alive until destruction.
class RequestLease final {
public:
    RequestLease();
    // A STORE-only session owns no LOOKUP locks.
    RequestLease(Connector& connector, const std::string& request_id);
    ~RequestLease();
    RequestLease(RequestLease&&) noexcept;
    RequestLease& operator=(RequestLease&&) noexcept;

    RequestLease(const RequestLease&) = delete;
    RequestLease& operator=(const RequestLease&) = delete;

    void ReleaseLocks() noexcept;
    // Workers take ownership of this range after common RETRIEVE preparation.
    // ReleaseLocks() subsequently frees only the unused prefix and suffix.
    void DelegateLocks(std::int64_t start, std::int64_t end) noexcept;

private:
    RequestLease(Connector& connector, protocol::IPCCacheServerKey key);

    struct State;
    std::unique_ptr<State> state_;
    friend class LookupContext;
};

// Owns one LOOKUP RPC. TakeLease transfers a completed result's remote ownership.
class LookupContext final {
public:
    LookupContext(Connector& connector, Request request);
    LookupContext(LookupContext&&) noexcept;
    LookupContext& operator=(LookupContext&&) noexcept;
    ~LookupContext();

    LookupContext(const LookupContext&) = delete;
    LookupContext& operator=(const LookupContext&) = delete;

    std::optional<LookupResult> Poll();
    RequestLease                TakeLease();

private:
    void Finish(bool success, std::int64_t matched_tokens, std::string error);
    void SubmitQuery();

    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace turbomind::lmcache

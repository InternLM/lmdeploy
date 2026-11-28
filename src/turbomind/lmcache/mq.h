#pragma once

#include "src/turbomind/lmcache/protocol.h"

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace turbomind::lmcache {

using ResponseFrames = std::vector<Bytes>;
struct RequestState;

class RequestHandle final {
public:
    RequestHandle() = default;
    ~RequestHandle();

    RequestHandle(const RequestHandle&) = delete;
    RequestHandle& operator=(const RequestHandle&) = delete;
    RequestHandle(RequestHandle&&) noexcept        = default;
    RequestHandle& operator=(RequestHandle&&) noexcept = delete;

    bool           Ready() const noexcept;
    bool           WaitUntil(std::chrono::steady_clock::time_point deadline) const;
    ResponseFrames Get();

private:
    explicit RequestHandle(std::shared_ptr<RequestState> state);
    void Abandon() noexcept;

    std::shared_ptr<RequestState> state_;
    std::future<ResponseFrames>   future_;

    friend class MessageQueueClient;
};

// One I/O thread owns the DEALER socket and correlates out-of-order replies.
class MessageQueueClient final {
public:
    enum class ResponseMode
    {
        kTrack,
        kIgnore,
    };

    explicit MessageQueueClient(std::string server_addr);
    ~MessageQueueClient();

    RequestHandle Submit(protocol::RequestType request_type,
                         ResponseFrames        payloads,
                         ResponseMode          response_mode = ResponseMode::kTrack);

    void Stop() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind::lmcache

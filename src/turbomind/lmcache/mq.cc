#include "src/turbomind/lmcache/mq.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/logger.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <deque>
#include <future>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <sys/eventfd.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <zmq.hpp>
#include <zmq_addon.hpp>

namespace turbomind::lmcache {
using protocol::Pack;
using protocol::RequestType;
using protocol::Unpack;

constexpr int kSocketIoTimeoutMs = 1'000;
constexpr int kPollTimeoutMs     = 1'000;

ResponseFrames ReceiveFrames(zmq::socket_t& socket)
{
    std::vector<zmq::message_t> messages;
    if (!zmq::recv_multipart(socket, std::back_inserter(messages))) {
        throw std::runtime_error("failed to receive LMCache response");
    }
    ResponseFrames frames;
    frames.reserve(messages.size());
    for (const auto& message : messages) {
        const auto* begin = static_cast<const std::uint8_t*>(message.data());
        frames.emplace_back(begin, begin + message.size());
    }
    return frames;
}

struct RequestState {
    std::atomic<bool>            resolved{false};
    std::promise<ResponseFrames> promise;
};

void Resolve(const std::shared_ptr<RequestState>& state, ResponseFrames response = {}, std::string error = {}) noexcept
{
    if (!state) {
        return;
    }
    if (state->resolved.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    try {
        if (error.empty()) {
            state->promise.set_value(std::move(response));
        }
        else {
            state->promise.set_exception(std::make_exception_ptr(std::runtime_error(std::move(error))));
        }
    }
    catch (...) {
    }
}

RequestHandle::RequestHandle(std::shared_ptr<RequestState> state):
    state_{std::move(state)}, future_{state_->promise.get_future()}
{
}

RequestHandle::~RequestHandle()
{
    Abandon();
}

bool RequestHandle::Ready() const noexcept
{
    // libstdc++ std::future::wait_for(0) is a timed wait and can block the engine thread.
    return state_ && state_->resolved.load(std::memory_order_acquire);
}

bool RequestHandle::WaitUntil(std::chrono::steady_clock::time_point deadline) const
{
    return state_ && future_.wait_until(deadline) == std::future_status::ready;
}

ResponseFrames RequestHandle::Get()
{
    return future_.get();
}

void RequestHandle::Abandon() noexcept
{
    if (state_) {
        Resolve(state_, {}, "LMCache request handle was abandoned");
    }
}

class MessageQueueClient::Impl {
public:
    explicit Impl(std::string server_addr): server_addr_{std::move(server_addr)}, context_{1}
    {
        TM_CHECK(!server_addr_.empty()) << "LMCache server address must not be empty";
        notifier_fd_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        TM_CHECK_GE(notifier_fd_, 0) << "failed to create LMCache message-queue notifier: " << std::strerror(errno);
        thread_ = std::thread(&Impl::Loop, this);
    }

    ~Impl()
    {
        Stop();
        if (notifier_fd_ >= 0) {
            close(notifier_fd_);
        }
    }

    void Submit(RequestType request_type, ResponseFrames payloads, std::shared_ptr<RequestState> state)
    {
        bool stopped{};
        bool notify{};
        {
            std::lock_guard<std::mutex> lock(outbound_mutex_);
            stopped = stopping_.load(std::memory_order_relaxed);
            if (!stopped) {
                const auto uid = state ? next_uid_++ : kIgnoredResponseUid;
                notify         = outbound_.empty();
                outbound_.push_back({uid, request_type, std::move(payloads), std::move(state)});
            }
        }
        if (stopped) {
            Resolve(state, {}, "LMCache message queue is not running");
            return;
        }
        if (notify) {
            Wake();
        }
    }

    void Stop() noexcept
    {
        bool notify{};
        {
            std::lock_guard<std::mutex> lock(outbound_mutex_);
            notify = !stopping_.exchange(true, std::memory_order_relaxed);
        }
        if (notify) {
            Wake();
        }
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void Wake() noexcept
    {
        const std::uint64_t value = 1;
        while (write(notifier_fd_, &value, sizeof(value)) < 0 && errno == EINTR) {}
    }

private:
    struct Task {
        std::uint64_t                 uid{};
        RequestType                   type{};
        ResponseFrames                payloads;
        std::shared_ptr<RequestState> state;
    };

    struct Pending {
        RequestType                   type{};
        std::shared_ptr<RequestState> state;
    };

    zmq::socket_t MakeSocket()
    {
        zmq::socket_t socket{context_, zmq::socket_type::dealer};
        socket.set(zmq::sockopt::linger, 0);
        socket.set(zmq::sockopt::immediate, 1);
        socket.set(zmq::sockopt::sndtimeo, kSocketIoTimeoutMs);
        socket.connect(server_addr_);
        return socket;
    }

    void DrainNotifier() noexcept
    {
        std::uint64_t value{};
        while (read(notifier_fd_, &value, sizeof(value)) < 0 && errno == EINTR) {}
    }

    void SendTask(zmq::socket_t& socket, Task& task)
    {
        TM_CHECK_EQ(task.uid == kIgnoredResponseUid, task.state == nullptr)
            << "UID 0 is reserved for LMCache requests that ignore responses";
        if (task.state && task.state->resolved.load(std::memory_order_relaxed)) {
            return;
        }
        auto                           uid_frame  = Pack(task.uid);
        auto                           type_frame = Pack(task.type);
        std::vector<zmq::const_buffer> frames;
        frames.reserve(task.payloads.size() + 2);
        frames.emplace_back(zmq::buffer(uid_frame));
        frames.emplace_back(zmq::buffer(type_frame));
        for (const auto& payload : task.payloads) {
            frames.emplace_back(zmq::buffer(payload));
        }
        const auto sent = zmq::send_multipart(socket, frames);
        if (!sent || *sent != frames.size()) {
            throw std::runtime_error("failed to send LMCache request to " + server_addr_);
        }
        if (task.state) {
            pending_.emplace(task.uid, Pending{task.type, task.state});
        }
    }

    void DrainOutbound(zmq::socket_t& socket)
    {
        std::deque<Task> tasks;
        {
            std::lock_guard<std::mutex> lock(outbound_mutex_);
            tasks.swap(outbound_);
        }
        while (!tasks.empty()) {
            auto task = std::move(tasks.front());
            tasks.pop_front();
            try {
                SendTask(socket, task);
            }
            catch (const std::exception& e) {
                Resolve(task.state, {}, e.what());
                for (auto& queued : tasks) {
                    Resolve(queued.state, {}, "LMCache transport unavailable before send");
                }
                break;
            }
        }
    }

    void ProcessInbound(zmq::socket_t& socket)
    {
        try {
            auto frames = ReceiveFrames(socket);
            if (frames.size() < 2) {
                throw std::runtime_error("LMCache response has fewer than two frames");
            }
            const auto uid = Unpack<std::uint64_t>(frames[0]);
            if (uid == kIgnoredResponseUid) {
                return;
            }
            const auto type = Unpack<RequestType>(frames[1]);
            auto       it   = pending_.find(uid);
            if (it == pending_.end()) {
                return;
            }

            auto state = it->second.state;
            if (type != it->second.type) {
                pending_.erase(it);
                Resolve(state, {}, "LMCache response UID/type mismatch");
                return;
            }
            frames.erase(frames.begin(), frames.begin() + 2);
            pending_.erase(it);
            Resolve(state, std::move(frames));
        }
        catch (const std::exception& e) {
            TM_LOG_WARN("LMCache MQ response failed: server={}, error={}", server_addr_, e.what());
            RejectPending(e.what());
        }
    }

    void PruneAbandoned()
    {
        for (auto it = pending_.begin(); it != pending_.end();) {
            if (it->second.state->resolved.load(std::memory_order_relaxed)) {
                it = pending_.erase(it);
            }
            else {
                ++it;
            }
        }
    }

    void RejectPending(const std::string& error)
    {
        for (auto& [_, pending] : pending_) {
            Resolve(pending.state, {}, error);
        }
        pending_.clear();
    }

    void RejectAll(const std::string& error)
    {
        {
            std::lock_guard<std::mutex> lock(outbound_mutex_);
            for (auto& task : outbound_) {
                Resolve(task.state, {}, error);
            }
            outbound_.clear();
        }
        RejectPending(error);
    }

    void Loop() noexcept
    {
        try {
            auto socket = MakeSocket();
            while (!stopping_.load(std::memory_order_relaxed)) {
                DrainOutbound(socket);
                PruneAbandoned();

                zmq::pollitem_t items[] = {
                    {socket.handle(), 0, ZMQ_POLLIN, 0},
                    {nullptr, notifier_fd_, ZMQ_POLLIN, 0},
                };
                const auto ready = zmq::poll(items, 2, std::chrono::milliseconds(kPollTimeoutMs));
                if (ready == 0) {
                    continue;
                }
                if (items[1].revents & ZMQ_POLLIN) {
                    DrainNotifier();
                }
                if (items[0].revents & ZMQ_POLLIN) {
                    ProcessInbound(socket);
                }
            }
        }
        catch (const std::exception& e) {
            TM_LOG_WARN("LMCache MQ polling thread failed: server={}, error={}", server_addr_, e.what());
            RejectAll(e.what());
        }
        stopping_.store(true, std::memory_order_relaxed);
        RejectAll("LMCache message queue stopped");
    }

    std::string    server_addr_;
    zmq::context_t context_;
    int            notifier_fd_{-1};

    static constexpr std::uint64_t             kIgnoredResponseUid = 0;
    std::uint64_t                              next_uid_{kIgnoredResponseUid + 1};
    std::atomic<bool>                          stopping_{false};
    std::mutex                                 outbound_mutex_;
    std::deque<Task>                           outbound_;
    std::unordered_map<std::uint64_t, Pending> pending_;
    std::thread                                thread_;
};

MessageQueueClient::MessageQueueClient(std::string server_addr): impl_{std::make_unique<Impl>(std::move(server_addr))}
{
}

MessageQueueClient::~MessageQueueClient() = default;

RequestHandle MessageQueueClient::Submit(RequestType request_type, ResponseFrames payloads, ResponseMode response_mode)
{
    if (response_mode == ResponseMode::kIgnore) {
        impl_->Submit(request_type, std::move(payloads), {});
        return {};
    }

    auto          state = std::make_shared<RequestState>();
    RequestHandle request{state};
    impl_->Submit(request_type, std::move(payloads), std::move(state));
    return request;
}

void MessageQueueClient::Stop() noexcept
{
    impl_->Stop();
}

}  // namespace turbomind::lmcache

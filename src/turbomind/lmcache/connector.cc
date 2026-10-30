#include "src/turbomind/lmcache/connector.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/lmcache/mq.h"
#include "src/turbomind/lmcache/protocol.h"
#include "src/turbomind/lmcache/transfer_geometry.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace turbomind::lmcache {
namespace {
using Clock = std::chrono::steady_clock;
using protocol::Pack;
using protocol::RequestType;
using protocol::Unpack;

constexpr int kHeartbeatIntervalMs    = 10'000;
constexpr int kHeartbeatTimeoutMs     = 10'000;
constexpr int kRegistrationTimeoutMs  = 120'000;
constexpr int kShutdownRequestTimeout = 1'000;
constexpr int kWaitSliceMs            = 100;

std::string FormatDeviceUuid(const cudaUUID_t& uuid)
{
    const auto*        bytes = reinterpret_cast<const unsigned char*>(uuid.bytes);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            out << '-';
        }
        out << std::setw(2) << static_cast<unsigned int>(bytes[i]);
    }
    return out.str();
}

std::int64_t MakeInstanceId()
{
    std::random_device                          random;
    std::uniform_int_distribution<std::int64_t> distribution{1, std::numeric_limits<std::int64_t>::max()};
    return distribution(random);
}

}  // namespace

class Connector::Impl {
public:
    explicit Impl(ConnectorConfig config): config_{std::move(config)}, instance_id_{MakeInstanceId()}
    {
        mq_         = std::make_unique<MessageQueueClient>(config_.server_addr);
        chunk_size_ = FetchChunkSize();
        TM_LOG_DEBUG("LMCache connector connected: server={}, worker_id={}, chunk_size={}",
                     config_.server_addr,
                     config_.worker_id,
                     chunk_size_);
        heartbeat_thread_ = std::thread(&Impl::HeartbeatLoop, this);
    }

    ~Impl()
    {
        stopping_.store(true, std::memory_order_release);
        heartbeat_cv_.notify_all();
        if (heartbeat_thread_.joinable()) {
            heartbeat_thread_.join();
        }
        UnregisterForShutdown();
        mq_->Stop();
    }

    int chunk_size() const noexcept
    {
        return chunk_size_;
    }

    std::int64_t instance_id() const noexcept
    {
        return instance_id_;
    }

    bool healthy() const noexcept
    {
        return healthy_.load(std::memory_order_acquire);
    }

    const ConnectorConfig& config() const noexcept
    {
        return config_;
    }

    std::string MakeScopedRequestId(const std::string& request_id) const
    {
        return (config_.session_id.empty() ? std::to_string(instance_id_) : config_.session_id) + ":" + request_id;
    }

    std::int64_t BlockId(int pool_index, const void* part_address) const noexcept
    {
        TM_CHECK_GE(pool_index, 0);
        TM_CHECK_LT(pool_index, static_cast<int>(pool_part_bytes_.size()));
        const auto addr      = reinterpret_cast<std::uintptr_t>(part_address);
        const auto pool_base = reinterpret_cast<std::uintptr_t>(pool_base_);
        TM_CHECK_EQ((addr - pool_base) % pool_part_bytes_[pool_index], 0)
            << "LMCache source address is not on the registered block grid";
        return static_cast<std::int64_t>((addr - pool_base) / pool_part_bytes_[pool_index]);
    }

    RequestHandle Submit(RequestType                      request_type,
                         ResponseFrames                   frames,
                         MessageQueueClient::ResponseMode response_mode = MessageQueueClient::ResponseMode::kTrack)
    {
        return mq_->Submit(request_type, std::move(frames), response_mode);
    }

    void Register(const RegistrationConfig& registration)
    {
        TM_CHECK(registration.base);
        TM_CHECK(!registration.pools.empty());

        cudaIpcMemHandle_t memory_handle{};
        TM_CUDA_CHECK(cudaIpcGetMemHandle(&memory_handle, registration.base));
        static_assert(sizeof(memory_handle.reserved) == 64);
        int device{};
        TM_CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        TM_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        std::vector<protocol::TurboMindCudaIPCWrapper> wrappers;
        std::vector<protocol::EngineGroupInfo>         groups;
        pool_part_bytes_.clear();
        pool_part_bytes_.reserve(registration.pools.size());
        for (int i = 0; i < static_cast<int>(registration.pools.size()); ++i) {
            const auto& pool = registration.pools[i];
            TM_CHECK_GT(pool.part_bytes, 0);
            TM_CHECK_GT(pool.tokens_per_block, 0);
            TM_CHECK_EQ(chunk_size_ % pool.tokens_per_block, 0);
            pool_part_bytes_.push_back(pool.part_bytes);

            const auto part     = static_cast<std::int64_t>(pool.part_bytes);
            const auto geometry = SelectTransferGeometry(
                pool.part_bytes, properties.maxThreadsPerBlock, chunk_size_ / pool.tokens_per_block);
            const auto blocks =
                static_cast<std::int64_t>((registration.size - registration.storage_offset_bytes) / pool.part_bytes);
            const auto                head_bytes = part / (2 * geometry.heads);
            protocol::EngineGroupInfo group{i, {}, pool.tokens_per_block, -1};
            for (int slice = 0; slice < geometry.slices; ++slice) {
                protocol::TurboMindCudaIPCWrapper wrapper;
                std::memcpy(wrapper.ipc_handle.data(), memory_handle.reserved, wrapper.ipc_handle.size());
                wrapper.nbytes = registration.size;
                if (geometry.strided) {
                    const auto bytes = static_cast<std::int64_t>(geometry.slice_bytes);
                    wrapper.shape    = {blocks, 1, bytes};
                    wrapper.stride   = {part, bytes, 1};
                }
                else {
                    wrapper.shape  = {blocks, 2, geometry.heads, 1, head_bytes};
                    wrapper.stride = {part, part / 2, head_bytes, head_bytes, 1};
                }
                wrapper.storage_offset = static_cast<std::int64_t>(registration.storage_offset_bytes)
                                         + slice * static_cast<std::int64_t>(geometry.slice_bytes);
                wrapper.device_uuid = FormatDeviceUuid(properties.uuid);
                group.layer_indices.push_back(wrappers.size());
                wrappers.push_back(std::move(wrapper));
            }
            groups.push_back(std::move(group));
        }

        pool_base_ = static_cast<char*>(registration.base) + registration.storage_offset_bytes;

        protocol::LayoutHints hints;
        ResponseFrames        payloads{Pack(instance_id_),
                                protocol::PackWrappers(wrappers),
                                Pack(config_.model_name),
                                Pack(static_cast<std::int64_t>(config_.world_size)),
                                Pack(std::string("vllm")),
                                Pack(hints),
                                Pack(groups)};
        Call(RequestType::kRegisterKvCache, payloads, kRegistrationTimeoutMs);

        {
            std::lock_guard lock{register_mutex_};
            registration_payloads_ = std::move(payloads);
            registered_            = true;
        }
        healthy_.store(true, std::memory_order_release);
        TM_LOG_DEBUG("LMCache connector registered: rank={}/{}, pools={}, size={}, storage_offset={}, model={}",
                     config_.worker_id,
                     config_.world_size,
                     registration.pools.size(),
                     registration.size,
                     registration.storage_offset_bytes,
                     config_.model_name);
    }

private:
    ResponseFrames Call(RequestType request_type, ResponseFrames frames, int timeout_ms)
    {
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
        auto       rpc      = Submit(request_type, std::move(frames));
        while (!rpc.Ready()) {
            if (stopping_.load(std::memory_order_acquire)) {
                throw std::runtime_error("LMCache connector is stopping");
            }
            const auto now = Clock::now();
            if (now >= deadline) {
                throw std::runtime_error("LMCache request timed out");
            }
            rpc.WaitUntil(std::min(deadline, now + std::chrono::milliseconds(kWaitSliceMs)));
        }
        return rpc.Get();
    }

    int FetchChunkSize()
    {
        const auto response   = Call(RequestType::kGetChunkSize, {}, kHeartbeatTimeoutMs);
        const auto chunk_size = Unpack<std::int64_t>(response.at(0));
        if (chunk_size <= 0 || chunk_size > std::numeric_limits<int>::max()) {
            throw std::runtime_error("LMCache returned an invalid chunk size");
        }
        return static_cast<int>(chunk_size);
    }

    void UnregisterForShutdown()
    {
        bool registered = false;
        {
            std::lock_guard lock{register_mutex_};
            registered  = registered_;
            registered_ = false;
        }
        if (!registered) {
            return;
        }
        try {
            // The heartbeat has already stopped; the MQ stays alive until
            // the unregister response confirms that the daemon dropped IPC.
            auto rpc = Submit(RequestType::kUnregisterKvCache, {Pack(instance_id_)});
            if (!rpc.WaitUntil(Clock::now() + std::chrono::milliseconds(kShutdownRequestTimeout))) {
                throw std::runtime_error("LMCache unregister timed out");
            }
            rpc.Get();
            TM_LOG_DEBUG("LMCache connector unregistered: rank={}/{}", config_.worker_id, config_.world_size);
        }
        catch (const std::exception& e) {
            TM_LOG_WARN("LMCache shutdown unregister failed: server={}, error={}", config_.server_addr, e.what());
        }
    }

    void MonitorServer()
    {
        ResponseFrames                ping_frames;
        std::optional<ResponseFrames> recover_payloads;
        const bool                    was_healthy = healthy_.load(std::memory_order_acquire);
        {
            std::lock_guard lock{register_mutex_};
            ping_frames = {
                Pack(registered_ ? std::optional<std::int64_t>{instance_id_} : std::optional<std::int64_t>{})};
            if (!was_healthy && registered_) {
                recover_payloads = registration_payloads_;
            }
        }
        try {
            const auto response = Call(RequestType::kPing, std::move(ping_frames), kHeartbeatTimeoutMs);
            if (!Unpack<bool>(response.at(0))) {
                throw std::runtime_error("LMCache PING returned an invalid response");
            }
            if (stopping_.load(std::memory_order_acquire)) {
                return;
            }
            if (recover_payloads) {
                Call(RequestType::kRegisterKvCache, *recover_payloads, kRegistrationTimeoutMs);
            }
            if (stopping_.load(std::memory_order_acquire)) {
                return;
            }
            if (!healthy_.exchange(true, std::memory_order_acq_rel)) {
                TM_LOG_WARN("LMCache server recovered: server={}{}",
                            config_.server_addr,
                            recover_payloads ? "; KV cache registration restored" : "");
            }
        }
        catch (const std::exception& e) {
            if (!stopping_.load(std::memory_order_acquire) && healthy_.exchange(false, std::memory_order_acq_rel)) {
                TM_LOG_WARN("LMCache server unhealthy: server={}, reason={}", config_.server_addr, e.what());
            }
        }
    }

    void HeartbeatLoop()
    {
        while (!stopping_.load(std::memory_order_acquire)) {
            MonitorServer();
            std::unique_lock<std::mutex> lock(heartbeat_mutex_);
            if (heartbeat_cv_.wait_for(lock, std::chrono::milliseconds(kHeartbeatIntervalMs), [&] {
                    return stopping_.load(std::memory_order_acquire);
                })) {
                break;
            }
        }
    }

    ConnectorConfig    config_;
    const std::int64_t instance_id_{};
    int                chunk_size_{};
    std::atomic<bool>  healthy_{true};

    void*                    pool_base_{};
    std::vector<std::size_t> pool_part_bytes_{};

    std::mutex     register_mutex_;
    bool           registered_{false};
    ResponseFrames registration_payloads_;

    std::unique_ptr<MessageQueueClient> mq_;

    std::mutex              heartbeat_mutex_;
    std::condition_variable heartbeat_cv_;
    std::atomic<bool>       stopping_{false};
    std::thread             heartbeat_thread_;
};

Connector::Connector(ConnectorConfig config): impl_{std::make_unique<Impl>(std::move(config))} {}
Connector::~Connector() = default;

int Connector::chunk_size() const noexcept
{
    return impl_->chunk_size();
}

std::int64_t Connector::instance_id() const noexcept
{
    return impl_->instance_id();
}

bool Connector::healthy() const noexcept
{
    return impl_->healthy();
}

void Connector::Register(const RegistrationConfig& registration)
{
    impl_->Register(registration);
}

std::int64_t Connector::BlockId(int pool_index, const void* part_address) const noexcept
{
    return impl_->BlockId(pool_index, part_address);
}

const ConnectorConfig& Connector::config() const noexcept
{
    return impl_->config();
}

std::string Connector::MakeScopedRequestId(const std::string& request_id) const
{
    return impl_->MakeScopedRequestId(request_id);
}

RequestHandle Connector::Submit(protocol::RequestType            request_type,
                                ResponseFrames                   frames,
                                MessageQueueClient::ResponseMode response_mode)
{
    return impl_->Submit(request_type, std::move(frames), response_mode);
}

}  // namespace turbomind::lmcache

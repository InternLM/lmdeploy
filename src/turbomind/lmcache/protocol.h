#pragma once

#include <msgpack.hpp>
#include <msgpack/adaptor/cpp17/optional.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace turbomind::lmcache {

using Bytes = std::vector<std::uint8_t>;

}  // namespace turbomind::lmcache

namespace turbomind::lmcache::protocol {

template<class T>
Bytes Pack(const T& value)
{
    msgpack::sbuffer buffer;
    msgpack::pack(buffer, value);
    const auto* begin = reinterpret_cast<const std::uint8_t*>(buffer.data());
    return {begin, begin + buffer.size()};
}

template<class T>
T Unpack(const Bytes& bytes)
{
    if (bytes.empty()) {
        throw std::runtime_error("LMCache protocol returned an empty frame");
    }
    return msgpack::unpack(reinterpret_cast<const char*>(bytes.data()), bytes.size()).get().as<T>();
}

// Values are frozen to the LMCache MP server RequestType enum.
enum class RequestType : std::int64_t
{
    kRegisterKvCache     = 1,
    kUnregisterKvCache   = 2,
    kStore               = 3,
    kRetrieve            = 4,
    kLookup              = 5,
    kQueryPrefetchStatus = 6,
    kWaitPrefetchStatus  = 7,
    kFreeLookupLocks     = 9,
    kEndSession          = 10,
    kGetChunkSize        = 18,
    kPing                = 19,
};

inline constexpr std::int8_t kTurboMindCudaIpcExtensionCode = 2;

struct IPCCacheServerKey {
    std::string                 model_name;
    std::int64_t                world_size{};
    std::optional<std::int64_t> worker_id;
    std::vector<std::int64_t>   token_ids;
    std::int64_t                start{};
    std::int64_t                end{};
    std::string                 request_id;
    std::string                 cache_salt;

    MSGPACK_DEFINE_MAP(model_name, world_size, worker_id, token_ids, start, end, request_id, cache_salt);
};

struct LayoutHints {
    std::string  kv_layout{"HND"};
    std::int64_t num_kv_heads{1};
    std::int64_t tokens_per_block{1};
    std::int64_t head_dim{1};

    MSGPACK_DEFINE_MAP(kv_layout, num_kv_heads, tokens_per_block, head_dim);
};

struct EngineGroupInfo {
    std::int64_t              engine_group_id{};
    std::vector<std::int64_t> layer_indices;
    std::int64_t              tokens_per_block{};
    std::int64_t              sw_size_tokens{-1};

    MSGPACK_DEFINE_MAP(engine_group_id, layer_indices, tokens_per_block, sw_size_tokens);
};

struct TurboMindCudaIPCWrapper {
    std::array<unsigned char, 64> ipc_handle{};
    std::uint64_t                 nbytes{};
    std::string                   dtype{"uint8"};
    std::vector<std::int64_t>     shape;
    std::vector<std::int64_t>     stride;
    std::int64_t                  storage_offset{};
    std::string                   device_uuid;
};

struct TransferResponse {
    Bytes event_handle;
    bool  success{};
};

template<class Stream>
msgpack::packer<Stream>& PackWrappers(msgpack::packer<Stream>&                    packer,
                                      const std::vector<TurboMindCudaIPCWrapper>& wrappers)
{
    packer.pack_array(static_cast<std::uint32_t>(wrappers.size()));
    for (const auto& wrapper : wrappers) {
        msgpack::sbuffer                  body;
        msgpack::packer<msgpack::sbuffer> value(body);
        value.pack_map(7);
        value.pack(std::string("ipc_handle"));
        value.pack_bin(static_cast<std::uint32_t>(wrapper.ipc_handle.size()));
        value.pack_bin_body(reinterpret_cast<const char*>(wrapper.ipc_handle.data()), wrapper.ipc_handle.size());
        value.pack(std::string("nbytes"));
        value.pack(wrapper.nbytes);
        value.pack(std::string("dtype"));
        value.pack(wrapper.dtype);
        value.pack(std::string("shape"));
        value.pack(wrapper.shape);
        value.pack(std::string("stride"));
        value.pack(wrapper.stride);
        value.pack(std::string("storage_offset"));
        value.pack(wrapper.storage_offset);
        value.pack(std::string("device_uuid"));
        value.pack(wrapper.device_uuid);
        const auto size = static_cast<std::uint32_t>(body.size());
        packer.pack_ext(size, kTurboMindCudaIpcExtensionCode);
        packer.pack_ext_body(body.data(), size);
    }
    return packer;
}

inline Bytes PackWrappers(const std::vector<TurboMindCudaIPCWrapper>& wrappers)
{
    msgpack::sbuffer                  buffer;
    msgpack::packer<msgpack::sbuffer> packer(buffer);
    PackWrappers(packer, wrappers);
    const auto* begin = reinterpret_cast<const std::uint8_t*>(buffer.data());
    return {begin, begin + buffer.size()};
}

inline Bytes PackBinary(const void* data, std::size_t size)
{
    msgpack::sbuffer                  buffer;
    msgpack::packer<msgpack::sbuffer> packer(buffer);
    packer.pack_bin(static_cast<std::uint32_t>(size));
    packer.pack_bin_body(static_cast<const char*>(data), size);
    const auto* begin = reinterpret_cast<const std::uint8_t*>(buffer.data());
    return {begin, begin + buffer.size()};
}

inline TransferResponse UnpackTransferResponse(const Bytes& bytes)
{
    if (bytes.empty()) {
        throw std::runtime_error("LMCache protocol returned an empty frame");
    }
    const auto  object_handle = msgpack::unpack(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    const auto& object        = object_handle.get();
    if (object.type != msgpack::type::ARRAY || object.via.array.size != 2) {
        throw std::runtime_error("LMCache transfer response must contain [event_handle, success]");
    }
    const auto& handle = object.via.array.ptr[0];
    if (handle.type != msgpack::type::BIN) {
        throw std::runtime_error("LMCache transfer response contains a non-binary event handle");
    }
    const auto& success = object.via.array.ptr[1];
    if (success.type != msgpack::type::BOOLEAN) {
        throw std::runtime_error("LMCache transfer response contains a non-boolean result");
    }
    const auto* begin = reinterpret_cast<const std::uint8_t*>(handle.via.bin.ptr);
    return {{begin, begin + handle.via.bin.size}, success.as<bool>()};
}

}  // namespace turbomind::lmcache::protocol

MSGPACK_ADD_ENUM(turbomind::lmcache::protocol::RequestType);

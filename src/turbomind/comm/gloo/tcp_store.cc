// Copyright (c) OpenMMLab. All rights reserved.

#include <chrono>
#include <netdb.h>
#include <thread>
#include <unistd.h>

#include <gloo/transport/tcp/device.h>
#include <gloo/transport/tcp/socket.h>

#include "src/turbomind/comm/gloo/tcp_store.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind::comm {

namespace {

// copy from pytorch https://github.com/pytorch/pytorch/blob/v2.8.0-rc4/torch/csrc/distributed/c10d/TCPStoreBackend.hpp

static const uint32_t validationMagicNumber = 0x3C85F7CE;

enum class CheckResponseType : uint8_t
{
    READY,
    NOT_READY
};

enum class QueryType : uint8_t
{
    VALIDATE,
    SET,
    COMPARE_SET,
    GET,
    ADD,
    CHECK,
    WAIT,
    GETNUMKEYS,
    DELETE_KEY,
    APPEND,
    MULTI_GET,
    MULTI_SET,
    CANCEL_WAIT,
    PING,
    QUEUE_PUSH,
    QUEUE_POP,
    QUEUE_LEN,
};

}  // namespace

struct Buffer {
    std::vector<char> buffer;

    template<typename T, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
    void append(T val)
    {
        char* ptr = (char*)&val;
        buffer.insert(buffer.end(), ptr, ptr + sizeof(T));
    }

    void append(const std::vector<char>& vec)
    {
        append((uint64_t)vec.size());
        buffer.insert(buffer.end(), vec.begin(), vec.end());
    }

    void append(const std::string& str)
    {
        append((uint64_t)str.size());
        buffer.insert(buffer.end(), str.begin(), str.end());
    }

    const char* data() const
    {
        return buffer.data();
    }

    size_t count() const
    {
        return buffer.size();
    }
};

void validate(std::shared_ptr<::gloo::transport::tcp::Socket>& socket)
{
    Buffer buffer;
    buffer.append(QueryType::VALIDATE);
    buffer.append(validationMagicNumber);
    socket->write(buffer.data(), buffer.count());
}

void ping(std::shared_ptr<::gloo::transport::tcp::Socket>& socket)
{
    Buffer buffer;
    buffer.append(QueryType::PING);
    uint32_t nonce         = getpid();
    uint32_t returnedNonce = -1;
    buffer.append(nonce);
    socket->write(buffer.data(), buffer.count());
    int r = socket->read(&returnedNonce, sizeof(returnedNonce));
    if (nonce != returnedNonce) {
        std::stringstream ss;
        ss << "Ping failed, nonce=" << nonce << ", returnedNonce=" << returnedNonce << ", socket read=" << r;
        throw std::runtime_error(ss.str());
    }
}

TCPStore::TCPStore(const std::string& host, int port)
{
    auto retry = 0;
    do {
        try {
            ::addrinfo hints{}, *res{};
            hints.ai_flags    = AI_V4MAPPED | AI_ALL | AI_NUMERICSERV;
            hints.ai_family   = AF_UNSPEC;
            hints.ai_socktype = SOCK_STREAM;

            int status = getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res);

            std::shared_ptr<addrinfo> holder(res, [](addrinfo* p) {
                if (p != nullptr) {
                    freeaddrinfo(p);
                }
            });

            if (status != 0) {
                throw std::runtime_error("getaddrinfo failed: " + std::string(gai_strerror(status)));
            }

            for (::addrinfo* addr = res; addr != nullptr; addr = addr->ai_next) {
                int fd = ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
                if (fd == -1) {
                    continue;
                }
                auto socket = std::make_shared<::gloo::transport::tcp::Socket>(fd);
                socket->connect(addr->ai_addr, addr->ai_addrlen);
                socket->noDelay(true);
                socket->recvTimeout(std::chrono::milliseconds(5000));
                socket->sendTimeout(std::chrono::milliseconds(5000));
                validate(socket);  // validate the connection
                ping(socket);      // check send/recv
                socket_ = std::move(socket);
                break;
            }

            if (socket_ == nullptr) {
                throw std::runtime_error("unable to connect to " + host + ":" + std::to_string(port));
            }
        }
        catch (const std::exception& e) {
            TM_LOG_WARNING("[TM][COMM] Failed to connect to store after %d retries: %s", retry, e.what());
            std::this_thread::sleep_for(std::chrono::seconds(1));
            retry += 1;
        }
    } while (socket_ == nullptr);
}

void TCPStore::set(const std::string& key, const std::vector<char>& data)
{
    std::lock_guard<std::mutex> lock(mutex_);
    Buffer                      buffer;
    buffer.append(QueryType::SET);
    buffer.append(key);
    buffer.append(data);
    socket_->write(buffer.data(), buffer.count());
}

std::vector<char> TCPStore::get(const std::string& key)
{
    wait({key});
    std::lock_guard<std::mutex> lock(mutex_);
    Buffer                      buffer;
    buffer.append(QueryType::GET);
    buffer.append(key);
    socket_->write(buffer.data(), buffer.count());

    uint64_t vec_size;
    socket_->read(&vec_size, sizeof(vec_size));
    std::vector<char> value(vec_size);
    socket_->read(value.data(), value.size());
    return value;
}

bool TCPStore::check(const std::vector<std::string>& keys)
{
    std::lock_guard<std::mutex> lock(mutex_);
    Buffer                      buffer;
    buffer.append(QueryType::CHECK);
    buffer.append((uint64_t)keys.size());
    for (const auto& key : keys) {
        buffer.append(key);
    }
    socket_->write(buffer.data(), buffer.count());

    CheckResponseType response;
    socket_->read(&response, sizeof(response));
    return response == CheckResponseType::READY;
}

void TCPStore::wait(const std::vector<std::string>& keys, const std::chrono::milliseconds& timeout)
{
    const auto start = std::chrono::steady_clock::now();
    while (!check(keys)) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
        if (elapsed > timeout) {
            std::stringstream ss;
            ss << "Wait timeout for key(s): [";
            for (const auto& key : keys) {
                ss << key << " ";
            }
            ss << "]";
            TM_LOG_ERROR("[TM][COMM] %s, elapsed %lld s", ss.str().c_str(), elapsed.count());
            throw std::runtime_error("Wait timeout for key(s): " + ss.str());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

TCPStore::~TCPStore() = default;

}  // namespace turbomind::comm

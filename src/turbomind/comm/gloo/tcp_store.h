// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <mutex>

#include <gloo/rendezvous/store.h>
#include <gloo/transport/tcp/socket.h>

namespace turbomind::comm {

class TCPStore: public gloo::rendezvous::Store {
public:
    explicit TCPStore(const std::string& host, int port);

    ~TCPStore();

    void set(const std::string& key, const std::vector<char>& data) override;

    std::vector<char> get(const std::string& key) override;

    bool check(const std::vector<std::string>& keys);

    void wait(const std::vector<std::string>& keys) override
    {
        wait(keys, std::chrono::seconds(30));
    }

    void wait(const std::vector<std::string>& keys, const std::chrono::milliseconds& timeout) override;

private:
    std::shared_ptr<::gloo::transport::tcp::Socket> socket_;
    std::mutex                                      mutex_;
};

}  // namespace turbomind::comm

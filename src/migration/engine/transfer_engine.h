#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/rdma_transport.h"

namespace migration {

using mr_hash_key_t = uint64_t;
typedef enum {
    SUCCESS
} trans_status_t;

class TransferEngine {
public:
    TransferEngine() {}

    int64_t registerRDMAContext(const std::string key, RDMAContext ctx)
    {
        throw std::runtime_error("Not Implemented Error");
    }

private:
    std::unordered_map<std::string, RDMAContext> transport_links_ = {};
};

}  // namespace migration

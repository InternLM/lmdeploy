#pragma once

#include "utils/json.hpp"
#include "utils/logging.h"

#include <cstdint>
#include <functional>
#include <infiniband/verbs.h>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>

namespace slime {

using json = nlohmann::json;

typedef struct RDMAInfo {
    uint32_t      qpn;
    union ibv_gid gid;
    int64_t       gidx;
    uint16_t      lid;
    uint64_t      psn;
    uint64_t      mtu;
    RDMAInfo() {}
    RDMAInfo(uint32_t qpn, union ibv_gid gid, int64_t gidx, uint16_t lid, uint64_t psn, uint64_t mtu):
        qpn(qpn), gidx(gidx), lid(lid), psn(psn), mtu(mtu), gid(gid)
    {
    }

    RDMAInfo(json json_config)
    {
        gid.global.subnet_prefix = json_config["gid"]["subnet_prefix"];
        gid.global.interface_id  = json_config["gid"]["interface_id"];
        gidx                     = json_config["gidx"];
        lid                      = json_config["lid"];
        qpn                      = json_config["qpn"];
        psn                      = json_config["psn"];
        mtu                      = json_config["mtu"];
    }

    json to_json()
    {
        json gid_config{{"subnet_prefix", gid.global.subnet_prefix}, {"interface_id", gid.global.interface_id}};
        return json{{"gid", gid_config}, {"gidx", gidx}, {"lid", lid}, {"qpn", qpn}, {"psn", psn}, {"mtu", mtu}};
    }

} rdma_info_t;

enum class WrType {
    BASE,
    RDMA_READ_ACK,
    RDMA_WRITE_ACK,
    RDMA_SEND_ACK,
    RDMA_RECV_ACK,
};

struct wr_info_base {
protected:
    WrType wr_type;

public:
    wr_info_base(WrType wr_type): wr_type(wr_type) {}
    virtual ~wr_info_base() = default;
    WrType get_wr_type() const
    {
        return wr_type;
    }
};

struct write_info: wr_info_base {
    std::function<void(int64_t)> callback;
    write_info(std::function<void(int64_t)> callback): wr_info_base(WrType::RDMA_WRITE_ACK), callback(callback) {}
};

struct read_info: wr_info_base {
    // call back function.
    std::function<void(unsigned int)> callback;
    read_info(std::function<void(int64_t)> callback): wr_info_base(WrType::RDMA_READ_ACK), callback(callback) {}
};

struct send_info: wr_info_base {
    std::function<void(unsigned int)> callback;
    send_info(std::function<void(int64_t)> callback): wr_info_base(WrType::RDMA_SEND_ACK), callback(callback) {}
};

struct recv_info: wr_info_base {
    std::function<void(unsigned int)> callback;
    recv_info(std::function<void(int64_t)> callback): wr_info_base(WrType::RDMA_RECV_ACK), callback(callback) {}
};

};  // namespace slime

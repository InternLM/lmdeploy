#pragma once

#include "utils/logging.h"
#include <cstdint>
#include <functional>
#include <infiniband/verbs.h>
#include <iostream>
#include <string>
#include <tuple>

namespace migration {

typedef struct TransferConfig {
    size_t ib_port;
    size_t gid_index;
} transfer_config_t;

transfer_config_t loadGlobalConfig();

typedef struct RDMAInfo {
    uint32_t      qpn;
    union ibv_gid gid;
    int64_t       gidx;
    uint16_t      lid;
    uint64_t      psn;
    uint64_t      mtu;
    RDMAInfo() {}
    RDMAInfo(uint32_t qpn,
             uint64_t gid_subnet_prefix,
             uint64_t gid_interface_id,
             int64_t  gidx,
             uint16_t lid,
             uint64_t psn,
             uint64_t mtu):
        qpn(qpn), gidx(gidx), lid(lid), psn(psn), mtu(mtu)
    {
        gid.global = {gid_subnet_prefix, gid_interface_id};
    }
    RDMAInfo(uint32_t qpn, union ibv_gid gid, int64_t gidx, uint16_t lid, uint64_t psn, uint64_t mtu):
        RDMAInfo(qpn, gid.global.interface_id, gid.global.subnet_prefix, gidx, lid, psn, mtu)
    {
    }
    std::tuple<uint64_t, uint64_t> get_gid()
    {
        return {gid.global.subnet_prefix, gid.global.interface_id};
    }
    void set_gid(std::tuple<uint64_t, uint64_t> remote_gid)
    {
        gid.global = {std::get<0>(remote_gid), std::get<1>(remote_gid)};
    }
    void log()
    {
        MIGRATION_LOG_INFO("GID: " << gid.global.subnet_prefix << ", " << gid.global.interface_id);
        MIGRATION_LOG_INFO("GIDX: " << gidx);
        MIGRATION_LOG_INFO("LID: " << lid);
        MIGRATION_LOG_INFO("QPN: " << qpn);
        MIGRATION_LOG_INFO("PSN: " << psn);
        MIGRATION_LOG_INFO("MTU: " << mtu);
    }
} rdma_info_t;

enum class WrType {
    BASE,
    RDMA_READ_ACK,
    RDMA_WRITE_ACK,
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
    std::function<void(int)> callback;
    write_info(std::function<void(int)> callback): wr_info_base(WrType::RDMA_WRITE_ACK), callback(callback) {}
};

struct read_info: wr_info_base {
    // call back function.
    std::function<void(unsigned int)> callback;
    read_info(std::function<void(unsigned int)> callback): wr_info_base(WrType::RDMA_READ_ACK), callback(callback) {}
};

};  // namespace migration
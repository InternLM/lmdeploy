#include "engine/rdma_transport.h"
#include "engine/config.h"
#include "engine/memory_pool.h"
#include "utils/ibv_helper.h"
#include "utils/logging.h"
#include "utils/utils.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include <bits/socket.h>
#include <infiniband/verbs.h>
#include <stdexcept>

namespace slime {

// this number should be big for lots of RMDA_WRITE requests
#define MAX_SEND_WR 8192

// this is only used for recving RDMA_SEND or IMM data. this should be bigger
// than max layers of model.
#define MAX_RECV_WR 8192

#define POLL_COUNT 64

RDMAContext::~RDMAContext()
{
    stop_cq_future();
}

void RDMAContext::launch_cq_future()
{
    cq_future_ = std::async(std::launch::async, [this]() -> void { cq_poll_handle(); });
}

void RDMAContext::stop_cq_future()
{
    if (!stop_ && cq_future_.valid()) {
        stop_ = true;

        // create fake wr to wake up cq thread
        ibv_req_notify_cq(cq_, 0);
        struct ibv_sge sge;
        memset(&sge, 0, sizeof(sge));
        sge.addr   = (uintptr_t)this;
        sge.length = sizeof(*this);
        sge.lkey   = 0;

        struct ibv_send_wr send_wr;
        memset(&send_wr, 0, sizeof(send_wr));
        // send_wr.wr_id      = (uintptr_t)this;
        send_wr.wr_id      = 0;
        send_wr.sg_list    = &sge;
        send_wr.num_sge    = 1;
        send_wr.opcode     = IBV_WR_SEND;
        send_wr.send_flags = IBV_SEND_SIGNALED;

        struct ibv_send_wr* bad_send_wr;
        {
            std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);
            ibv_post_send(qp_, &send_wr, &bad_send_wr);
        }
        // wait thread done
        cq_future_.get();
    }
}

int64_t RDMAContext::cq_poll_handle()
{
    SLIME_LOG_INFO("Polling CQ");

    if (!connected_) {
        SLIME_LOG_ERROR("Start CQ handle before connected, please construct first");
        return -1;
    }
    if (comp_channel_ == NULL)
        SLIME_LOG_ERROR("comp_channel_ should be constructed");

    while (!stop_) {
        struct ibv_cq* ev_cq;
        void*          cq_context;

        if (ibv_get_cq_event(comp_channel_, &ev_cq, &cq_context) != 0) {
            SLIME_LOG_ERROR("Failed to get CQ event");
            return -1;
        }

        ibv_ack_cq_events(ev_cq, 1);
        if (ibv_req_notify_cq(ev_cq, 0) != 0) {
            SLIME_LOG_ERROR("Failed to request CQ notification");
            return -1;
        }

        struct ibv_wc wc[POLL_COUNT];

        while (size_t nr_poll = ibv_poll_cq(cq_, POLL_COUNT, wc)) {
            if (nr_poll < 0) {
                SLIME_LOG_WARN("Worker: Failed to poll completion queues");
                continue;
            }
            for (size_t i = 0; i < nr_poll; ++i) {
                int64_t status_code;
                if (wc[i].status == IBV_WC_SUCCESS) {
                    SLIME_LOG_INFO("WR completed successfully.");
                    status_code = 200;
                }
                else {
                    SLIME_LOG_ERROR("WR failed with status: " << ibv_wc_status_str(wc[i].status) << std::endl);
                    status_code = wc[i].status;
                }
                if (wc[i].wr_id != 0) {
                    wr_info_base* ptr  = reinterpret_cast<wr_info_base*>(wc[i].wr_id);
                    auto*         info = reinterpret_cast<read_info*>(ptr);
                    switch (WrType wr_type = ptr->get_wr_type()) {
                        case WrType::RDMA_SEND_ACK:
                        case WrType::RDMA_RECV_ACK:
                        case WrType::RDMA_READ_ACK:
                            info->callback(wc[i].imm_data);
                            break;
                        default:
                            SLIME_ABORT("Unimplemented WrType " << int64_t(wr_type));
                    }
                    delete info;
                }
            }
        }
    }
    return 0;
}

int64_t
RDMAContext::send_async(std::string mr_key, uint64_t offset, uint64_t length, std::function<void(int64_t)> callback)
{
    send_info* callback_info = new send_info(callback);

    int ret;

    struct ibv_mr* mr        = memory_pool_.get_mr(mr_key);
    json           remote_mr = memory_pool_.get_remote_mr(mr_key);

    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr   = (uintptr_t)mr->addr + offset;
    sge.length = length;
    sge.lkey   = mr->lkey;

    struct ibv_send_wr wr, *bad_wr = NULL;
    memset(&wr, 0, sizeof(wr));

    wr.wr_id      = (uintptr_t)callback_info;
    wr.opcode     = IBV_WR_SEND;
    wr.sg_list    = &sge;
    wr.num_sge    = 1;
    wr.send_flags = IBV_SEND_SIGNALED;

    {
        std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);
        ret = ibv_post_send(qp_, &wr, &bad_wr);
    }

    if (ret) {
        SLIME_LOG_ERROR("Failed to post RDMA send : " << strerror(ret));
        return -1;
    }

    return 0;
}

int64_t
RDMAContext::recv_async(std::string mr_key, uint64_t offset, uint64_t length, std::function<void(int64_t)> callback)
{
    recv_info* callback_info = new recv_info(callback);

    int ret;

    struct ibv_mr* mr        = memory_pool_.get_mr(mr_key);
    json           remote_mr = memory_pool_.get_remote_mr(mr_key);

    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr   = (uintptr_t)mr->addr + offset;
    sge.length = length;
    sge.lkey   = mr->lkey;

    struct ibv_recv_wr wr, *bad_wr = NULL;
    memset(&wr, 0, sizeof(wr));

    wr.wr_id   = (uintptr_t)callback_info;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    {
        std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);
        ret = ibv_post_recv(qp_, &wr, &bad_wr);
    }

    if (ret) {
        SLIME_LOG_ERROR("Failed to post RDMA send : " << strerror(ret));
        return -1;
    }

    return 0;
}

int64_t RDMAContext::r_rdma_async(std::string                  mr_key,
                                  uintptr_t                    target_offset,
                                  uintptr_t                    source_offset,
                                  uint64_t                     length,
                                  std::function<void(int64_t)> callback)
{
    read_info* callback_info = new read_info([callback](unsigned int code) { callback(code); });

    int ret;

    struct ibv_mr* mr        = memory_pool_.get_mr(mr_key);
    json           remote_mr = memory_pool_.get_remote_mr(mr_key);

    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr   = (uintptr_t)mr->addr + source_offset;
    sge.length = length;
    sge.lkey   = mr->lkey;

    struct ibv_send_wr wr, *bad_wr = NULL;
    memset(&wr, 0, sizeof(wr));

    wr.wr_id               = (uintptr_t)callback_info;
    wr.opcode              = IBV_WR_RDMA_READ;
    wr.sg_list             = &sge;
    wr.num_sge             = 1;
    wr.send_flags          = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_mr["addr"].get<uint64_t>() + target_offset;
    wr.wr.rdma.rkey        = remote_mr["rkey"].get<uint32_t>();

    {
        std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);
        ret = ibv_post_send(qp_, &wr, &bad_wr);
    }

    if (ret) {
        SLIME_LOG_ERROR("Failed to post RDMA send : " << strerror(ret));
        return -1;
    }

    return 0;
}

int64_t RDMAContext::batch_r_rdma_async(std::string                  mr_key,
                                        const std::vector<uint64_t>& target_offsets,
                                        const std::vector<uint64_t>& source_offsets,
                                        uint64_t                     length,
                                        std::function<void(int64_t)> callback)
{
    auto*  call_back_info = new read_info(callback);
    size_t batch_size     = target_offsets.size();

    struct ibv_send_wr* bad_wr      = NULL;
    struct ibv_send_wr* wr          = new ibv_send_wr[batch_size];
    struct ibv_sge*     sge         = new ibv_sge[batch_size];
    struct ibv_mr*      mr          = memory_pool_.get_mr(mr_key);
    json                remote_mr   = memory_pool_.get_remote_mr(mr_key);
    uint64_t            remote_addr = remote_mr["addr"].get<uint64_t>();
    uint32_t            remote_rkey = remote_mr["rkey"].get<uint32_t>();
    for (size_t i = 0; i < batch_size; ++i) {
        memset(&sge[i], 0, sizeof(ibv_sge));
        sge[i].addr   = (uint64_t)mr->addr + source_offsets[i];
        sge[i].length = length;
        sge[i].lkey   = mr->lkey;

        wr[i].wr_id               = (i == batch_size - 1) ? (uintptr_t)call_back_info : 0;
        wr[i].opcode              = IBV_WR_RDMA_READ;
        wr[i].sg_list             = &sge[i];
        wr[i].num_sge             = 1;
        wr[i].send_flags          = (i == batch_size - 1) ? IBV_SEND_SIGNALED : 0;
        wr[i].wr.rdma.remote_addr = remote_addr + target_offsets[i];
        wr[i].wr.rdma.rkey        = remote_rkey;
        wr[i].next                = (i == batch_size - 1) ? NULL : &wr[i + 1];
    }

    int ret = 0;
    {
        std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);
        ret = ibv_post_send(qp_, wr, &bad_wr);
    }

    delete[] wr;
    delete[] sge;

    if (ret) {
        SLIME_LOG_ERROR("Failed to post RDMA send : " << strerror(ret));
        return -1;
    }

    return 0;
}

int64_t RDMAContext::modify_qp_to_rtsr(RDMAInfo remote_rdma_info)
{
    int                ret;
    struct ibv_qp_attr attr = {};
    int                flags;

    SLIME_ASSERT(!connected_, "Already connected!");
    remote_rdma_info_ = std::move(remote_rdma_info);

    // Modify QP to Ready to Receive (RTR) state
    memset(&attr, 0, sizeof(attr));
    attr.qp_state           = IBV_QPS_RTR;
    attr.path_mtu           = (enum ibv_mtu)std::min((uint32_t)remote_rdma_info_.mtu, (uint32_t)local_rdma_info_.mtu);
    attr.dest_qp_num        = remote_rdma_info_.qpn;
    attr.rq_psn             = remote_rdma_info_.psn;
    attr.max_dest_rd_atomic = 16;
    attr.min_rnr_timer      = 12;
    attr.ah_attr.dlid       = 0;
    attr.ah_attr.sl         = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num      = 1;

    if (local_rdma_info_.gidx == -1) {
        // IB
        attr.ah_attr.dlid      = local_rdma_info_.lid;
        attr.ah_attr.is_global = 0;
    }
    else {
        // RoCE v2
        attr.ah_attr.is_global      = 1;
        attr.ah_attr.grh.dgid       = remote_rdma_info.gid;
        attr.ah_attr.grh.sgid_index = local_rdma_info_.gidx;
        attr.ah_attr.grh.hop_limit  = 1;
    }

    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC
            | IBV_QP_MIN_RNR_TIMER;

    ret = ibv_modify_qp(qp_, &attr, flags);
    if (ret) {
        SLIME_LOG_ERROR("Failed to modify QP to RTR: reason: " << strerror(ret));
        return -1;
    }

    // Modify QP to RTS state
    memset(&attr, 0, sizeof(attr));
    attr.qp_state      = IBV_QPS_RTS;
    attr.timeout       = 14;
    attr.retry_cnt     = 7;
    attr.rnr_retry     = 7;
    attr.sq_psn        = local_rdma_info_.psn;
    attr.max_rd_atomic = 16;

    flags =
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

    ret = ibv_modify_qp(qp_, &attr, flags);
    if (ret) {
        SLIME_LOG_ERROR("Failed to modify QP to RTS");
        return -1;
    }
    SLIME_LOG_INFO("RDMA exchange done");
    connected_ = true;

    if (ibv_req_notify_cq(cq_, 0)) {
        SLIME_LOG_ERROR("Failed to request notify for CQ");
        return -1;
    }
    return 0;
}

int64_t RDMAContext::init_rdma_context(std::string dev_name, uint8_t ib_port, std::string link_type)
{
    uint16_t      lid;
    enum ibv_mtu  active_mtu;
    union ibv_gid gid;
    int64_t       gidx;
    uint32_t      psn;

    if (initialized_) {
        SLIME_LOG_ERROR("allready initialized.");
        return -1;
    }

    /* Get RDMA Device Info */
    struct ibv_device** dev_list;
    struct ibv_device*  ib_dev;
    int                 num_devices;
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        SLIME_LOG_ERROR("Failed to get RDMA devices list");
        return -1;
    }

    for (int i = 0; i < num_devices; ++i) {
        char* dev_name_from_list = (char*)ibv_get_device_name(dev_list[i]);
        if (strcmp(dev_name_from_list, dev_name.c_str()) == 0) {
            SLIME_LOG_INFO("found device " << dev_name_from_list);
            ib_dev  = dev_list[i];
            ib_ctx_ = ibv_open_device(ib_dev);
            break;
        }
    }

    if (!ib_ctx_) {
        SLIME_LOG_WARN("Can't find or failed to open the specified device, try to open "
                       "the default device "
                       << (char*)ibv_get_device_name(dev_list[0]));
        ib_ctx_ = ibv_open_device(dev_list[0]);
        if (!ib_ctx_) {
            SLIME_LOG_ERROR("Failed to open the default device");
            return -1;
        }
    }

    struct ibv_device_attr device_attr;
    if (ibv_query_device(ib_ctx_, &device_attr) != 0)
        SLIME_LOG_ERROR("Failed to query device");
    SLIME_LOG_DEBUG("Max Memory Region:" << device_attr.max_mr);
    SLIME_LOG_DEBUG("Max Memory Region Size:" << device_attr.max_mr_size);
    SLIME_LOG_DEBUG("Max Memory QP WR:" << device_attr.max_qp_wr);

    struct ibv_port_attr port_attr;
    ib_port_ = ib_port;
    if (ibv_query_port(ib_ctx_, ib_port, &port_attr)) {
        SLIME_LOG_ERROR("Unable to query port {} attributes\n" << ib_port_);
        return -1;
    }
    if ((port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND && link_type == "Ethernet")
        || (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET && link_type == "IB")) {
        SLIME_LOG_ERROR("port link layer and config link type don't match");
        return -1;
    }
    if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        gidx = -1;
    }
    else {
        gidx = ibv_find_sgid_type(ib_ctx_, ib_port_, IBV_GID_TYPE_ROCE_V2, AF_INET);
        if (gidx < 0) {
            SLIME_LOG_ERROR("Failed to find GID");
            return -1;
        }
    }

    lid        = port_attr.lid;
    active_mtu = port_attr.active_mtu;

    /* Alloc Protected Domain (PD) */
    pd_ = ibv_alloc_pd(ib_ctx_);
    if (!pd_) {
        SLIME_LOG_ERROR("Failed to allocate PD");
        return -1;
    }
    memory_pool_ = MemoryPool(pd_);

    /* Alloc Complete Queue (CQ) */
    SLIME_ASSERT(ib_ctx_, "init rdma context first");
    comp_channel_ = ibv_create_comp_channel(ib_ctx_);
    cq_           = ibv_create_cq(ib_ctx_, MAX_SEND_WR + MAX_RECV_WR, NULL, comp_channel_, 0);
    SLIME_ASSERT(cq_, "create CQ failed");

    /* Create Queue Pair (QP) */
    struct ibv_qp_init_attr qp_init_attr = {};
    qp_init_attr.send_cq                 = cq_;
    qp_init_attr.recv_cq                 = cq_;
    qp_init_attr.qp_type                 = IBV_QPT_RC;  // Reliable Connection
    qp_init_attr.cap.max_send_wr         = MAX_SEND_WR;
    qp_init_attr.cap.max_recv_wr         = MAX_RECV_WR;
    qp_init_attr.cap.max_send_sge        = 1;
    qp_init_attr.cap.max_recv_sge        = 1;
    qp_init_attr.sq_sig_all              = false;

    qp_ = ibv_create_qp(pd_, &qp_init_attr);
    if (!qp_) {
        SLIME_LOG_ERROR("Failed to create QP");
        return -1;
    }

    /* Modify QP to INIT state */
    struct ibv_qp_attr attr = {};
    attr.qp_state           = IBV_QPS_INIT;
    attr.port_num           = ib_port_;
    attr.pkey_index         = 0;
    attr.qp_access_flags =
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    int ret = ibv_modify_qp(qp_, &attr, flags);
    if (ret) {
        SLIME_LOG_ERROR("Failed to modify QP to INIT");
    }

    /* Set Packet Sequence Number (PSN) */
    srand48(time(NULL));
    psn = lrand48() & 0xffffff;

    /* Get GID */
    if (gidx != -1 && ibv_query_gid(ib_ctx_, 1, gidx, &gid)) {
        SLIME_LOG_ERROR("Failed to get GID");
    }

    /* Set Local RDMA Info */
    local_rdma_info_.gidx = gidx;
    local_rdma_info_.qpn  = qp_->qp_num;
    local_rdma_info_.psn  = psn;
    local_rdma_info_.gid  = gid;
    local_rdma_info_.lid  = lid;
    local_rdma_info_.mtu  = (uint32_t)active_mtu;

    initialized_ = true;
    return 0;
}
}  // namespace slime

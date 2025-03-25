#include "rdma_transport.h"
#include "ibv_helper.h"
#include "logging.h"
#include "utils.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <unistd.h>
#include <vector>

#include <bits/socket.h>
#include <infiniband/verbs.h>
#include <stdexcept>

namespace migration {

// this number should be big for lots of RMDA_WRITE requests
#define MAX_SEND_WR 8192

// this is only used for recving RDMA_SEND or IMM data. this should be bigger
// than max layers of model.
#define MAX_RECV_WR 64

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
        send_wr.wr_id      = (uintptr_t)this;
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

void RDMAContext::cq_poll_handle()
{
    MIGRATION_LOG_INFO("Polling CQ");

    MIGRATION_ASSERT(connected_, "Please construct first");
    MIGRATION_ASSERT(comp_channel_ != nullptr, "comp_channel_ should be constructed");

    while (!stop_) {
        struct ibv_cq* ev_cq;
        void*          cq_context;

        if (ibv_get_cq_event(comp_channel_, &ev_cq, &cq_context) != 0) {
            MIGRATION_ABORT("Failed to get CQ event");
        }

        ibv_ack_cq_events(ev_cq, 1);
        if (ibv_req_notify_cq(ev_cq, 0) != 0) {
            MIGRATION_ABORT("Failed to request CQ notification");
        }

        struct ibv_wc wc = {0};

        while (ibv_poll_cq(cq_, 1, &wc) > 0) {
            if (wc.status == IBV_WC_SUCCESS) {
                MIGRATION_LOG_INFO("RDMA READ completed successfully." << std::endl);
                if (wc.wr_id != 0) {
                    wr_info_base* ptr = reinterpret_cast<wr_info_base*>(wc.wr_id);
                    if (ptr->get_wr_type() == WrType::RDMA_READ_ACK) {
                        MIGRATION_LOG_DEBUG("read cache done: Received IMM, imm_data: ", wc.imm_data);
                        auto* info = reinterpret_cast<read_info*>(ptr);
                        info->callback(wc.imm_data);
                        delete info;
                    }
                }
            }
            else {
                std::cerr << "RDMA READ failed with status: " << ibv_wc_status_str(wc.status) << std::endl;
            }
        }
    }
}

int64_t RDMAContext::batch_r_rdma_async(const std::vector<uintptr_t>&     target_addrs,
                                        const std::vector<uintptr_t>&     source_addrs,
                                        uint64_t                          lengths,
                                        std::string                       mr_key,
                                        int64_t                           remote_rkey,
                                        std::function<void(unsigned int)> callback)
{
    auto* call_back_info = new read_info([callback](unsigned int code) { callback(code); });
    int   batch_size     = target_addrs.size();

    struct ibv_send_wr* batch_wr_head = NULL;
    struct ibv_send_wr* cur_wr        = NULL;
    struct ibv_send_wr* bad_wr        = NULL;
    for (int i = 0; i < batch_size; ++i) {
        struct ibv_sge sge;
        memset(&sge, 0, sizeof(sge));
        sge.addr   = source_addrs[i];
        sge.length = lengths;
        sge.lkey   = memory_region_[mr_key]->lkey;

        struct ibv_send_wr* wr = NULL;
        wr                     = (ibv_send_wr*)malloc(sizeof(ibv_send_wr));
        memset(wr, 0, sizeof(ibv_send_wr));

        wr->wr_id               = 0;
        wr->opcode              = IBV_WR_RDMA_READ;
        wr->sg_list             = &sge;
        wr->num_sge             = 1;
        wr->send_flags          = IBV_SEND_SIGNALED;
        wr->wr.rdma.remote_addr = target_addrs[i];
        wr->wr.rdma.rkey        = remote_rkey;

        if (batch_wr_head == NULL) {
            batch_wr_head = wr;
        }
        if (cur_wr != NULL) {
            cur_wr->next = wr;
        }
        cur_wr = wr;
    }

    if (cur_wr != NULL) {
        // Only call the callback at the last wr
        cur_wr->wr_id = (uintptr_t)call_back_info;
        cur_wr->next  = NULL;
    }

    int ret = 0;
    {
        std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);
        ret = ibv_post_send(qp_, batch_wr_head, &bad_wr);
    }

    if (ret) {
        MIGRATION_ABORT("Failed to post RDMA send : " << strerror(ret));
        return -1;
    }

    return 0;
}

int64_t RDMAContext::r_rdma_async(uintptr_t                         target_addr,
                                  uintptr_t                         source_addr,
                                  uint64_t                          length,
                                  std::string                       mr_key,
                                  int64_t                           remote_rkey,
                                  std::function<void(unsigned int)> callback)
{
    auto* call_back_info = new read_info([callback](unsigned int code) { callback(code); });

    int ret;

    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr   = source_addr;
    sge.length = length;
    sge.lkey   = memory_region_[mr_key]->lkey;

    struct ibv_send_wr wr, *bad_wr = NULL;
    memset(&wr, 0, sizeof(wr));

    wr.wr_id               = (uintptr_t)call_back_info;
    wr.opcode              = IBV_WR_RDMA_READ;
    wr.sg_list             = &sge;
    wr.num_sge             = 1;
    wr.send_flags          = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = target_addr;
    wr.wr.rdma.rkey        = remote_rkey;

    {
        std::unique_lock<std::mutex> lock(rdma_post_send_mutex_);
        ret = ibv_post_send(qp_, &wr, &bad_wr);
    }

    if (ret) {
        MIGRATION_ABORT("Failed to post RDMA send : " << strerror(ret));
        return -1;
    }

    return 0;
}

void RDMAContext::modify_qp_to_rtsr(RDMAInfo remote_rdma_info)
{
    int                ret;
    struct ibv_qp_attr attr = {};
    int                flags;

    MIGRATION_ASSERT(!connected_, "Already connected!");
    remote_rdma_info_ = std::move(remote_rdma_info);

    // Modify QP to Ready to Receive (RTR) state
    memset(&attr, 0, sizeof(attr));
    attr.qp_state           = IBV_QPS_RTR;
    attr.path_mtu           = (enum ibv_mtu)std::min((uint32_t)remote_rdma_info_.mtu, (uint32_t)local_rdma_info_.mtu);
    attr.dest_qp_num        = remote_rdma_info_.qpn;
    attr.rq_psn             = remote_rdma_info_.psn;
    attr.max_dest_rd_atomic = 4;
    attr.min_rnr_timer      = 12;
    attr.ah_attr.dlid       = 0;  // RoCE v2 is used.
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

    struct ibv_device_attr device_attr;
    ibv_query_device(ib_ctx_, &device_attr);

    ret = ibv_modify_qp(qp_, &attr, flags);
    if (ret) {
        MIGRATION_ABORT("Failed to modify QP to RTR: reason: " << strerror(ret));
    }

    // Modify QP to RTS state
    memset(&attr, 0, sizeof(attr));
    attr.qp_state      = IBV_QPS_RTS;
    attr.timeout       = 14;
    attr.retry_cnt     = 7;
    attr.rnr_retry     = 7;
    attr.sq_psn        = local_rdma_info_.psn;
    attr.max_rd_atomic = 1;

    flags =
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

    ret = ibv_modify_qp(qp_, &attr, flags);
    if (ret) {
        MIGRATION_ABORT("Failed to modify QP to RTS");
    }
    MIGRATION_LOG_INFO("RDMA exchange done");
    connected_ = true;

    if (ibv_req_notify_cq(cq_, 0)) {
        MIGRATION_ABORT("Failed to request notify for CQ");
    }
}

int64_t RDMAContext::registerMemoryRegion(std::string mem_key, int64_t addr, size_t length)
{

    /* MemoryRegion Access Right = 777 */
    const static int access_rights = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    ibv_mr* mr = ibv_reg_mr(pd_, (void*)addr, length, access_rights);

    MIGRATION_ASSERT(mr, " Failed to register memory " << addr);

    MIGRATION_LOG_INFO("Memory region: " << (void*)addr << " -- " << (void*)((uintptr_t)addr + length)
                                         << ", Device name: " << device_name_ << ", Length: " << length << " ("
                                         << length / 1024 / 1024 << " MB)"
                                         << ", Permission: " << access_rights << ", LKey: " << mr->lkey
                                         << ", RKey: " << mr->rkey);

    memory_region_[mem_key] = mr;
    return 0;
}

int64_t RDMAContext::init_rdma_context(std::string dev_name, uint8_t ib_port, std::string link_type)
{
    uint16_t      lid;
    enum ibv_mtu  active_mtu;
    union ibv_gid gid;
    int64_t       gidx;
    uint32_t      psn;

    MIGRATION_ASSERT(!initialized_, "allready initialized.");

    /* Get RDMA Device Info */
    struct ibv_device** dev_list;
    struct ibv_device*  ib_dev;
    int                 num_devices;
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        MIGRATION_ABORT("Failed to get RDMA devices list");
        return -1;
    }

    for (int i = 0; i < num_devices; ++i) {
        char* dev_name_from_list = (char*)ibv_get_device_name(dev_list[i]);
        if (strcmp(dev_name_from_list, dev_name.c_str()) == 0) {
            MIGRATION_LOG_INFO("found device {}" << dev_name_from_list);
            ib_dev  = dev_list[i];
            ib_ctx_ = ibv_open_device(ib_dev);
            break;
        }
    }

    if (!ib_ctx_) {
        MIGRATION_LOG_INFO("Can't find or failed to open the specified device, try to open "
                           "the default device "
                           << (char*)ibv_get_device_name(dev_list[0]));
        ib_ctx_ = ibv_open_device(dev_list[0]);
        if (!ib_ctx_) {
            MIGRATION_ABORT("Failed to open the default device");
            return -1;
        }
    }

    MIGRATION_LOG_INFO("Get NIC:" << dev_name);
    struct ibv_device_attr device_attr;
    MIGRATION_ASSERT_EQ(ibv_query_device(ib_ctx_, &device_attr), 0, "Failed to query device");
    MIGRATION_LOG_INFO("Max Memory Region:" << device_attr.max_mr);
    MIGRATION_LOG_INFO("Max Memory Region Size:" << device_attr.max_mr_size);
    MIGRATION_LOG_INFO("Max Memory QP WR:" << device_attr.max_qp_wr);

    struct ibv_port_attr port_attr;
    ib_port_ = ib_port;
    if (ibv_query_port(ib_ctx_, ib_port, &port_attr)) {
        MIGRATION_ABORT("Unable to query port {} attributes\n" << ib_port_);
        return -1;
    }
    if ((port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND && link_type == "Ethernet")
        || (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET && link_type == "IB")) {
        MIGRATION_ABORT("port link layer and config link type don't match");
        return -1;
    }
    if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        gidx = -1;
    }
    else {
        gidx = ibv_find_sgid_type(ib_ctx_, ib_port_, IBV_GID_TYPE_ROCE_V2, AF_INET);
        if (gidx < 0) {
            MIGRATION_ABORT("Failed to find GID");
            return -1;
        }
    }

    lid        = port_attr.lid;
    active_mtu = port_attr.active_mtu;

    /* Alloc Protected Domain (PD) */
    pd_ = ibv_alloc_pd(ib_ctx_);
    if (!pd_) {
        MIGRATION_ABORT("Failed to allocate PD");
        return -1;
    }

    /* Alloc Complete Queue (CQ) */
    MIGRATION_ASSERT(ib_ctx_, "init rdma context first");
    comp_channel_ = ibv_create_comp_channel(ib_ctx_);
    cq_           = ibv_create_cq(ib_ctx_, MAX_SEND_WR + MAX_RECV_WR, NULL, comp_channel_, 0);
    MIGRATION_ASSERT(cq_, "create CQ failed");

    /* Create Queue Pair (QP) */
    struct ibv_qp_init_attr qp_init_attr = {};
    qp_init_attr.send_cq                 = cq_;
    qp_init_attr.recv_cq                 = cq_;
    qp_init_attr.qp_type                 = IBV_QPT_RC;  // Reliable Connection
    qp_init_attr.cap.max_send_wr         = MAX_SEND_WR;
    qp_init_attr.cap.max_recv_wr         = MAX_RECV_WR;
    qp_init_attr.cap.max_send_sge        = 1;
    qp_init_attr.cap.max_recv_sge        = 1;

    qp_ = ibv_create_qp(pd_, &qp_init_attr);
    if (!qp_) {
        MIGRATION_ABORT("Failed to create QP");
        return -1;
    }

    /* Modify QP to INIT state */
    struct ibv_qp_attr attr = {};
    attr.qp_state           = IBV_QPS_INIT;
    attr.port_num           = 1;
    attr.pkey_index         = 0;
    attr.qp_access_flags    = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;

    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    int ret = ibv_modify_qp(qp_, &attr, flags);
    if (ret) {
        MIGRATION_ABORT("Failed to modify QP to INIT");
    }

    /* Set Packet Sequence Number (PSN) */
    srand48(time(NULL));
    psn = lrand48() & 0xffffff;

    /* Get GID */
    if (gidx != -1 && ibv_query_gid(ib_ctx_, 1, gidx, &gid)) {
        MIGRATION_ABORT("Failed to get GID");
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
}  // namespace migration

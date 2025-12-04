// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/check.h"

namespace turbomind::comm {

extern std::unique_ptr<HostGroupId> CreateThreadGroupId();
extern std::unique_ptr<HostGroupId> CreateGlooGroupId();

struct HybridCommImpl: public HybridHostCommImpl {

    HybridCommImpl(int n_ranks, int rank, int node_rank, HostGroupId* gloo_group_id, HostGroupId* thread_group_id):
        n_ranks_{n_ranks},  //
        rank_{rank},
        node_rank_(node_rank)
    {

        gloo_comm_    = gloo_group_id->CreateCommunicator(n_ranks, rank);
        auto r2nr     = ::turbomind::comm::AllGather(gloo_comm_, node_rank);
        same_process_ = r2nr.front() == r2nr.back();
        if (same_process_) {
            intra_comm_ = thread_group_id->CreateCommunicator(n_ranks, rank);
            return;
        }

        for (int r = 0; r < n_ranks_; ++r) {
            if (r2nr[r] == node_rank) {
                rank_to_intra_[r] = static_cast<int>(ranks_in_node_.size());
                ranks_in_node_.push_back(r);
            }
        }

        intra_comm_ = thread_group_id->CreateCommunicator(ranks_in_node_.size(), rank_to_intra_[rank]);
        inter_comm_ = gloo_comm_->Split(rank_to_intra_[rank_] == 0, 0);

        int inter_rank = inter_comm_->rank();
        ::turbomind::comm::Broadcast(intra_comm_, inter_rank, 0);
        rank_to_inter_ = ::turbomind::comm::AllGather(gloo_comm_, inter_rank);
    }

    HybridCommImpl(std::shared_ptr<HostCommImpl> gloo_comm, std::shared_ptr<HostCommImpl> intra_comm, int node_rank):
        gloo_comm_{std::move(gloo_comm)},
        intra_comm_{std::move(intra_comm)},
        rank_{gloo_comm_->rank()},
        n_ranks_{gloo_comm_->n_ranks()},
        node_rank_(node_rank)
    {
        auto r2nr     = ::turbomind::comm::AllGather(gloo_comm_, node_rank);
        same_process_ = r2nr.front() == r2nr.back();
        if (same_process_) {
            return;
        }

        for (int r = 0; r < n_ranks_; ++r) {
            if (r2nr[r] == node_rank) {
                rank_to_intra_[r] = static_cast<int>(ranks_in_node_.size());
                ranks_in_node_.push_back(r);
            }
        }

        inter_comm_ = gloo_comm_->Split(rank_to_intra_[rank_] == 0, 0);

        int inter_rank = inter_comm_->rank();
        ::turbomind::comm::Broadcast(intra_comm_, inter_rank, 0);
        rank_to_inter_ = ::turbomind::comm::AllGather(gloo_comm_, inter_rank);
    }

    std::shared_ptr<HostCommImpl> Split(int color, int key) override
    {
        if (!is_same_process()) {
            auto new_gloo_comm  = gloo_comm_->Split(color, key);
            auto new_intra_comm = intra_comm_->Split(color, key);
            return std::make_shared<HybridCommImpl>(new_gloo_comm, new_intra_comm, node_rank_);
        }
        else {
            return intra_comm_->Split(color, key);
        }
    }

    int rank() const override
    {
        return rank_;
    }

    int n_ranks() const override
    {
        return n_ranks_;
    }

    IpcHostCommImpl* inter_comm() const override
    {
        return inter_comm_->as<IpcHostCommImpl>();
    }

    HostCommImpl* intra_comm() const override
    {
        return intra_comm_;
    }

    const std::unordered_map<int, int>& get_rank_to_intra() const override
    {
        return rank_to_intra_;
    }

    const std::vector<int>& get_rank_to_inter() const
    {
        return rank_to_inter_;
    }

    bool is_same_process() const override
    {
        return same_process_;
    }

    void Sync(bool blocking) override
    {
        if (!is_same_process() && rank_to_intra_[rank_] == 0) {
            inter_comm_->Sync(blocking);
        }
        intra_comm_->Sync(blocking);
    }

    void Broadcast(void* data, int count, DataType dtype, int root, copy_fn copy) override
    {
        if (is_same_process()) {
            return intra_comm_->Broadcast(data, count, dtype, root, copy);
        }

        bool root_node = rank_to_intra_.count(root) > 0;
        if (root_node) {
            intra_comm_->Broadcast(data, count, dtype, rank_to_intra_[root], copy);
        }
        if (rank_to_intra_[rank_] == 0) {
            inter_comm_->Broadcast(data, count, dtype, rank_to_inter_[root], copy);
        }
        if (!root_node) {
            intra_comm_->Broadcast(data, count, dtype, 0, copy);
        }
    }

    void AllGather(void* data, int count, DataType dtype, copy_fn copy) override
    {
        if (is_same_process()) {
            return intra_comm_->AllGather(data, count, dtype, copy);
        }

        // TODO: support allgatherv in gloo comm (each node may has different rank size)
        return gloo_comm_->AllGather(data, count, dtype, copy);
    }

    void AllReduce(void* data, int count, DataType dtype, RedOp red_op) override
    {
        if (is_same_process()) {
            return intra_comm_->AllReduce(data, count, dtype, red_op);
        }

        intra_comm_->AllReduce(data, count, dtype, red_op);
        if (rank_to_intra_[rank_] == 0) {
            inter_comm_->AllReduce(data, count, dtype, red_op);
        }
        intra_comm_->Broadcast(data, byte_size(dtype) * count, data_type_v<uint8_t>, 0, detail::copy_fn<uint8_t>);
    }

    HostComm gloo_comm_{};   // primitive comm, used for initializing inter_comm and intra_comm
    HostComm inter_comm_{};  // inter-node comm
    HostComm intra_comm_{};  // intra-node comm

    int rank_;       // group rank
    int n_ranks_;    // group size
    int node_rank_;  // node rank

    std::vector<int>             ranks_in_node_;    // group ranks in intra-node
    std::unordered_map<int, int> rank_to_intra_{};  // map group rank to intra-node rank
    std::vector<int>             rank_to_inter_{};  // map group rank to inter-node rank

    bool same_process_;
};

class HybridGroupId: public HostGroupId {
public:
    HybridGroupId()
    {
        thread_group_id_ = CreateThreadGroupId();
        gloo_group_id_   = CreateGlooGroupId();
    }

    void Initialize() override
    {
        thread_group_id_->Initialize();
        gloo_group_id_->Initialize();
    }

    void Export(std::ostream& os) override
    {
        thread_group_id_->Export(os);
        gloo_group_id_->Export(os);
    }

    void Import(std::istream& is) override
    {
        thread_group_id_->Import(is);
        gloo_group_id_->Import(is);
    }

    HostComm CreateCommunicator(int n_ranks, int rank, int node_rank)
    {
        auto impl = std::make_shared<HybridCommImpl>(n_ranks,  //
                                                     rank,
                                                     node_rank,
                                                     gloo_group_id_.get(),
                                                     thread_group_id_.get());
        return std::static_pointer_cast<HostCommImpl>(impl);
    }

    std::unique_ptr<HostGroupId> thread_group_id_;
    std::unique_ptr<HostGroupId> gloo_group_id_;
};

std::unique_ptr<HostGroupId> CreateHybridGroupId()
{
    return std::make_unique<HybridGroupId>();
}

}  // namespace turbomind::comm

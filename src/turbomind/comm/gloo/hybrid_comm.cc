// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/check.h"

namespace turbomind::comm {

extern std::unique_ptr<HostGroupId> CreateThreadGroupId();
extern std::unique_ptr<HostGroupId> CreateGlooGroupId();

struct HybridCommImpl: public HostCommImpl {

    HybridCommImpl(int n_ranks, int rank, int node_rank, HostGroupId* gloo_group_id, HostGroupId* thread_group_id):
        n_ranks_{n_ranks},  //
        rank_{rank},
        node_rank_(node_rank)
    {
        gloo_comm_     = gloo_group_id->CreateCommunicator(n_ranks, rank);
        rank_to_nodes_ = ::turbomind::comm::AllGather(gloo_comm_, node_rank);
        same_process_  = rank_to_nodes_.front() == rank_to_nodes_.back();
        if (same_process_) {
            intra_comm_ = thread_group_id->CreateCommunicator(n_ranks, rank);
        }
        else {
            init_inter_comm();
            intra_comm_ = thread_group_id->CreateCommunicator(intra_n_ranks_, rank_to_intra_[rank_]);
        }
    }

    HybridCommImpl(std::shared_ptr<HostCommImpl> gloo_comm, std::shared_ptr<HostCommImpl> intra_comm, int node_rank):
        gloo_comm_{std::move(gloo_comm)},
        intra_comm_{std::move(intra_comm)},
        rank_{gloo_comm_->rank()},
        n_ranks_{gloo_comm_->n_ranks()},
        node_rank_(node_rank)
    {
        rank_to_nodes_ = ::turbomind::comm::AllGather(gloo_comm_, node_rank);
        same_process_  = rank_to_nodes_.front() == rank_to_nodes_.back();
        if (same_process_) {}
        else {
            init_inter_comm();
        }
    }

    void init_inter_comm()
    {
        int intra_n_ranks = 0;
        int intra_rank    = -1;
        for (int r = 0; r < n_ranks_; ++r) {
            if (rank_to_nodes_[r] == node_rank_) {
                if (r == rank_) {
                    intra_rank = intra_n_ranks;
                }
                intra_n_ranks++;
            }
        }

        intra_n_ranks_ = intra_n_ranks;
        gloo_comm_->AllReduce(&intra_n_ranks_, 1, DataType::kInt, RedOp::kMin);
        TM_CHECK_EQ(intra_n_ranks_, intra_n_ranks) << "The number of ranks in each node should be same.";
        TM_CHECK_GT(intra_rank, -1) << "Invalid intra_rank.";
        rank_to_intra_ = ::turbomind::comm::AllGather(gloo_comm_, intra_rank);

        inter_comm_    = gloo_comm_->Split(rank_to_intra_[rank_], 0);
        rank_to_inter_ = ::turbomind::comm::AllGather(gloo_comm_, inter_comm_->rank());
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

    void Broadcast(void* data, int count, DataType dtype, int root, copy_fn copy, ser_fn ser, des_fn des) override
    {
        if (!ser || !des) {
            return Broadcast(data, count, dtype, root, copy);
        }

        if (rank_to_intra_[root] == rank_to_intra_[rank_]) {  // same ith rank in node
            inter_comm_->Broadcast(data, count, dtype, rank_to_inter_[root], copy, ser, des);
        }
        intra_comm_->Broadcast(data, count, dtype, rank_to_intra_[root], copy);
    }

    void Broadcast(void* data, int count, DataType dtype, int root, copy_fn copy)
    {
        if (is_same_process()) {
            return intra_comm_->Broadcast(data, count, dtype, root, copy);
        }

        if (rank_to_intra_[root] == rank_to_intra_[rank_]) {  // same ith rank in node
            inter_comm_->Broadcast(data, count, dtype, rank_to_inter_[root], copy);
        }
        intra_comm_->Broadcast(data, count, dtype, rank_to_intra_[root], copy);
    }

    void AllGather(void* data, int count, DataType dtype, copy_fn copy, ser_fn ser, des_fn des) override
    {
        if (!ser || !des) {
            return AllGather(data, count, dtype, copy);
        }

        return gloo_comm_->AllGather(data, count, dtype, copy, ser, des);
    }

    void AllGather(void* data, int count, DataType dtype, copy_fn copy)
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
    int intra_n_ranks_;

    std::vector<int> rank_to_nodes_{};  // map group rank to node rank (not global)
    std::vector<int> rank_to_intra_{};  // map group rank to intra-node rank
    std::vector<int> rank_to_inter_{};  // map group rank to inter-node rank

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

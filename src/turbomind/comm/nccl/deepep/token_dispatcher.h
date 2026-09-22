#pragma once

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/core.h"

#include <memory>

namespace turbomind::comm {

class TokenDispatcherImpl;

class TokenDispatcher {
public:
    TokenDispatcher(HostComm h_comm);

    ~TokenDispatcher();

    void Init(int num_max_tokens_per_rank, int hidden, int num_topk, int num_local_experts, bool use_fp8_dispatch);

    void Dispatch(Tensor&       x,
                  Tensor&       topk_idx,
                  Tensor&       topk_weights,
                  int           num_max_tokens_per_rank,
                  Tensor&       out_x,
                  Tensor&       out_topk_weights,
                  Buffer_<int>& f2n,
                  Buffer_<int>& f2E,
                  Buffer_<int>& en2f,
                  Buffer_<int>& offsets);

    void Combine(Tensor& x, Tensor& out_x);

private:
    std::unique_ptr<TokenDispatcherImpl> impl_;
};

}  // namespace turbomind::comm

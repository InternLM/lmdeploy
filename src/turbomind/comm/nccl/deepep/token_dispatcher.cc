#include "src/turbomind/comm/nccl/deepep/token_dispatcher.h"
#include "src/turbomind/comm/nccl/deepep/moe_a2a_utils.h"
#include "src/turbomind/comm/nccl/deepep/symm_ctx.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <nccl.h>

#include "csrc/utils/system.hpp"
#include <deep_ep/common/layout.cuh>

#include "csrc/jit/device_runtime.hpp"
#include "csrc/jit/kernel_runtime.hpp"
#include "csrc/kernels/elastic/combine.hpp"
#include "csrc/kernels/elastic/dispatch.hpp"

#include "csrc/kernels/backend/symmetric.hpp"

#include <mutex>

using namespace deep_ep;
using namespace deep_ep::elastic;

extern "C" void turbomind_nccl_stub_anchor();

namespace turbomind::comm {

class TokenDispatcherImpl {
public:
    TokenDispatcherImpl(HostComm h_comm);

    ~TokenDispatcherImpl();

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
    int64_t GetNumBufferBytes(int num_max_tokens_per_rank, int hidden, int num_topk, bool use_fp8_dispatch) const;

    int64_t GetDispatchBufferSize(int  num_max_tokens_per_rank,
                                  int  hidden,
                                  int  num_sf_packs,
                                  int  num_topk,
                                  int  elem_size,
                                  int  num_scaleout_ranks,
                                  int  num_scaleup_ranks,
                                  bool is_scaleup_nvlink) const;

    int64_t GetCombineBufferSize(int  num_max_tokens_per_rank,
                                 int  hidden,
                                 int  num_topk,
                                 int  num_scaleout_ranks,
                                 int  num_scaleup_ranks,
                                 bool is_scaleup_nvlink,
                                 bool allow_multiple_reduction) const;

    bool initialized_ = false;

    int        rank_;
    int        n_ranks_;
    ncclComm_t comm_{};
    HostComm   h_comm_{};

    // Buffer bytes = GPU buffer + CPU buffer (excludes workspace)
    // Memory layout: [[[Workspace] GPU buffer] CPU buffer]
    int64_t num_buffer_bytes_;
    int64_t num_gpu_buffer_bytes_;
    int64_t num_cpu_buffer_bytes_{};  // not used
    void*   buffer_{};

    // Timeout settings
    int     num_cpu_timeout_secs_;
    int64_t num_gpu_timeout_cycles_;

    // Workspace
    void* workspace_{};
    void* host_workspace_{};
    void* mapped_host_workspace_{};

    std::shared_ptr<layout::WorkspaceLayout> workspace_layout_wo_expert_;

    // NCCL context
    std::shared_ptr<NCCLSymmetricMemoryContext> nccl_context_;

    // Some EP hybrid mode settings
    static constexpr int kNumMaxChannelsPerSM = 8;
    static constexpr int kNumMaxSMs           = 160;
    static constexpr int kNumMaxChannels      = kNumMaxChannelsPerSM * kNumMaxSMs;

    inline static std::once_flag init_flag_{};

    int num_sms_;
    int num_qps_;
    int num_local_experts_;
    int num_experts_;
    int num_max_tokens_per_rank_;
    int num_pre_max_tokens_per_rank_{};

    Tensor_<int>   topk_idx_;
    Tensor_<int>   psum_num_recv_tokens_per_scaleup_rank_;
    Tensor_<int>   psum_num_recv_tokens_per_expert_;
    Tensor_<int>   num_unaligned_recv_tokens_per_expert_;
    Buffer_<int>   dst_buffer_slot_idx_buf_;
    Tensor_<int>   dst_buffer_slot_idx_;
    Buffer_<int>   token_metadata_at_forward_buf_;
    Tensor_<int>   token_metadata_at_forward_;
    Buffer_<int>   channel_linked_list_buf_;
    Tensor_<int>   channel_linked_list_;
    Tensor         recv_x_;
    Tensor_<int>   recv_topk_idx_;
    Tensor_<float> recv_topk_weights_;
    Tensor_<int>   recv_src_metadata_;
};

TokenDispatcherImpl::TokenDispatcherImpl(HostComm h_comm): h_comm_{h_comm}
{
    // Keep the NCCL compatibility stub in DT_NEEDED even when the build-time NCCL
    // already provides every DeepEP symbol and the linker uses --as-needed.
    turbomind_nccl_stub_anchor();

    rank_    = h_comm->rank();
    n_ranks_ = h_comm->n_ranks();

    ncclUniqueId uid{};
    if (rank_ == 0) {
        NCCLCHECK(ncclGetUniqueId(&uid));
    }
    static_assert(std::is_trivially_copyable_v<ncclUniqueId>);
    Broadcast(h_comm, uid, 0);

    NCCLCHECK(ncclCommInitRank(&comm_, n_ranks_, uid, rank_));

    // Create NCCL symmetric memory context
    nccl_context_ = std::make_shared<NCCLSymmetricMemoryContext>(comm_, n_ranks_, rank_);

    // prepare root path, using absolute path for development
    static auto cuda_home = FindCudaHome();
    static auto nccl_root = FindNcclRoot();
    static auto lib_root  = FindLibRoot();
    std::call_once(init_flag_, [&] {
        deep_ep::jit::Compiler::prepare_init(lib_root, cuda_home, nccl_root);
        deep_ep::jit::KernelRuntime::prepare_init(cuda_home);
        deep_ep::jit::IncludeParser::prepare_init(lib_root);
        TM_LOG_INFO("cuda_home: {}, nccl_root: {}, lib_root: {}", cuda_home, nccl_root, lib_root);
    });

    // barrier last
    h_comm_->Sync();
}

TokenDispatcherImpl::~TokenDispatcherImpl()
{
    nccl_context_.reset();
    if (auto ec = ncclCommDestroy(comm_); ec != ncclSuccess) {
        TM_LOG_ERROR("Rank {}: Failed to destory communicator: {}", rank_, ncclGetErrorString(ec));
    }
}

void TokenDispatcherImpl::Init(
    int num_max_tokens_per_rank, int hidden, int num_topk, int num_local_experts, bool use_fp8_dispatch)
{
    TM_CHECK(!use_fp8_dispatch) << "FP8 dispatch is not supported yet";
    num_buffer_bytes_     = GetNumBufferBytes(num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch);
    num_gpu_buffer_bytes_ = num_buffer_bytes_ - num_cpu_buffer_bytes_;

    // Workspace is aligned to 2 MB so that it sits cleanly at the front of the GPU segment
    const auto num_workspace_bytes =
        math::align<int64_t>(layout::WorkspaceLayout::get_num_bytes(), symmetric::kNumAlignmentBytes);

    const auto num_sym_bytes = num_workspace_bytes + num_buffer_bytes_;
    nccl_context_->Init(num_sym_bytes, num_cpu_buffer_bytes_);

    // Verify the symmetric memory layout matches our expectations
    TM_CHECK(num_workspace_bytes + num_gpu_buffer_bytes_ == nccl_context_->num_gpu_bytes_);
    TM_CHECK(num_cpu_buffer_bytes_ == nccl_context_->num_cpu_bytes_);

    // Timeout
    num_cpu_timeout_secs_   = 300;
    num_gpu_timeout_cycles_ = 100 * jit::device_runtime->get_clock_rate();

    // Assign workspaces and buffers
    workspace_                  = this->nccl_context_->mapped_window_ptr_;
    workspace_layout_wo_expert_ = std::make_shared<layout::WorkspaceLayout>(
        workspace_, nccl_context_->num_scaleout_ranks_, nccl_context_->num_scaleup_ranks_, 0);
    buffer_ = static_cast<uint8_t*>(workspace_) + num_workspace_bytes;
    TM_CUDA_CHECK(cudaMemset(workspace_, 0, num_workspace_bytes));

    // Allocate host workspaces
    // we use do_cpu_sync=False, so we don't need to allocate host workspace

    // Settings
    num_sms_                 = std::min(64, jit::device_runtime->get_num_sms());  // not prefer_overlap_with_compute
    num_qps_                 = std::min(num_sms_ * 16 + 1, nccl_context_->num_allocated_qps_);
    num_local_experts_       = num_local_experts;
    num_experts_             = num_local_experts * nccl_context_->num_ranks_;
    num_max_tokens_per_rank_ = num_max_tokens_per_rank;

    // Allocate buffer
    psum_num_recv_tokens_per_scaleup_rank_ = {{nccl_context_->num_scaleup_ranks_}, kDEVICE};
    psum_num_recv_tokens_per_expert_       = {{num_local_experts + 1}, kDEVICE};
    num_unaligned_recv_tokens_per_expert_  = {{num_local_experts}, kDEVICE};

    dst_buffer_slot_idx_buf_ = {num_max_tokens_per_rank * num_topk, kDEVICE};

    recv_x_            = {{num_max_tokens_per_rank * nccl_context_->num_ranks_, hidden}, kBfloat16, kDEVICE};
    recv_topk_idx_     = {{num_max_tokens_per_rank * nccl_context_->num_ranks_, num_topk}, kDEVICE};
    recv_topk_weights_ = {{num_max_tokens_per_rank * nccl_context_->num_ranks_, num_topk}, kDEVICE};
    recv_src_metadata_ = {{num_max_tokens_per_rank * nccl_context_->num_ranks_, num_topk + 2}, kDEVICE};

    // barrier last
    TM_CUDA_CHECK(cudaDeviceSynchronize());
    h_comm_->Sync();
    initialized_ = true;
}

void TokenDispatcherImpl::Dispatch(Tensor&       x,
                                   Tensor&       topk_idx,
                                   Tensor&       topk_weights,
                                   int           num_max_tokens_per_rank,
                                   Tensor&       out_x,
                                   Tensor&       out_topk_weights,
                                   Buffer_<int>& f2n,
                                   Buffer_<int>& f2E,
                                   Buffer_<int>& en2f,
                                   Buffer_<int>& offsets)
{
    const auto [num_tokens, hidden] = x.shapes(0, 1);
    const int num_topk              = topk_idx.shape(1);
    TM_CHECK(num_max_tokens_per_rank <= num_max_tokens_per_rank_);
    TM_CHECK(x.dtype() == kBfloat16);
    TM_CHECK(topk_idx.dtype() == kInt32);
    TM_CHECK(topk_weights.dtype() == kFloat32);
    topk_idx_                    = topk_idx;
    num_pre_max_tokens_per_rank_ = num_max_tokens_per_rank;

    // Decide number of channels by shared memory consumption
    // Only for hybrid version
    int       num_channels_per_sm = 1, num_channels = 1;
    const int num_smem_bytes = jit::device_runtime->get_num_smem_bytes();
    if (nccl_context_->num_scaleout_ranks_ > 1) {
        const auto dispatch_token_layout =
            get_dispatch_token_layout(hidden, byte_size(x.dtype()), 0 /* num_sf_packs */, num_topk);
        const auto combine_token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
        TM_CHECK(num_sms_ <= kNumMaxSMs);
        num_channels_per_sm =
            std::min<int>((num_smem_bytes - get_num_notify_smem_bytes(nccl_context_->num_ranks_, num_experts_))
                              / dispatch_token_layout.get_num_bytes<true>(),
                          32 - kNumNotifyWarps);
        num_channels_per_sm =
            std::min<int>(num_smem_bytes / combine_token_layout.get_num_bytes<true>(), num_channels_per_sm);
        num_channels_per_sm = std::min<int>(
            /* 2 kinds of warps */ num_channels_per_sm / 2, kNumMaxChannelsPerSM);
        num_channels_per_sm = std::min<int>(num_channels_per_sm, 4);
        num_channels        = num_sms_ * num_channels_per_sm;
        TM_CHECK(num_channels > 0);
    }

    // Non-hybrid mode handles
    if (nccl_context_->num_scaleout_ranks_ == 1) {
        dst_buffer_slot_idx_ = {dst_buffer_slot_idx_buf_, {num_tokens, num_topk}};
    }

    token_metadata_at_forward_ = {};
    channel_linked_list_       = {};
    if (nccl_context_->num_scaleout_ranks_ > 1) {
        auto AssignTensor = [](auto& tensor, auto& buffer, const Layout& layout) {
            TM_CHECK_EQ(buffer.dtype(), tensor.dtype());
            if (buffer.size() < layout.size()) {
                using BufferType = std::remove_cvref_t<decltype(buffer)>;
                buffer           = BufferType{layout.size(), kDEVICE};
            }
            tensor = {buffer, layout};
        };

        const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels);
        const auto num_max_forwarded_tokens   = nccl_context_->num_scaleout_ranks_ * num_max_tokens_per_channel + 1;
        const auto num_forward_metadata_dims  = 2 + num_topk * 2;
        AssignTensor(dst_buffer_slot_idx_,
                     dst_buffer_slot_idx_buf_,
                     {num_channels, nccl_context_->num_scaleout_ranks_, num_max_tokens_per_channel, num_topk});
        AssignTensor(token_metadata_at_forward_,
                     token_metadata_at_forward_buf_,
                     {num_channels, num_max_forwarded_tokens, num_forward_metadata_dims});
        AssignTensor(channel_linked_list_,
                     channel_linked_list_buf_,
                     {num_channels,
                      nccl_context_->num_scaleout_ranks_ * num_max_tokens_per_channel + 1,
                      nccl_context_->num_scaleup_ranks_});
    }

    launch_dispatch(x.data_or((void*)nullptr),
                    nullptr /* sf_ptr */,
                    topk_idx.data_or((int32_t*)nullptr),
                    topk_weights.data_or((float*)nullptr),
                    nullptr /* copied_topk_idx */,
                    nullptr /* cumulative_local_expert_recv_stats */,
                    psum_num_recv_tokens_per_scaleup_rank_.data(),
                    psum_num_recv_tokens_per_expert_.data(),
                    num_unaligned_recv_tokens_per_expert_.data(),
                    dst_buffer_slot_idx_.data(),
                    token_metadata_at_forward_.data_or((int*)nullptr),
                    num_tokens,
                    num_max_tokens_per_rank,
                    hidden,
                    byte_size(x.dtype()),
                    0 /* num_sf_packs*/,
                    0 /* sf_token_stride*/,
                    0 /* sf_hidden_stride*/,
                    num_experts_,
                    num_topk,
                    1,
                    nccl_context_->dev_comm_,
                    nccl_context_->window_,
                    buffer_,
                    workspace_,
                    mapped_host_workspace_,
                    nccl_context_->scaleout_rank_idx_,
                    nccl_context_->scaleup_rank_idx_,
                    nccl_context_->num_scaleout_ranks_,
                    nccl_context_->num_scaleup_ranks_,
                    nccl_context_->is_scaleup_nvlink_,
                    num_sms_,
                    num_channels_per_sm,
                    num_smem_bytes,
                    num_qps_,
                    num_gpu_timeout_cycles_,
                    false /* cached_mode*/,
                    false /* do_cpu_sync*/,
                    core::Context::stream().handle());
    TM_CUDA_CHECK(cudaGetLastError());

    int num_recv_tokens = num_max_tokens_per_rank * nccl_context_->num_ranks_;  // worst case
    launch_dispatch_copy_epilogue(buffer_,
                                  workspace_,
                                  psum_num_recv_tokens_per_scaleup_rank_.data(),
                                  psum_num_recv_tokens_per_expert_.data() + 1,
                                  recv_x_.raw_data(),
                                  nullptr /* recv_sf */,
                                  recv_topk_idx_.data(),
                                  recv_topk_weights_.data(),
                                  recv_src_metadata_.data(),
                                  channel_linked_list_.data_or((int*)nullptr),
                                  num_unaligned_recv_tokens_per_expert_.data(),
                                  num_recv_tokens,
                                  num_max_tokens_per_rank,
                                  hidden * byte_size(x.dtype()),
                                  0 /* num_sf_packs*/,
                                  0 /* sf_token_stride*/,
                                  0 /* sf_hidden_stride*/,
                                  num_experts_,
                                  num_topk,
                                  1,
                                  nccl_context_->scaleout_rank_idx_,
                                  nccl_context_->scaleup_rank_idx_,
                                  nccl_context_->num_scaleout_ranks_,
                                  nccl_context_->num_scaleup_ranks_,
                                  jit::device_runtime->get_num_sms(),
                                  jit::device_runtime->get_num_smem_bytes(),
                                  num_channels,
                                  false /* do_expand */,
                                  false /* cached_mode */,
                                  false /* do_zero_padding */,
                                  core::Context::stream().handle());
    TM_CUDA_CHECK(cudaGetLastError());

    // The unaligned per-expert counts are no longer needed by non-expanded Combine.
    // Consume them as reverse atomic cursors so mapping needs no extra workspace.
    invokeMoeA2AMapping(f2n.data(),
                        f2E.data(),
                        en2f.data(),
                        recv_topk_idx_.data(),
                        psum_num_recv_tokens_per_scaleup_rank_.data() + nccl_context_->num_scaleup_ranks_ - 1,
                        psum_num_recv_tokens_per_expert_.data(),
                        num_unaligned_recv_tokens_per_expert_.data(),
                        num_recv_tokens,
                        num_topk,
                        num_local_experts_,
                        core::Context::stream().handle());

    // output
    out_x            = {recv_x_.raw_data(), {num_recv_tokens, hidden}, kBfloat16, kDEVICE};
    out_topk_weights = {recv_topk_weights_.raw_data(), {num_recv_tokens, num_topk}, kFloat32, kDEVICE};
    offsets          = {psum_num_recv_tokens_per_expert_.data(), num_local_experts_ + 1, kDEVICE};
}

void TokenDispatcherImpl::Combine(Tensor& x, Tensor& out_x)
{
    TM_CHECK(x.dtype() == kBfloat16);
    const auto [num_reduced_tokens, hidden]    = x.shapes(0, 1);
    const auto [num_combined_tokens, num_topk] = topk_idx_.shapes(0, 1);

    const auto reduce_buffer = launch_combine(x.raw_data(),
                                              nullptr /* topk_weights */,
                                              recv_src_metadata_.data(),
                                              psum_num_recv_tokens_per_scaleup_rank_.data(),
                                              token_metadata_at_forward_.data_or((int*)nullptr),
                                              channel_linked_list_.data_or((int*)nullptr),
                                              nccl_context_->dev_comm_,
                                              nccl_context_->window_,
                                              buffer_,
                                              workspace_,
                                              num_reduced_tokens,
                                              num_pre_max_tokens_per_rank_,
                                              hidden,
                                              num_experts_,
                                              num_topk,
                                              num_qps_,
                                              num_gpu_timeout_cycles_,
                                              nccl_context_->num_scaleout_ranks_,
                                              nccl_context_->num_scaleup_ranks_,
                                              nccl_context_->scaleout_rank_idx_,
                                              nccl_context_->scaleup_rank_idx_,
                                              nccl_context_->is_scaleup_nvlink_,
                                              num_sms_,
                                              jit::device_runtime->get_num_smem_bytes(),
                                              token_metadata_at_forward_ ? token_metadata_at_forward_.shape(0) : 1,
                                              false /* use_expanded_layout */,
                                              true /* allow_multiple_reduction */,
                                              core::Context::stream().handle());
    TM_CUDA_CHECK(cudaGetLastError());

    launch_combine_reduce_epilogue(recv_x_.raw_data(),
                                   nullptr /* combined_topk_weights */,
                                   topk_idx_.data(),
                                   num_combined_tokens,
                                   num_pre_max_tokens_per_rank_,
                                   hidden,
                                   num_experts_,
                                   num_topk,
                                   reduce_buffer,
                                   nullptr /* bias_0 */,
                                   nullptr /* bias_1 */,
                                   nccl_context_->num_scaleout_ranks_,
                                   nccl_context_->num_scaleup_ranks_,
                                   nccl_context_->scaleout_rank_idx_,
                                   nccl_context_->scaleup_rank_idx_,
                                   jit::device_runtime->get_num_sms(),
                                   jit::device_runtime->get_num_smem_bytes(),
                                   false /* use_expanded_layout */,
                                   true /* allow_multiple_reduction */,
                                   core::Context::stream().handle());
    TM_CUDA_CHECK(cudaGetLastError());

    out_x = {recv_x_.raw_data(), {num_combined_tokens, hidden}, kBfloat16, kDEVICE};

    topk_idx_                    = {};
    num_pre_max_tokens_per_rank_ = 0;
}

int64_t TokenDispatcherImpl::GetNumBufferBytes(int  num_max_tokens_per_rank,
                                               int  hidden,
                                               int  num_topk,
                                               bool use_fp8_dispatch) const
{
    TM_CHECK(num_max_tokens_per_rank > 0 && hidden > 0) << "num_max_tokens_per_rank and hidden must be positive";
    // The worst case SF bytes must be less than the main part
    TM_CHECK(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);

    // NOTES: there are lots of `kNumTopk <= 32` restrictions, so we use 32 to calculate token size
    num_topk = num_topk == 0 ? 32 : num_topk;

    // Dispatch size
    const auto elem_size = use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
    const auto num_sf_packs =
        use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0;  // An approximation for number of SF packs
    const auto num_dispatch_bytes = GetDispatchBufferSize(num_max_tokens_per_rank,
                                                          hidden,
                                                          num_sf_packs,
                                                          num_topk,
                                                          elem_size,
                                                          nccl_context_->num_scaleout_ranks_,
                                                          nccl_context_->num_scaleup_ranks_,
                                                          nccl_context_->is_scaleup_nvlink_);

    // Combine layout
    const auto num_combine_bytes = GetCombineBufferSize(num_max_tokens_per_rank,
                                                        hidden,
                                                        num_topk,
                                                        nccl_context_->num_scaleout_ranks_,
                                                        nccl_context_->num_scaleup_ranks_,
                                                        nccl_context_->is_scaleup_nvlink_,
                                                        true /* allow_multiple_reduction */);

    // Return the maximum of those layouts, aligned to 2 MB
    return math::align(std::max(num_dispatch_bytes, num_combine_bytes), symmetric::kNumAlignmentBytes);
}

int64_t TokenDispatcherImpl::GetDispatchBufferSize(int  num_max_tokens_per_rank,
                                                   int  hidden,
                                                   int  num_sf_packs,
                                                   int  num_topk,
                                                   int  elem_size,
                                                   int  num_scaleout_ranks,
                                                   int  num_scaleup_ranks,
                                                   bool is_scaleup_nvlink) const
{
    const auto num_ranks    = num_scaleup_ranks * num_scaleout_ranks;
    const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);

    if (num_scaleout_ranks == 1) {
        // Direct dispatch
        const auto send_buffer_layout =
            layout::BufferLayout<false>(token_layout, is_scaleup_nvlink ? 0 : 1, num_max_tokens_per_rank);
        const auto recv_buffer_layout = layout::BufferLayout<false>(token_layout, num_ranks, num_max_tokens_per_rank);
        return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
    }
    else {
        // Hybrid dispatch
        const auto scaleup_recv_buffer =
            layout::BufferLayout<false>(token_layout, num_scaleup_ranks, num_scaleout_ranks * num_max_tokens_per_rank);
        const auto scaleout_send_buffer = layout::BufferLayout<false>(token_layout, 1, num_max_tokens_per_rank);
        const auto scaleout_recv_buffer = layout::BufferLayout<false>(
            token_layout,
            num_scaleout_ranks,
            /* kNumChannels * kNumMaxTokensPerChannel */ num_max_tokens_per_rank + kNumMaxChannels);
        return scaleup_recv_buffer.get_num_bytes() + scaleout_send_buffer.get_num_bytes()
               + scaleout_recv_buffer.get_num_bytes();
    }
}

int64_t TokenDispatcherImpl::GetCombineBufferSize(int  num_max_tokens_per_rank,
                                                  int  hidden,
                                                  int  num_topk,
                                                  int  num_scaleout_ranks,
                                                  int  num_scaleup_ranks,
                                                  bool is_scaleup_nvlink,
                                                  bool allow_multiple_reduction) const
{
    const auto num_ranks    = num_scaleup_ranks * num_scaleout_ranks;
    const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);

    if (num_scaleout_ranks == 1) {
        // Direct combine
        const auto num_tokens_in_layout = allow_multiple_reduction ? std::min(num_ranks, num_topk) : num_topk;
        const auto send_buffer_layout   = layout::BufferLayout<false>(
            token_layout,
            is_scaleup_nvlink ? 0 : num_ranks,
            // For single reduction cases, the maximum number of received tokens is
            // `num_ranks * num_topk * num_max_tokens_per_rank` (we assume the bad case of `do_expand=True`)
            num_max_tokens_per_rank * (allow_multiple_reduction ? 1 : num_topk));
        const auto recv_buffer_layout =
            layout::BufferLayout<false>(token_layout, num_tokens_in_layout, num_max_tokens_per_rank);
        return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
    }
    else {
        // Hybrid combine
        const int num_tokens_in_scaleup_layout =
            allow_multiple_reduction ? std::min(num_scaleup_ranks, num_topk) : num_topk;
        const int num_tokens_in_scaleout_layout =
            allow_multiple_reduction ? std::min(num_scaleout_ranks, num_topk) : num_topk;
        const auto scaleup_recv_buffer = layout::BufferLayout<false>(
            token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * num_max_tokens_per_rank);
        const auto scaleout_recv_buffer =
            layout::BufferLayout<false>(token_layout, num_tokens_in_scaleout_layout, num_max_tokens_per_rank);
        const auto scaleout_send_buffer =
            layout::BufferLayout<false>(token_layout,
                                        allow_multiple_reduction ? 1 : num_topk,
                                        /* kNumChannels * num_scaleout_ranks * kNumMaxTokensPerChannel */
                                        num_scaleout_ranks * (num_max_tokens_per_rank + kNumMaxChannels));
        return scaleup_recv_buffer.get_num_bytes() + scaleout_send_buffer.get_num_bytes()
               + scaleout_recv_buffer.get_num_bytes();
    }
}

TokenDispatcher::TokenDispatcher(HostComm host_comm): impl_(std::make_unique<TokenDispatcherImpl>(host_comm)) {}

TokenDispatcher::~TokenDispatcher() = default;

void TokenDispatcher::Init(
    int num_max_tokens_per_rank, int hidden, int num_topk, int num_local_experts, bool use_fp8_dispatch)
{
    impl_->Init(num_max_tokens_per_rank, hidden, num_topk, num_local_experts, use_fp8_dispatch);
}

void TokenDispatcher::Dispatch(Tensor&       x,
                               Tensor&       topk_idx,
                               Tensor&       topk_weights,
                               int           num_max_tokens_per_rank,
                               Tensor&       out_x,
                               Tensor&       out_topk_weights,
                               Buffer_<int>& f2n,
                               Buffer_<int>& f2E,
                               Buffer_<int>& en2f,
                               Buffer_<int>& offsets)
{
    impl_->Dispatch(
        x, topk_idx, topk_weights, num_max_tokens_per_rank, out_x, out_topk_weights, f2n, f2E, en2f, offsets);
}

void TokenDispatcher::Combine(Tensor& x, Tensor& out_x)
{
    impl_->Combine(x, out_x);
}

}  // namespace turbomind::comm

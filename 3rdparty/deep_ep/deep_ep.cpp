#include "deep_ep.hpp"

#include "kernels/api.cuh"
#include "kernels/exception.cuh"
#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/string_utils.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <type_traits>
#include <unordered_map>

using turbomind::fmtstr;
using turbomind::round_up;

namespace shared_memory {
void cu_mem_set_access_all(void* ptr, size_t size)
{
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    CUmemAccessDesc access_desc[device_count];
    for (int idx = 0; idx < device_count; ++idx) {
        access_desc[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc[idx].location.id   = idx;
        access_desc[idx].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    CU_CHECK(cuMemSetAccess((CUdeviceptr)ptr, size, access_desc, device_count));
}

void cu_mem_free(void* ptr)
{
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemRelease(handle));
}

size_t get_size_align_to_granularity(size_t size_raw, size_t granularity)
{
    size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
    if (size == 0)
        size = granularity;
    return size;
}

SharedMemoryAllocator::SharedMemoryAllocator(bool use_fabric): use_fabric(use_fabric) {}

void SharedMemoryAllocator::malloc(void** ptr, size_t size_raw)
{
    if (use_fabric) {
        CUdevice device;
        CU_CHECK(cuCtxGetDevice(&device));

        CUmemAllocationProp prop  = {};
        prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        prop.location.id          = device;

        size_t granularity = 0;
        CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        size_t size = get_size_align_to_granularity(size_raw, granularity);

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemCreate(&handle, size, &prop, 0));

        CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    }
    else {
        CUDA_CHECK(cudaMalloc(ptr, size_raw));
    }
}

void SharedMemoryAllocator::free(void* ptr)
{
    if (use_fabric) {
        cu_mem_free(ptr);
    }
    else {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void SharedMemoryAllocator::get_mem_handle(MemHandle* mem_handle, void* ptr)
{
    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    mem_handle->size = size;

    if (use_fabric) {
        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

        CU_CHECK(cuMemExportToShareableHandle(
            &mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    }
    else {
        CUDA_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
    }
}

void SharedMemoryAllocator::open_mem_handle(void** ptr, MemHandle* mem_handle)
{
    if (use_fabric) {
        size_t size = mem_handle->size;

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemImportFromShareableHandle(
            &handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));

        CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, 0, 0, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    }
    else {
        CUDA_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));
    }
}

void SharedMemoryAllocator::close_mem_handle(void* ptr)
{
    if (use_fabric) {
        cu_mem_free(ptr);
    }
    else {
        CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
    }
}
}  // namespace shared_memory

namespace deep_ep {

Buffer::Buffer(int      rank,
               int      num_ranks,
               int64_t  num_nvl_bytes,
               int64_t  num_rdma_bytes,
               int64_t  num_ll_rdma_bytes,
               bool     low_latency_mode,
               bool     enable_shrink,
               bool     use_fabric,
               int      qps_per_rank,
               HostComm h_comm):
    rank(rank),
    num_ranks(num_ranks),
    num_nvl_bytes(num_nvl_bytes),
    low_latency_mode(low_latency_mode),
    num_rdma_bytes(num_rdma_bytes),
    num_ll_rdma_bytes(num_ll_rdma_bytes),
    enable_shrink(enable_shrink),
    shared_memory_allocator(use_fabric),
    qps_per_rank(qps_per_rank),
    h_comm(h_comm)
{
    // Common checks
    EP_STATIC_ASSERT(NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0
                   and (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0
                   and (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(num_nvl_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks
                   and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0) {
        EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);
    }

    // Get ranks
    CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

    // Get device info
    cudaDeviceProp device_prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    num_device_sms = device_prop.multiProcessorCount;

    // Number of per-channel bytes cannot be large
    EP_HOST_ASSERT(ceil_div<int64_t>(num_nvl_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(ceil_div<int64_t>(num_rdma_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());

    auto comm_stream = turbomind::core::Context::stream().handle();

    // Create 32 MiB workspace
    CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    // MoE counter
    CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    // MoE expert-level counter
    CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
        moe_recv_expert_counter[i] = -1;

    // MoE RDMA-level counter
    if (num_rdma_ranks > 0) {
        CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
        *moe_recv_rdma_counter = -1;
    }

    // NVLink
    if (num_nvl_bytes > 0) {
        allocate_sync_nvl_buffer();
    }

    // RDMA
    if (num_rdma_bytes || num_ll_rdma_bytes) {
        allocate_rdma_buffer();
    }

    turbomind::core::Context::stream().Sync();
    h_comm->Sync();

    // Ready to use
    available = true;
}

void Buffer::allocate_sync_nvl_buffer()
{
    // Metadata memory
    int64_t barrier_signal_bytes     = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes         = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

    auto stream = turbomind::core::Context::stream().handle();

    HostComm h_nvl_comm = h_comm->Split(rdma_rank, 0);
    TM_CHECK_EQ(h_nvl_comm->is_same_process(), true);

    ipc_comm = CreateDeviceCommunicator("cuda-ipc", h_nvl_comm->n_ranks(), nvl_rank, h_nvl_comm);

    buffer_ptrs[nvl_rank] =
        ipc_comm->Allocate(num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes);

    buffer_ptrs_gpu =
        reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

    // Set barrier signals
    barrier_signal_ptrs[nvl_rank] =
        reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
    barrier_signal_ptrs_gpu = reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes
                                                      + barrier_signal_bytes + buffer_ptr_bytes);

    // No need to synchronize, will do a full device sync during `sync`
    CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, stream));

    AllGather(h_nvl_comm, buffer_ptrs, 1);

    for (int i = 0; i < num_nvl_ranks; ++i) {
        if (i != nvl_rank) {
            barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
        }
    }

    // Copy all buffer and barrier signal pointers to GPU
    CUDA_CHECK(cudaMemcpyAsync(
        buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(barrier_signal_ptrs_gpu,
                               barrier_signal_ptrs,
                               sizeof(int*) * NUM_MAX_NVL_PEERS,
                               cudaMemcpyHostToDevice,
                               stream));
}

void Buffer::allocate_rdma_buffer()
{
    TM_CHECK_EQ(comm, nullptr);
    if ((not low_latency_mode) and (num_rdma_ranks == 1)) {
        return;
    }

    std::vector<uint8_t> unique_ids;
    if (rank == 0) {
        unique_ids = deep_ep::internode::get_unique_id();
    }
    Broadcast(h_comm, unique_ids, 0);

    comm = std::make_shared<internode::NCCLGINBackend>();
    comm->init(unique_ids, rank, num_ranks, low_latency_mode, qps_per_rank);
    internode::barrier(comm.get());

    auto stream = turbomind::core::Context::stream().handle();

    if (num_rdma_bytes) {
        // Allocate High-Throughput RDMA buffer
        rdma_buffer_ptr = internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES, comm.get());
        // Register memory with NCCL communicators (sets up windows for RDMA)
        internode::register_memory(rdma_buffer_ptr, num_rdma_bytes, comm.get());
    }

    if (num_ll_rdma_bytes) {
        // Allocate Low-Latency RDMA buffer
        rdma_ll_buffer_ptr = internode::alloc(num_ll_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES, comm.get());
        // Register memory with NCCL communicators (sets up windows for RDMA)
        internode::register_memory(rdma_ll_buffer_ptr, num_ll_rdma_bytes, comm.get());

        // Clean buffer (mainly for low-latency mode)
        CUDA_CHECK(cudaMemsetAsync(rdma_ll_buffer_ptr, 0, num_ll_rdma_bytes, stream));

        internode_ll::set_p2p_disabled_flag(comm->is_p2p_disabled());
    }

    // Allocate and clean shrink buffer
    if (enable_shrink) {
        int num_mask_buffer_bytes = num_ranks * sizeof(int);
        int num_sync_buffer_bytes = num_ranks * sizeof(int);
        mask_buffer_ptr =
            reinterpret_cast<int*>(internode::alloc(num_mask_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES, comm.get()));
        sync_buffer_ptr =
            reinterpret_cast<int*>(internode::alloc(num_sync_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES, comm.get()));
        CUDA_CHECK(cudaMemsetAsync(mask_buffer_ptr, 0, num_mask_buffer_bytes, stream));
        CUDA_CHECK(cudaMemset(sync_buffer_ptr, 0, num_sync_buffer_bytes));
    }

    // Barrier
    internode::barrier(comm.get());
}

bool Buffer::is_available() const
{
    return available;
}

bool Buffer::is_internode_available() const
{
    return is_available() and num_ranks > NUM_MAX_NVL_PEERS;
}

int Buffer::get_num_rdma_ranks() const
{
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const
{
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const
{
    return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const
{
    return device_id;
}

void Buffer::destroy()
{
    TM_LOG_DEBUG("[NCCLEP][{}] Destroying buffer", rank);
    EP_HOST_ASSERT(not destroyed);

    // Synchronize
    auto comm_stream = turbomind::core::Context::stream().handle();

    if (num_nvl_bytes > 0 && ipc_comm) {
        turbomind::core::Context::stream().Sync();
        ipc_comm->Free(buffer_ptrs[nvl_rank]);
        ipc_comm = {};
    }

    // Free NVSHMEM
    if (is_available() && comm != nullptr) {
        turbomind::core::Context::stream().Sync();
        if (num_rdma_bytes > 0) {
            internode::free(rdma_buffer_ptr, comm.get());
        }
        if (num_ll_rdma_bytes > 0) {
            internode::free(rdma_ll_buffer_ptr, comm.get());
        }
        if (enable_shrink) {
            internode::free(mask_buffer_ptr, comm.get());
            internode::free(sync_buffer_ptr, comm.get());
        }
        internode::finalize(comm.get());
    }

    // Free workspace and MoE counter
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));

    // Free chunked mode staffs
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));

    destroyed = true;
    available = false;
}

std::tuple<Tensor, std::optional<Tensor>, Tensor, Tensor>  //
Buffer::get_dispatch_layout(const Tensor& topk_idx, int num_experts)
{

    auto num_tokens               = static_cast<int>(topk_idx.shape(0));
    auto num_topk                 = static_cast<int>(topk_idx.shape(1));
    auto num_tokens_per_rank      = Tensor_<int>{{num_ranks}, turbomind::kDEVICE};
    auto num_tokens_per_rdma_rank = std::optional<Tensor>();
    auto num_tokens_per_expert    = Tensor_<int>{{num_experts}, turbomind::kDEVICE};
    auto is_token_in_rank         = Tensor_<bool>{{num_tokens, num_ranks}, turbomind::kDEVICE};
    if (is_internode_available()) {
        num_tokens_per_rdma_rank = Buffer_<int>{num_rdma_ranks, turbomind::kDEVICE};
    }
    static_assert(sizeof(topk_idx_t) == sizeof(int64_t), "topk_idx_t must be int64_t");

    auto stream = turbomind::core::Context::stream().handle();
    layout::get_dispatch_layout(topk_idx.data<topk_idx_t>(),
                                num_tokens_per_rank.data(),
                                num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data<int>() :
                                                                       nullptr,
                                num_tokens_per_expert.data(),
                                is_token_in_rank.data_or((bool*)nullptr),  // num_tokens may be zero
                                num_tokens,
                                num_topk,
                                num_ranks,
                                num_experts,
                                stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank};
}

std::tuple<Tensor,
           std::optional<Tensor>,
           std::optional<Tensor>,
           std::optional<Tensor>,
           std::vector<int>,
           Tensor,
           Tensor,
           Tensor,
           Tensor,
           Tensor,
           Tensor>
Buffer::intranode_dispatch(const Tensor&                x,
                           const std::optional<Tensor>& x_scales,
                           const std::optional<Tensor>& topk_idx,
                           const std::optional<Tensor>& topk_weights,
                           const std::optional<Tensor>& num_tokens_per_rank,
                           const Tensor&                is_token_in_rank,
                           const std::optional<Tensor>& num_tokens_per_expert,
                           int                          cached_num_recv_tokens,
                           const std::optional<Tensor>& cached_rank_prefix_matrix,
                           const std::optional<Tensor>& cached_channel_prefix_matrix,
                           int                          expert_alignment,
                           int                          num_worst_tokens,
                           const Config&                config)
{
    bool cached_mode = cached_rank_prefix_matrix.has_value();

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
    }
    else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }
    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.dtype() == turbomind::kBool);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->dtype() == turbomind::kInt32);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->dtype() == turbomind::kInt32);
    }
    else {
        EP_HOST_ASSERT(num_tokens_per_expert->dtype() == turbomind::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rank->dtype() == turbomind::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.ndim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.shape(1) * byte_size(x.dtype())) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.ndim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.shape(0) == x.shape(0) and is_token_in_rank.shape(1) == num_ranks);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->ndim() == 2 and cached_rank_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rank_prefix_matrix->shape(0) == num_ranks
                       and cached_rank_prefix_matrix->shape(1) == num_ranks);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->ndim() == 2 and cached_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_channel_prefix_matrix->shape(0) == num_ranks
                       and cached_channel_prefix_matrix->shape(1) == num_channels);
    }
    else {
        EP_HOST_ASSERT(num_tokens_per_expert->ndim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->shape(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->shape(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
        EP_HOST_ASSERT(num_tokens_per_rank->ndim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->shape(0) == num_ranks);
    }

    auto num_tokens        = static_cast<int>(x.shape(0));
    auto hidden            = static_cast<int>(x.shape(1));
    auto num_experts       = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->shape(0));
    auto num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int               num_topk         = 0;
    const topk_idx_t* topk_idx_ptr     = nullptr;
    const float*      topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->shape(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->ndim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->ndim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->shape(0) and num_tokens == topk_weights->shape(0));
        EP_HOST_ASSERT(num_topk == topk_weights->shape(1));
        EP_HOST_ASSERT(topk_weights->dtype() == turbomind::kFloat32);
        topk_idx_ptr     = topk_idx->data_or((topk_idx_t*)nullptr);
        topk_weights_ptr = topk_weights->data_or((float*)nullptr);
    }

    // FP8 scales checks
    const float* x_scales_ptr = nullptr;
    int          num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(byte_size(x.dtype()) == 1);
        EP_HOST_ASSERT(x_scales->dtype() == turbomind::kFloat32 or x_scales->dtype() == turbomind::kInt32);
        EP_HOST_ASSERT(x_scales->ndim() == 2);
        EP_HOST_ASSERT(x_scales->shape(0) == num_tokens);
        num_scales          = x_scales->ndim() == 1 ? 1 : static_cast<int>(x_scales->shape(1));
        x_scales_ptr        = x_scales->data_or((float*)nullptr);
        scale_token_stride  = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Create handles (only return for non-cached mode)
    int              num_recv_tokens       = -1;
    auto             rank_prefix_matrix    = Tensor();
    auto             channel_prefix_matrix = Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // used to compute offsets in MoeFfnLayer
    auto moe_recv_expert_counter_ten = Tensor({num_local_experts}, turbomind::kInt32, turbomind::kDEVICE);

    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    int num_memset_int = num_channels * num_ranks * 4;
    if (cached_mode) {
        EP_HOST_ASSERT(0);
        // num_recv_tokens       = cached_num_recv_tokens;
        // rank_prefix_matrix    = cached_rank_prefix_matrix.value();
        // channel_prefix_matrix = cached_channel_prefix_matrix.value();

        // // Copy rank prefix matrix and clean flags
        // intranode::cached_notify_dispatch(rank_prefix_matrix.data_ptr<int>(),
        //                                   num_memset_int,
        //                                   buffer_ptrs_gpu,
        //                                   barrier_signal_ptrs_gpu,
        //                                   rank,
        //                                   num_ranks,
        //                                   comm_stream);
    }
    else {
        rank_prefix_matrix    = Tensor({num_ranks, num_ranks}, turbomind::kInt32, turbomind::kDEVICE);
        channel_prefix_matrix = Tensor({num_ranks, num_channels}, turbomind::kInt32, turbomind::kDEVICE);

        // Send sizes
        // Meta information:
        //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
        // NOTES: no more token dropping in this version
        *moe_recv_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes);
        intranode::notify_dispatch(num_tokens_per_rank->data<int>(),
                                   moe_recv_counter_mapped,
                                   num_ranks,
                                   num_tokens_per_expert->data<int>(),
                                   moe_recv_expert_counter_mapped,
                                   moe_recv_expert_counter_ten.data<int>(),
                                   num_experts,
                                   num_tokens,
                                   is_token_in_rank.data_or((bool*)nullptr),  // num_tokens may be zero
                                   channel_prefix_matrix.data<int>(),
                                   rank_prefix_matrix.data<int>(),
                                   num_memset_int,
                                   expert_alignment,
                                   buffer_ptrs_gpu,
                                   barrier_signal_ptrs_gpu,
                                   rank,
                                   turbomind::core::Context::stream().handle(),
                                   num_channels);

        if (num_worst_tokens > 0) {
            // No CPU sync, just allocate the worst case
            num_recv_tokens = num_worst_tokens;

            // Must be forward with top-k stuffs
            EP_HOST_ASSERT(topk_idx.has_value());
            EP_HOST_ASSERT(topk_weights.has_value());
        }
        else {
            // Synchronize total received tokens and tokens per expert
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now()
                                                                     - start_time)
                        .count()
                    > NUM_CPU_TIMEOUT_SECS)
                    throw std::runtime_error("DeepEP error: CPU recv timeout");
            }
            num_recv_tokens_per_expert_list =
                std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }

    // Allocate new tensors
    auto recv_x                     = Tensor({num_recv_tokens, hidden}, x.dtype(), turbomind::kDEVICE);
    auto recv_src_idx               = Tensor({num_recv_tokens}, turbomind::kInt32, turbomind::kDEVICE);
    auto recv_topk_idx              = std::optional<Tensor>();
    auto recv_topk_weights          = std::optional<Tensor>();
    auto recv_x_scales              = std::optional<Tensor>();
    auto recv_channel_prefix_matrix = Tensor({num_ranks, num_channels}, turbomind::kInt32, turbomind::kDEVICE);
    auto send_head                  = Tensor({num_tokens, num_ranks}, turbomind::kInt32, turbomind::kDEVICE);

    // Assign pointers
    topk_idx_t* recv_topk_idx_ptr     = nullptr;
    float*      recv_topk_weights_ptr = nullptr;
    float*      recv_x_scales_ptr     = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx         = Tensor({num_recv_tokens, num_topk}, topk_idx->dtype(), topk_idx->device());
        recv_topk_weights     = Tensor({num_recv_tokens, num_topk}, topk_weights->dtype(), topk_weights->device());
        recv_topk_idx_ptr     = recv_topk_idx->data_or((topk_idx_t*)nullptr);
        recv_topk_weights_ptr = recv_topk_weights->data_or((float*)nullptr);
    }
    if (x_scales.has_value()) {
        recv_x_scales     = x_scales->ndim() == 1 ?
                                Tensor({num_recv_tokens}, x_scales->dtype(), x_scales->device()) :
                                Tensor({num_recv_tokens, num_scales}, x_scales->dtype(), x_scales->device());
        recv_x_scales_ptr = recv_x_scales->data_or((float*)nullptr);
    }

    // Dispatch
    EP_HOST_ASSERT(
        num_ranks * num_ranks * sizeof(int) +             // Size prefix matrix
            num_channels * num_ranks * sizeof(int) +      // Channel start offset
            num_channels * num_ranks * sizeof(int) +      // Channel end offset
            num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * byte_size(recv_x.dtype())
            +                                                                                  // Data buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +  // Source index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(topk_idx_t)
            +  // Top-k index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float)
            +  // Top-k weight buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float)
                * num_scales  // FP8 scale buffer
        <= num_nvl_bytes);

    intranode::dispatch(recv_x.data_or((void*)nullptr),
                        recv_x_scales_ptr,
                        recv_src_idx.data_or((int*)nullptr),
                        recv_topk_idx_ptr,
                        recv_topk_weights_ptr,
                        recv_channel_prefix_matrix.data<int>(),
                        send_head.data_or((int*)nullptr),
                        x.data_or((void*)nullptr),
                        x_scales_ptr,
                        topk_idx_ptr,
                        topk_weights_ptr,
                        is_token_in_rank.data_or((bool*)nullptr),
                        channel_prefix_matrix.data<int>(),
                        num_tokens,
                        num_worst_tokens,
                        static_cast<int>(hidden * byte_size(recv_x.dtype()) / sizeof(int4)),
                        num_topk,
                        num_experts,
                        num_scales,
                        scale_token_stride,
                        scale_hidden_stride,
                        buffer_ptrs_gpu,
                        rank,
                        num_ranks,
                        turbomind::core::Context::stream().handle(),
                        config.num_sms,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            moe_recv_expert_counter_ten,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head};
}

std::tuple<Tensor,  //
           std::optional<Tensor>>
Buffer::intranode_combine(const Tensor&                x,
                          const std::optional<Tensor>& topk_weights,
                          const std::optional<Tensor>& bias_0,
                          const std::optional<Tensor>& bias_1,
                          const Tensor&                src_idx,
                          const Tensor&                rank_prefix_matrix,
                          const Tensor&                channel_prefix_matrix,
                          Tensor&                      send_head,
                          const Config&                config)
{
    EP_HOST_ASSERT(x.ndim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_idx.ndim() == 1 and src_idx.is_contiguous() and src_idx.dtype() == turbomind::kInt32);
    EP_HOST_ASSERT(send_head.ndim() == 2 and send_head.is_contiguous() and send_head.dtype() == turbomind::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.ndim() == 2 and rank_prefix_matrix.is_contiguous()
                   and rank_prefix_matrix.dtype() == turbomind::kInt32);
    EP_HOST_ASSERT(channel_prefix_matrix.ndim() == 2 and channel_prefix_matrix.is_contiguous()
                   and channel_prefix_matrix.dtype() == turbomind::kInt32);

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens = static_cast<int>(x.shape(0)), hidden = static_cast<int>(x.shape(1));
    auto num_recv_tokens = static_cast<int>(send_head.shape(0));
    EP_HOST_ASSERT(src_idx.shape(0) == num_tokens);
    EP_HOST_ASSERT(send_head.shape(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.shape(0) == num_ranks and rank_prefix_matrix.shape(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.shape(0) == num_ranks and channel_prefix_matrix.shape(1) == num_channels);
    EP_HOST_ASSERT((hidden * byte_size(x.dtype())) % sizeof(int4) == 0);

    int          num_topk              = 0;
    auto         recv_topk_weights     = std::optional<Tensor>();
    const float* topk_weights_ptr      = nullptr;
    float*       recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->ndim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->shape(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->dtype() == turbomind::kFloat32);
        num_topk              = static_cast<int>(topk_weights->shape(1));
        topk_weights_ptr      = topk_weights->data_or((float*)nullptr);
        recv_topk_weights     = Tensor({num_recv_tokens, num_topk}, turbomind::kFloat32, turbomind::kDEVICE);
        recv_topk_weights_ptr = recv_topk_weights->data_or((float*)nullptr);
    }

    // Launch barrier and reset queue head and tail
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
    intranode::cached_notify_combine(buffer_ptrs_gpu,
                                     send_head.data_or((int*)nullptr),
                                     num_channels,
                                     num_recv_tokens,
                                     num_channels * num_ranks * 2,
                                     barrier_signal_ptrs_gpu,
                                     rank,
                                     num_ranks,
                                     turbomind::core::Context::stream().handle());

    // Assign bias pointers
    auto  bias_opts    = std::vector<std::optional<Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.ndim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.dtype() == x.dtype());
            EP_HOST_ASSERT(bias.shape(0) == num_recv_tokens and bias.shape(1) == hidden);
            bias_ptrs[i] = bias.data_or((void*)nullptr);
        }

    // Combine data
    auto recv_x = Tensor({num_recv_tokens, hidden}, x.dtype(), turbomind::kDEVICE);
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * byte_size(x.dtype())
                       +  // Data buffer
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int)
                       +  // Source index buffer
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk
                           * sizeof(float)  // Top-k weight buffer
                   <= num_nvl_bytes);
    intranode::combine(CUDA_R_16BF,
                       recv_x.data_or((void*)nullptr),
                       recv_topk_weights_ptr,
                       x.data_or((void*)nullptr),
                       topk_weights_ptr,
                       bias_ptrs[0],
                       bias_ptrs[1],
                       src_idx.data_or((int*)nullptr),
                       rank_prefix_matrix.data<int>(),
                       channel_prefix_matrix.data<int>(),
                       send_head.data_or((int*)nullptr),
                       num_tokens,
                       num_recv_tokens,
                       hidden,
                       num_topk,
                       buffer_ptrs_gpu,
                       rank,
                       num_ranks,
                       turbomind::core::Context::stream().handle(),
                       config.num_sms,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens);

    return {recv_x, std::nullopt};
}

std::tuple<Tensor,  //
           std::optional<Tensor>,
           Tensor,
           Tensor,
           Tensor>
Buffer::low_latency_dispatch(const Tensor&                x,
                             const Tensor&                topk_idx,
                             const std::optional<Tensor>& cumulative_local_expert_recv_stats,
                             const std::optional<Tensor>& dispatch_wait_recv_cost_stats,
                             int                          num_max_dispatch_tokens_per_rank,
                             int                          num_experts,
                             bool                         use_fp8,
                             bool                         round_scale,
                             bool                         use_ue8m0)
{
    // Tensor checks
    // By default using `ptp128c` FP8 cast
    EP_HOST_ASSERT(x.ndim() == 2 and x.is_contiguous() and x.dtype() == turbomind::kBfloat16);
    EP_HOST_ASSERT(x.shape(1) % sizeof(int4) == 0 and x.shape(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.ndim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.shape(0) == topk_idx.shape(0) and x.shape(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.dtype() == turbomind::kInt64);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);

    // Diagnosis tensors
    EP_HOST_ASSERT(not cumulative_local_expert_recv_stats.has_value());
    EP_HOST_ASSERT(not dispatch_wait_recv_cost_stats.has_value());
    // if (cumulative_local_expert_recv_stats.has_value()) {
    //     EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dtype() == turbomind::kInt32);
    //     EP_HOST_ASSERT(cumulative_local_expert_recv_stats->ndim() == 1
    //                    and cumulative_local_expert_recv_stats->is_contiguous());
    //     EP_HOST_ASSERT(cumulative_local_expert_recv_stats->shape(0) == num_experts / num_ranks);
    // }
    // if (dispatch_wait_recv_cost_stats.has_value()) {
    //     EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dtype() == turbomind::kInt64);
    //     EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->ndim() == 1 and
    //     dispatch_wait_recv_cost_stats->is_contiguous()); EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->shape(0) ==
    //     num_ranks);
    // }

    auto num_tokens        = static_cast<int>(x.shape(0));
    auto hidden            = static_cast<int>(x.shape(1));
    auto num_topk          = static_cast<int>(topk_idx.shape(1));
    auto num_local_experts = num_experts / num_ranks;

    // Buffer control
    LowLatencyLayout layout(rdma_ll_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_ll_rdma_bytes);
    auto buffer      = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Allocate packed tensors
    auto packed_recv_x = Tensor({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                use_fp8 ? turbomind::kFloat8_e4m3 : x.dtype(),
                                turbomind::kDEVICE);

    auto packed_recv_src_info = Tensor(
        {num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, turbomind::kInt32, turbomind::kDEVICE);
    auto packed_recv_layout_range = Tensor({num_local_experts, num_ranks}, turbomind::kInt64, turbomind::kDEVICE);
    auto packed_recv_count        = Tensor({num_local_experts}, turbomind::kInt32, turbomind::kDEVICE);

    // Allocate column-majored scales
    auto  packed_recv_x_scales     = std::optional<Tensor>();
    void* packed_recv_x_scales_ptr = nullptr;
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0
                   and "TMA requires the number of tokens to be multiple of 4");

    if (use_fp8) {
        // TODO: support unaligned cases
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (not use_ue8m0) {
            packed_recv_x_scales =
                Tensor({num_local_experts, hidden / 128, num_ranks * num_max_dispatch_tokens_per_rank},
                       turbomind::kFloat32,
                       turbomind::kDEVICE);
        }
        else {
            EP_HOST_ASSERT(round_scale);
            packed_recv_x_scales =
                Tensor({num_local_experts, hidden / 512, num_ranks * num_max_dispatch_tokens_per_rank},
                       turbomind::kInt32,
                       turbomind::kDEVICE);
        }
        packed_recv_x_scales     = packed_recv_x_scales->transpose(1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->data_or((float*)nullptr);
    }

    // Kernel launch
    auto      next_clean_meta = next_buffer.clean_meta();
    const int phases          = LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE;
    auto      dev_comm        = comm->get_device_communicator(true);
    auto      nccl_win        = comm->get_device_nccl_window(rdma_ll_buffer_ptr);
    auto      signals_base    = comm->get_signals_base(low_latency_buffer_idx, true);

    internode_ll::dispatch(
        packed_recv_x.raw_data(),
        packed_recv_x_scales_ptr,
        packed_recv_src_info.data<int>(),
        packed_recv_layout_range.data<int64_t>(),
        packed_recv_count.data<int>(),
        mask_buffer_ptr,
        nullptr,
        nullptr,
        buffer.dispatch_rdma_recv_data_buffer,
        buffer.dispatch_rdma_recv_count_buffer,
        buffer.dispatch_rdma_send_buffer,
        reinterpret_cast<size_t>(buffer.dispatch_rdma_recv_data_buffer) - reinterpret_cast<size_t>(rdma_ll_buffer_ptr),
        reinterpret_cast<size_t>(buffer.dispatch_rdma_recv_count_buffer) - reinterpret_cast<size_t>(rdma_ll_buffer_ptr),
        reinterpret_cast<size_t>(buffer.dispatch_rdma_send_buffer) - reinterpret_cast<size_t>(rdma_ll_buffer_ptr),
        x.raw_data(),
        topk_idx.data<topk_idx_t>(),
        next_clean_meta.first,
        next_clean_meta.second,
        num_tokens,
        hidden,
        num_max_dispatch_tokens_per_rank,
        num_topk,
        num_experts,
        rank,
        num_ranks,
        use_fp8,
        round_scale,
        use_ue8m0,
        workspace,
        num_device_sms,
        nccl_win,
        dev_comm,
        signals_base,
        turbomind::core::Context::stream().handle(),
        phases);

    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range};
}

std::tuple<Tensor>  //
Buffer::low_latency_combine(const Tensor&                x,
                            const Tensor&                expert_offsets,
                            const Tensor&                topk_idx,
                            const Tensor&                topk_weights,
                            const Tensor&                src_info,
                            const Tensor&                layout_range,
                            const std::optional<Tensor>& combine_wait_recv_cost_stats,
                            int                          num_max_dispatch_tokens_per_rank,
                            int                          num_experts,
                            bool                         use_logfmt,
                            bool                         zero_copy,
                            const std::optional<Tensor>& out)
{
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    EP_HOST_ASSERT(x.ndim() == 2 and x.is_contiguous() and x.dtype() == turbomind::kBfloat16);
    EP_HOST_ASSERT(x.shape(1) % sizeof(int4) == 0 and x.shape(1) % 128 == 0);
    EP_HOST_ASSERT(expert_offsets.is_contiguous() and expert_offsets.dtype() == turbomind::kInt32);
    EP_HOST_ASSERT(expert_offsets.shape(0) == num_experts / num_ranks + 1);
    EP_HOST_ASSERT(topk_idx.ndim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.shape(0) == topk_weights.shape(0) and topk_idx.shape(1) == topk_weights.shape(1));
    EP_HOST_ASSERT(topk_idx.dtype() == turbomind::kInt64);
    EP_HOST_ASSERT(topk_weights.ndim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.shape(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.dtype() == turbomind::kFloat32);
    EP_HOST_ASSERT(src_info.ndim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.dtype() == turbomind::kInt32 /*and x.shape(0) == src_info.shape(0)*/);
    EP_HOST_ASSERT(layout_range.ndim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.dtype() == turbomind::kInt64);
    EP_HOST_ASSERT(layout_range.shape(0) == num_experts / num_ranks and layout_range.shape(1) == num_ranks);

    EP_HOST_ASSERT(not combine_wait_recv_cost_stats.has_value());
    // if (combine_wait_recv_cost_stats.has_value()) {
    //     EP_HOST_ASSERT(combine_wait_recv_cost_stats->dtype() == turbomind::kInt64);
    //     EP_HOST_ASSERT(combine_wait_recv_cost_stats->ndim() == 1 and combine_wait_recv_cost_stats->is_contiguous());
    //     EP_HOST_ASSERT(combine_wait_recv_cost_stats->shape(0) == num_ranks);
    // }

    auto hidden              = static_cast<int>(x.shape(1));
    auto num_topk            = static_cast<int>(topk_weights.shape(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.shape(0));

    // Buffer control
    LowLatencyLayout layout(rdma_ll_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_ll_rdma_bytes);
    auto buffer      = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Allocate output tensor
    Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->ndim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->shape(0) == num_combined_tokens and out->shape(1) == hidden);
        EP_HOST_ASSERT(out->dtype() == x.dtype());
        combined_x = out.value();
    }
    else {
        combined_x = Tensor({num_combined_tokens, hidden}, x.dtype(), turbomind::kDEVICE);
    }

    // Kernel launch
    auto      next_clean_meta = next_buffer.clean_meta();
    const int phases          = LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE;
    auto      dev_comm        = comm->get_device_communicator(true);
    auto      nccl_win        = comm->get_device_nccl_window(rdma_ll_buffer_ptr);
    auto      signals_base    = comm->get_signals_base(low_latency_buffer_idx, true);

    internode_ll::combine(
        combined_x.data_or((void*)nullptr),
        buffer.combine_rdma_recv_data_buffer,
        buffer.combine_rdma_recv_flag_buffer,
        buffer.combine_rdma_send_buffer,
        reinterpret_cast<size_t>(buffer.combine_rdma_recv_data_buffer) - reinterpret_cast<size_t>(rdma_ll_buffer_ptr),
        reinterpret_cast<size_t>(buffer.combine_rdma_recv_flag_buffer) - reinterpret_cast<size_t>(rdma_ll_buffer_ptr),
        reinterpret_cast<size_t>(buffer.combine_rdma_send_buffer) - reinterpret_cast<size_t>(rdma_ll_buffer_ptr),
        x.data_or((void*)nullptr),
        expert_offsets.data<int>(),
        topk_idx.data_or((topk_idx_t*)nullptr),
        topk_weights.data_or((float*)nullptr),
        src_info.data<int>(),
        layout_range.data<int64_t>(),
        mask_buffer_ptr,
        nullptr,
        next_clean_meta.first,
        next_clean_meta.second,
        num_combined_tokens,
        hidden,
        num_max_dispatch_tokens_per_rank,
        num_topk,
        num_experts,
        rank,
        num_ranks,
        use_logfmt,
        workspace,
        num_device_sms,
        nccl_win,
        dev_comm,
        signals_base,
        turbomind::core::Context::stream().handle(),
        phases,
        zero_copy);

    return {combined_x};
}

std::tuple<Tensor,
           std::optional<Tensor>,
           std::optional<Tensor>,
           std::optional<Tensor>,
           std::vector<int>,
           Tensor,
           Tensor,
           Tensor,
           std::optional<Tensor>,
           Tensor,
           std::optional<Tensor>,
           Tensor,
           std::optional<Tensor>,
           std::optional<Tensor>,
           std::optional<Tensor>>
Buffer::internode_dispatch(const Tensor&                x,
                           const std::optional<Tensor>& x_scales,
                           const std::optional<Tensor>& topk_idx,
                           const std::optional<Tensor>& topk_weights,
                           const std::optional<Tensor>& num_tokens_per_rank,
                           const std::optional<Tensor>& num_tokens_per_rdma_rank,
                           const Tensor&                is_token_in_rank,
                           const std::optional<Tensor>& num_tokens_per_expert,
                           int                          cached_num_recv_tokens,
                           int                          cached_num_rdma_recv_tokens,
                           const std::optional<Tensor>& cached_rdma_channel_prefix_matrix,
                           const std::optional<Tensor>& cached_recv_rdma_rank_prefix_sum,
                           const std::optional<Tensor>& cached_gbl_channel_prefix_matrix,
                           const std::optional<Tensor>& cached_recv_gbl_rank_prefix_sum,
                           int                          expert_alignment,
                           int                          num_worst_tokens,
                           const Config&                config)
{

    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
    }
    else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dtype() == turbomind::kInt32);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dtype() == turbomind::kInt32);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dtype() == turbomind::kInt32);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dtype() == turbomind::kInt32);
    }
    else {
        EP_HOST_ASSERT(num_tokens_per_rank->dtype() == turbomind::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->dtype() == turbomind::kInt32);
        EP_HOST_ASSERT(num_tokens_per_expert->dtype() == turbomind::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.ndim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.shape(1) * byte_size(x.dtype())) % sizeof(int4) == 0);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->ndim() == 2
                       and cached_rdma_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->shape(0) == num_rdma_ranks
                       and cached_rdma_channel_prefix_matrix->shape(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->ndim() == 1
                       and cached_recv_rdma_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->shape(0) == num_rdma_ranks);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->ndim() == 2
                       and cached_gbl_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->shape(0) == num_ranks
                       and cached_gbl_channel_prefix_matrix->shape(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->ndim() == 1
                       and cached_recv_gbl_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->shape(0) == num_ranks);
    }
    else {
        EP_HOST_ASSERT(num_tokens_per_rank->ndim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->ndim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->ndim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->shape(0) == num_ranks);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->shape(0) == num_rdma_ranks);
        EP_HOST_ASSERT(num_tokens_per_expert->shape(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->shape(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }

    auto num_tokens        = static_cast<int>(x.shape(0));
    auto hidden            = static_cast<int>(x.shape(1));
    auto hidden_int4       = static_cast<int>(x.shape(1) * byte_size(x.dtype()) / sizeof(int4));
    auto num_experts       = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->shape(0));
    auto num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int               num_topk         = 0;
    const topk_idx_t* topk_idx_ptr     = nullptr;
    const float*      topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->shape(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->ndim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->ndim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->shape(0) and num_tokens == topk_weights->shape(0));
        EP_HOST_ASSERT(num_topk == topk_weights->shape(1));
        EP_HOST_ASSERT(topk_weights->dtype() == turbomind::kFloat32);
        topk_idx_ptr     = topk_idx->data_or((topk_idx_t*)nullptr);
        topk_weights_ptr = topk_weights->data_or((float*)nullptr);
    }

    // FP8 scales checks
    const float* x_scales_ptr        = nullptr;
    int          num_scales          = 0;
    int          scale_token_stride  = 0;
    int          scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(byte_size(x.dtype()) == 1);
        EP_HOST_ASSERT(x_scales->dtype() == turbomind::kFloat32 or x_scales->dtype() == turbomind::kInt32);
        EP_HOST_ASSERT(x_scales->ndim() == 2);
        EP_HOST_ASSERT(x_scales->shape(0) == num_tokens);
        num_scales          = x_scales->ndim() == 1 ? 1 : static_cast<int>(x_scales->shape(1));
        x_scales_ptr        = x_scales->data_or((float*)nullptr);
        scale_token_stride  = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Create handles (only return for non-cached mode)
    int              num_recv_tokens = -1, num_rdma_recv_tokens = -1;
    auto             rdma_channel_prefix_matrix = Tensor();
    auto             recv_rdma_rank_prefix_sum  = Tensor();
    auto             gbl_channel_prefix_matrix  = Tensor();
    auto             recv_gbl_rank_prefix_sum   = Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // used to compute offsets in MoeFfnLayer
    auto moe_recv_expert_counter_ten = Tensor({num_local_experts}, turbomind::kInt32, turbomind::kDEVICE);

    auto dev_comm     = comm->get_device_communicator(false);
    auto nccl_win     = comm->get_device_nccl_window(rdma_buffer_ptr);
    auto signals_base = comm->get_signals_base(0, false);
    auto gin_base_ptr = rdma_buffer_ptr;

    if (cached_mode) {
        EP_HOST_ASSERT(not cached_mode);
    }
    else {
        rdma_channel_prefix_matrix = Tensor({num_rdma_ranks, num_channels}, turbomind::kInt32, turbomind::kDEVICE);
        recv_rdma_rank_prefix_sum  = Tensor({num_rdma_ranks}, turbomind::kInt32, turbomind::kDEVICE);
        gbl_channel_prefix_matrix  = Tensor({num_ranks, num_channels}, turbomind::kInt32, turbomind::kDEVICE);
        recv_gbl_rank_prefix_sum   = Tensor({num_ranks}, turbomind::kInt32, turbomind::kDEVICE);

        // Send sizes
        *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        internode::notify_dispatch(num_tokens_per_rank->data<int>(),
                                   moe_recv_counter_mapped,
                                   num_ranks,
                                   num_tokens_per_rdma_rank->data<int>(),
                                   moe_recv_rdma_counter_mapped,
                                   num_tokens_per_expert->data<int>(),
                                   moe_recv_expert_counter_mapped,
                                   moe_recv_expert_counter_ten.data<int>(),
                                   num_experts,
                                   is_token_in_rank.data_or((bool*)nullptr),
                                   num_tokens,
                                   num_worst_tokens,
                                   num_channels,
                                   hidden_int4,
                                   num_scales,
                                   num_topk,
                                   expert_alignment,
                                   rdma_channel_prefix_matrix.data<int>(),
                                   recv_rdma_rank_prefix_sum.data<int>(),
                                   gbl_channel_prefix_matrix.data<int>(),
                                   recv_gbl_rank_prefix_sum.data<int>(),
                                   rdma_buffer_ptr,
                                   config.num_max_rdma_chunked_recv_tokens,
                                   buffer_ptrs_gpu,
                                   config.num_max_nvl_chunked_recv_tokens,
                                   barrier_signal_ptrs_gpu,
                                   rank,
                                   turbomind::core::Context::stream().handle(),
                                   config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                   num_nvl_bytes,
                                   low_latency_mode,
                                   gin_base_ptr,
                                   dev_comm,
                                   nccl_win,
                                   signals_base);

        // Synchronize total received tokens and tokens per expert
        if (num_worst_tokens > 0) {
            num_recv_tokens      = num_worst_tokens;
            num_rdma_recv_tokens = num_worst_tokens;
        }
        else {
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens      = static_cast<int>(*moe_recv_counter);
                num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now()
                                                                     - start_time)
                        .count()
                    > NUM_CPU_TIMEOUT_SECS) {
                    printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n",
                           rank,
                           num_recv_tokens,
                           num_rdma_recv_tokens);
                    for (int i = 0; i < num_local_experts; ++i)
                        printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                    throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
                }
            }
            num_recv_tokens_per_expert_list =
                std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }

    // Allocate new tensors
    auto recv_x                          = Tensor({num_recv_tokens, hidden}, x.dtype(), turbomind::kDEVICE);
    auto recv_topk_idx                   = std::optional<Tensor>();
    auto recv_topk_weights               = std::optional<Tensor>();
    auto recv_x_scales                   = std::optional<Tensor>();
    auto recv_src_meta                   = std::optional<Tensor>();
    auto recv_rdma_channel_prefix_matrix = std::optional<Tensor>();
    auto recv_gbl_channel_prefix_matrix  = std::optional<Tensor>();
    auto send_rdma_head                  = std::optional<Tensor>();
    auto send_nvl_head                   = std::optional<Tensor>();
    if (not cached_mode) {
        recv_src_meta =
            Tensor({num_recv_tokens, internode::get_source_meta_bytes()}, turbomind::kUint8, turbomind::kDEVICE);
        recv_rdma_channel_prefix_matrix = Tensor({num_rdma_ranks, num_channels}, turbomind::kInt32, turbomind::kDEVICE);
        recv_gbl_channel_prefix_matrix  = Tensor({num_ranks, num_channels}, turbomind::kInt32, turbomind::kDEVICE);
        send_rdma_head                  = Tensor({num_tokens, num_rdma_ranks}, turbomind::kInt32, turbomind::kDEVICE);
        send_nvl_head = Tensor({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, turbomind::kInt32, turbomind::kDEVICE);
    }

    // Assign pointers
    topk_idx_t* recv_topk_idx_ptr     = nullptr;
    float*      recv_topk_weights_ptr = nullptr;
    float*      recv_x_scales_ptr     = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx         = Tensor({num_recv_tokens, num_topk}, topk_idx->dtype(), turbomind::kDEVICE);
        recv_topk_weights     = Tensor({num_recv_tokens, num_topk}, topk_weights->dtype(), turbomind::kDEVICE);
        recv_topk_idx_ptr     = recv_topk_idx->data_or((topk_idx_t*)nullptr);
        recv_topk_weights_ptr = recv_topk_weights->data_or((float*)nullptr);
    }
    if (x_scales.has_value()) {
        recv_x_scales     = x_scales->ndim() == 1 ?
                                Tensor({num_recv_tokens}, x_scales->dtype(), turbomind::kDEVICE) :
                                Tensor({num_recv_tokens, num_scales}, x_scales->dtype(), turbomind::kDEVICE);
        recv_x_scales_ptr = recv_x_scales->data_or((float*)nullptr);
    }

    // Launch data dispatch
    // NOTES: the buffer size checks are moved into the `.cu` file
    internode::dispatch(recv_x.data_or((void*)nullptr),
                        recv_x_scales_ptr,
                        recv_topk_idx_ptr,
                        recv_topk_weights_ptr,
                        cached_mode ? nullptr : recv_src_meta->data_or((void*)nullptr),
                        x.data_or((void*)nullptr),
                        x_scales_ptr,
                        topk_idx_ptr,
                        topk_weights_ptr,
                        cached_mode ? nullptr : send_rdma_head->data_or((int*)nullptr),
                        cached_mode ? nullptr : send_nvl_head->data_or((int*)nullptr),
                        cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data<int>(),
                        cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data<int>(),
                        rdma_channel_prefix_matrix.data<int>(),
                        recv_rdma_rank_prefix_sum.data<int>(),
                        gbl_channel_prefix_matrix.data<int>(),
                        recv_gbl_rank_prefix_sum.data<int>(),
                        is_token_in_rank.data_or((bool*)nullptr),
                        num_tokens,
                        num_worst_tokens,
                        hidden_int4,
                        num_scales,
                        num_topk,
                        num_experts,
                        scale_token_stride,
                        scale_hidden_stride,
                        rdma_buffer_ptr,
                        config.num_max_rdma_chunked_send_tokens,
                        config.num_max_rdma_chunked_recv_tokens,
                        buffer_ptrs_gpu,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens,
                        rank,
                        num_ranks,
                        cached_mode,
                        turbomind::core::Context::stream().handle(),
                        num_channels,
                        low_latency_mode,
                        gin_base_ptr,
                        dev_comm,
                        nccl_win,
                        signals_base);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            moe_recv_expert_counter_ten,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head};
}

std::tuple<Tensor, std::optional<Tensor>>  //
Buffer::internode_combine(const Tensor&                x,
                          const std::optional<Tensor>& topk_weights,
                          const std::optional<Tensor>& bias_0,
                          const std::optional<Tensor>& bias_1,
                          const Tensor&                src_meta,
                          const Tensor&                is_combined_token_in_rank,
                          const Tensor&                rdma_channel_prefix_matrix,
                          const Tensor&                rdma_rank_prefix_sum,
                          const Tensor&                gbl_channel_prefix_matrix,
                          Tensor&                      combined_rdma_head,
                          Tensor&                      combined_nvl_head,
                          const Config&                config)
{
    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.ndim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_meta.ndim() == 2 and src_meta.is_contiguous() and src_meta.dtype() == turbomind::kUint8);
    EP_HOST_ASSERT(is_combined_token_in_rank.ndim() == 2 and is_combined_token_in_rank.is_contiguous()
                   and is_combined_token_in_rank.dtype() == turbomind::kBool);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.ndim() == 2 and rdma_channel_prefix_matrix.is_contiguous()
                   and rdma_channel_prefix_matrix.dtype() == turbomind::kInt32);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.ndim() == 1 and rdma_rank_prefix_sum.is_contiguous()
                   and rdma_rank_prefix_sum.dtype() == turbomind::kInt32);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.ndim() == 2 and gbl_channel_prefix_matrix.is_contiguous()
                   and gbl_channel_prefix_matrix.dtype() == turbomind::kInt32);
    EP_HOST_ASSERT(combined_rdma_head.ndim() == 2 and combined_rdma_head.is_contiguous()
                   and combined_rdma_head.dtype() == turbomind::kInt32);
    EP_HOST_ASSERT(combined_nvl_head.ndim() == 2 and combined_nvl_head.is_contiguous()
                   and combined_nvl_head.dtype() == turbomind::kInt32);

    auto num_tokens          = static_cast<int>(x.shape(0));
    auto hidden              = static_cast<int>(x.shape(1));
    auto hidden_int4         = static_cast<int>(x.shape(1) * byte_size(x.dtype()) / sizeof(int4));
    auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.shape(0));
    EP_HOST_ASSERT((hidden * byte_size(x.dtype())) % sizeof(int4) == 0);
    EP_HOST_ASSERT(src_meta.shape(1) == internode::get_source_meta_bytes());
    EP_HOST_ASSERT(is_combined_token_in_rank.shape(1) == num_ranks);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.shape(0) == num_rdma_ranks
                   and rdma_channel_prefix_matrix.shape(1) == num_channels);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.shape(0) == num_rdma_ranks);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.shape(0) == num_ranks
                   and gbl_channel_prefix_matrix.shape(1) == num_channels);
    EP_HOST_ASSERT(combined_rdma_head.ndim() == 2 and combined_rdma_head.shape(0) == num_combined_tokens
                   and combined_rdma_head.shape(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_nvl_head.ndim() == 2 and combined_nvl_head.shape(1) == NUM_MAX_NVL_PEERS);

    // Top-k checks
    int          num_topk                  = 0;
    auto         combined_topk_weights     = std::optional<Tensor>();
    const float* topk_weights_ptr          = nullptr;
    float*       combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->ndim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->shape(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->dtype() == turbomind::kFloat32);
        num_topk                  = static_cast<int>(topk_weights->shape(1));
        topk_weights_ptr          = topk_weights->data_or((float*)nullptr);
        combined_topk_weights     = Tensor({num_combined_tokens, num_topk}, turbomind::kFloat32, turbomind::kDEVICE);
        combined_topk_weights_ptr = combined_topk_weights->data_or((float*)nullptr);
    }

    // Extra check for avoid-dead-lock design
    EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

    auto gin_base_ptr = rdma_buffer_ptr;
    auto dev_comm     = comm->get_device_communicator(false);
    auto nccl_win     = comm->get_device_nccl_window(rdma_buffer_ptr);
    auto signals_base = comm->get_signals_base(0, false);

    // Launch barrier and reset queue head and tail
    internode::cached_notify(hidden_int4,
                             0,
                             0,
                             num_topk,
                             num_ranks,
                             num_channels,
                             num_combined_tokens,
                             combined_rdma_head.data_or((int*)nullptr),
                             rdma_channel_prefix_matrix.data<int>(),
                             rdma_rank_prefix_sum.data<int>(),
                             combined_nvl_head.data_or((int*)nullptr),
                             rdma_buffer_ptr,
                             config.num_max_rdma_chunked_recv_tokens,
                             buffer_ptrs_gpu,
                             config.num_max_nvl_chunked_recv_tokens,
                             barrier_signal_ptrs_gpu,
                             rank,
                             turbomind::core::Context::stream().handle(),
                             config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                             num_nvl_bytes,
                             false,
                             low_latency_mode,
                             gin_base_ptr,
                             dev_comm,
                             nccl_win,
                             signals_base);

    // Assign bias pointers
    auto  bias_opts    = std::vector<std::optional<Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.ndim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.dtype() == x.dtype());
            EP_HOST_ASSERT(bias.shape(0) == num_combined_tokens and bias.shape(1) == hidden);
            bias_ptrs[i] = bias.data_or((void*)nullptr);
        }

    // Launch data combine
    auto combined_x = Tensor({num_combined_tokens, hidden}, x.dtype(), turbomind::kDEVICE);
    internode::combine(CUDA_R_16BF,
                       combined_x.data_or((void*)nullptr),
                       combined_topk_weights_ptr,
                       is_combined_token_in_rank.data_or((bool*)nullptr),
                       x.data_or((void*)nullptr),
                       topk_weights_ptr,
                       bias_ptrs[0],
                       bias_ptrs[1],
                       combined_rdma_head.data_or((int*)nullptr),
                       combined_nvl_head.data_or((int*)nullptr),
                       src_meta.data_or((void*)nullptr),
                       rdma_channel_prefix_matrix.data<int>(),
                       rdma_rank_prefix_sum.data<int>(),
                       gbl_channel_prefix_matrix.data<int>(),
                       num_tokens,
                       num_combined_tokens,
                       hidden,
                       num_topk,
                       rdma_buffer_ptr,
                       config.num_max_rdma_chunked_send_tokens,
                       config.num_max_rdma_chunked_recv_tokens,
                       buffer_ptrs_gpu,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens,
                       rank,
                       num_ranks,
                       turbomind::core::Context::stream().handle(),
                       num_channels,
                       low_latency_mode,
                       gin_base_ptr,
                       dev_comm,
                       nccl_win,
                       signals_base);

    return {combined_x, combined_topk_weights};
}

Config Buffer::get_dispatch_config()
{
    static std::unordered_map<int, Config> config_map = {
        {2, Config(num_sms, 24, 256, 6, 128)},
        {4, Config(num_sms, 6, 256, 6, 128)},
        {8, Config(num_sms, 6, 256, 6, 128)},
        {16, Config(num_sms, 36, 288, 20, 128)},
        {24, Config(num_sms, 32, 288, 8, 128)},
        {32, Config(num_sms, 32, 288, 8, 128)},
        {48, Config(num_sms, 32, 288, 8, 128)},
        {64, Config(num_sms, 32, 288, 8, 128)},
        {96, Config(num_sms, 20, 480, 12, 128)},
        {128, Config(num_sms, 20, 560, 12, 128)},
        {144, Config(num_sms, 32, 720, 12, 128)},
        {160, Config(num_sms, 28, 720, 12, 128)},
    };
    const auto it = config_map.find(num_ranks);
    TM_CHECK(it != config_map.end());
    return it->second;
}

Config Buffer::get_combine_config()
{
    static std::unordered_map<int, Config> config_map = {
        {2, Config(num_sms, 10, 256, 6, 128)},
        {4, Config(num_sms, 9, 256, 6, 128)},
        {8, Config(num_sms, 4, 256, 6, 128)},
        {16, Config(num_sms, 4, 288, 12, 128)},
        {24, Config(num_sms, 1, 288, 8, 128)},
        {32, Config(num_sms, 1, 288, 8, 128)},
        {48, Config(num_sms, 1, 288, 8, 128)},
        {64, Config(num_sms, 1, 288, 8, 128)},
        {96, Config(num_sms, 1, 480, 8, 128)},
        {128, Config(num_sms, 1, 560, 8, 128)},
        {144, Config(num_sms, 2, 720, 8, 128)},
        {160, Config(num_sms, 2, 720, 8, 128)},
    };
    const auto it = config_map.find(num_ranks);
    TM_CHECK(it != config_map.end());
    return it->second;
}

};  // namespace deep_ep

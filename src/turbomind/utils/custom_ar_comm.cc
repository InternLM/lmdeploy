/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "custom_ar_comm.h"

namespace fastertransformer {

template<typename T>
CustomAllReduceComm<T>::CustomAllReduceComm(size_t rank_size, size_t rank): rank_size_(rank_size), rank_(rank)
{
    param_.barrier_flag = 0;
    // NOTE: assume All Reduce happens within the node (DGX A100)
    param_.rank       = rank_;
    param_.local_rank = rank_;
    param_.node_id    = 0;
}

template<typename T>
CustomAllReduceComm<T>::~CustomAllReduceComm()
{
    cudaPointerAttributes comm_buffer_attributes, barrier_attributes;
    check_cuda_error(cudaPointerGetAttributes(&comm_buffer_attributes, param_.peer_comm_buffer_ptrs[rank_]));
    check_cuda_error(cudaPointerGetAttributes(&barrier_attributes, param_.peer_barrier_ptrs[rank_]));
    if (comm_buffer_attributes.type == 2) {
        check_cuda_error(cudaFree(param_.peer_comm_buffer_ptrs[rank_]));
    }
    if (barrier_attributes.type == 2) {
        check_cuda_error(cudaFree(param_.peer_barrier_ptrs[rank_]));
    }
}

template<typename T>
void CustomAllReduceComm<T>::customAllReduce(size_t elts, cudaStream_t stream)
{
    param_.elts_total   = elts;
    param_.barrier_flag = FLAG(param_.barrier_flag + 1);

    invokeOneOrTwoShotAllReduceKernel<T>(param_, stream);

    // swap back
    output_tensor_->at(0).data = (const void*)tmp_tensor_data_;
}

template<typename T>
void CustomAllReduceComm<T>::allocateAndExchangePeerAccessPointer(
    std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms)
{
    assert(custom_all_reduce_comms->size() == rank_size_);
    assert(rank_ == 0);
    // Enable Peer to Peer Access
    enableP2P(rank_size_);
    for (size_t i = 0; i < rank_size_; i++) {
        check_cuda_error(cudaSetDevice(i));
        check_cuda_error(cudaMalloc(&(param_.peer_comm_buffer_ptrs[i]), CUSTOM_AR_SIZE_THRESHOLD));
        check_cuda_error(
            cudaMalloc(&(param_.peer_barrier_ptrs[i]), rank_size_ * (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t)));
        check_cuda_error(
            cudaMemset(param_.peer_barrier_ptrs[i], 0, rank_size_ * (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t)));
        T*        current_peer_comm_buffer_ptr = param_.peer_comm_buffer_ptrs[i];
        uint32_t* current_peer_barrier_ptr     = param_.peer_barrier_ptrs[i];
        // Assume current comm allocates device memory on all ranks (rank_ == 0)
        for (size_t j = 1; j < rank_size_; j++) {
            static_cast<CustomAllReduceComm<T>*>(custom_all_reduce_comms->at(j).get())
                ->param_.peer_comm_buffer_ptrs[i] = current_peer_comm_buffer_ptr;
            static_cast<CustomAllReduceComm<T>*>(custom_all_reduce_comms->at(j).get())->param_.peer_barrier_ptrs[i] =
                current_peer_barrier_ptr;
        }
    }

    // Set default local_output_buffer_ptr to local peer_comm_buffer_ptrs
    for (size_t i = 0; i < rank_size_; i++) {
        static_cast<CustomAllReduceComm<T>*>(custom_all_reduce_comms->at(i).get())->param_.local_output_buffer_ptr =
            static_cast<CustomAllReduceComm<T>*>(custom_all_reduce_comms->at(i).get())->param_.peer_comm_buffer_ptrs[i];
    }
}

template<typename T>
void CustomAllReduceComm<T>::enableP2P(int ngpus)
{
    int peer_access_available = 0;
    for (int i = 0; i < ngpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < ngpus; j++) {
            if (i == j) {
                continue;
            }
            cudaDeviceCanAccessPeer(&peer_access_available, i, j);
            // Custom AR Kernels need DGX A100 NVSWITCH connections
            assert(peer_access_available);
            cudaDeviceEnablePeerAccess(j, 0);
        }
    }
}

template<typename T>
bool CustomAllReduceComm<T>::swapInternalBuffer(std::vector<Tensor>* tensor_buffer, size_t elts)
{
    // Check if all reduce elts meet the requirement of custom kernels
    // If meet, then swap the local comm buffer ptr with output tensor data pointer (avoid additional
    // memory movement)
    if (rank_size_ > 1 && elts * sizeof(T) <= CUSTOM_AR_SIZE_THRESHOLD) {
        tmp_tensor_data_               = (T*)(tensor_buffer->at(0).data);
        output_tensor_                 = tensor_buffer;
        tensor_buffer->at(0).data      = param_.peer_comm_buffer_ptrs[rank_];
        param_.local_output_buffer_ptr = tmp_tensor_data_;
        return true;
    }
    return false;
}

template<typename T>
void initCustomAllReduceComm(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms,
                             int                                               enable_custom_all_reduce,
                             size_t                                            rank_size)
{
    if (enable_custom_all_reduce == 0) {
        // don't use custom all reduce kernels, fall back to NCCL
        for (size_t i = 0; i < rank_size; i++) {
            custom_all_reduce_comms->push_back(nullptr);
        }
        return;
    }

    if (rank_size != RANKS_PER_NODE) {
#ifdef BUILD_MULTI_GPU
        if (rank_size > 1) {
            FT_LOG_WARNING("Custom All Reduce only supports 8 Ranks currently. Using NCCL as Comm.");
        }
#else
        FT_CHECK_WITH_INFO(rank_size == 1,
                           fmtstr("Custom All Reduce only supports 8 Ranks currently, got rank_size %ld. FT needs "
                                  "the NCCL library to communicate among devices but has built without NCCL. "
                                  "Please use the flag -DBUILD_MULTI_GPU=ON when compiling.",
                                  rank_size));
#endif
        for (size_t i = 0; i < rank_size; i++) {
            custom_all_reduce_comms->push_back(nullptr);
        }
        return;
    }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    for (size_t i = 0; i < rank_size; i++) {
        custom_all_reduce_comms->push_back(std::make_shared<CustomAllReduceComm<T>>(rank_size, i));
    }
    custom_all_reduce_comms->at(0)->allocateAndExchangePeerAccessPointer(custom_all_reduce_comms);
#else
    FT_LOG_WARNING("Custom All Reduce is not supported before CUDA 11.2. Using NCCL as Comm.");
    for (size_t i = 0; i < rank_size; i++) {
        custom_all_reduce_comms->push_back(nullptr);
    }
#endif
}

// Template instantiation
template class CustomAllReduceComm<uint16_t>;
#ifdef ENABLE_BF16
template class CustomAllReduceComm<__nv_bfloat16>;
#endif
template class CustomAllReduceComm<uint32_t>;
template void
initCustomAllReduceComm<uint16_t>(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms,
                                  int                                               enable_custom_all_reduce,
                                  size_t                                            rank_size);
#ifdef ENABLE_BF16
template void
initCustomAllReduceComm<__nv_bfloat16>(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms,
                                       int                                               enable_custom_all_reduce,
                                       size_t                                            rank_size);
#endif
template void
initCustomAllReduceComm<uint32_t>(std::vector<std::shared_ptr<AbstractCustomComm>>* custom_all_reduce_comms,
                                  int                                               enable_custom_all_reduce,
                                  size_t                                            rank_size);

}  // namespace fastertransformer

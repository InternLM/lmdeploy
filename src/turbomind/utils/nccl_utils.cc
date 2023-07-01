/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/turbomind/utils/nccl_utils.h"
#include <atomic>

namespace turbomind {

#ifdef BUILD_MULTI_GPU
template<typename T>
ncclDataType_t getNcclDataType()
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    ncclDataType_t nccl_data_type;
    if (std::is_same<T, float>::value) {
        nccl_data_type = ncclFloat;
    }
    else if (std::is_same<T, half>::value) {
        nccl_data_type = ncclHalf;
    }
#if defined(ENABLE_BF16) && defined(ENABLE_BF16_NCCL)
    else if (std::is_same<T, __nv_bfloat16>::value) {
        nccl_data_type = ncclBfloat16;
    }
#endif
    else if (std::is_same<T, int>::value) {
        nccl_data_type = ncclInt;
    }
    else if (std::is_same<T, char>::value) {
        nccl_data_type = ncclChar;
    }
    else if (std::is_same<T, bool>::value) {
        nccl_data_type = ncclInt8;
    }
    else {
        printf("[ERROR] NCCL only support float, half, bfloat16, int, char, and bool. \n");
        exit(-1);
    }
    return nccl_data_type;
}
#endif

template<typename T>
void ftNcclAllReduceSum(const T* send_buf, T* recv_buf, const int data_size, NcclParam nccl_param, cudaStream_t stream)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    ncclDataType_t nccl_data_type = getNcclDataType<T>();
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllReduce(
        (const void*)send_buf, (void*)recv_buf, data_size, nccl_data_type, ncclSum, nccl_param.nccl_comm_, stream));
    NCCLCHECK(ncclGroupEnd());
#endif
    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void ftNcclAllGather(
    const T* send_buf, T* recv_buf, const int data_size, const int rank, NcclParam nccl_param, cudaStream_t stream)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    ncclDataType_t nccl_data_type = getNcclDataType<T>();
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(
        ncclAllGather(send_buf + rank * data_size, recv_buf, data_size, nccl_data_type, nccl_param.nccl_comm_, stream));
    NCCLCHECK(ncclGroupEnd());
#endif
    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void ftNcclSend(const T* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    ncclDataType_t nccl_data_type = getNcclDataType<T>();
    NCCLCHECK(ncclSend(send_buf, data_size, nccl_data_type, peer, nccl_param.nccl_comm_, stream));
#endif
    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template void
ftNcclSend(const float* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclSend(const half* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
#ifdef ENABLE_BF16
template void ftNcclSend(
    const __nv_bfloat16* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
#endif
template void
ftNcclSend(const int* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclSend(const bool* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclSend(const char* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclRecv(T* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    ncclDataType_t nccl_data_type = getNcclDataType<T>();
    NCCLCHECK(ncclRecv(recv_buf, data_size, nccl_data_type, peer, nccl_param.nccl_comm_, stream));
#endif
    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template void
ftNcclRecv(float* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclRecv(half* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
ftNcclRecv(__nv_bfloat16* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
#endif
template void ftNcclRecv(int* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclRecv(bool* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclRecv(char* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclBroadCast(T* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    ncclDataType_t nccl_data_type = getNcclDataType<T>();
    NCCLCHECK(ncclBcast(buff, data_size, nccl_data_type, root, nccl_param.nccl_comm_, stream));
#endif
    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template void
ftNcclBroadCast(char* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclBroadCast(bool* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclBroadCast(int* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclBroadCast(float* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclBroadCast(half* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
ftNcclBroadCast(__nv_bfloat16* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);
#endif

template void ftNcclAllReduceSum(
    const float* send_buf, float* recv_buf, const int data_size, NcclParam nccl_param, cudaStream_t stream);

template void ftNcclAllReduceSum(
    const half* send_buf, half* recv_buf, const int data_size, NcclParam nccl_param, cudaStream_t stream);

template void ftNcclAllReduceSum(
    const int32_t* send_buf, int32_t* recv_buf, const int data_size, NcclParam nccl_param, cudaStream_t stream);

#ifdef ENABLE_BF16
template void ftNcclAllReduceSum(const __nv_bfloat16* send_buf,
                                 __nv_bfloat16*       recv_buf,
                                 const int            data_size,
                                 NcclParam            nccl_param,
                                 cudaStream_t         stream);
#endif

template void ftNcclAllGather(const float* send_buf,
                              float*       recv_buf,
                              const int    data_size,
                              const int    rank,
                              NcclParam    nccl_param,
                              cudaStream_t stream);

template void ftNcclAllGather(const half*  send_buf,
                              half*        recv_buf,
                              const int    data_size,
                              const int    rank,
                              NcclParam    nccl_param,
                              cudaStream_t stream);

#ifdef ENABLE_BF16
template void ftNcclAllGather(const __nv_bfloat16* send_buf,
                              __nv_bfloat16*       recv_buf,
                              const int            data_size,
                              const int            rank,
                              NcclParam            nccl_param,
                              cudaStream_t         stream);
#endif

void ftNcclGroupStart()
{
#ifdef BUILD_MULTI_GPU
    NCCLCHECK(ncclGroupStart());
#endif
}

void ftNcclGroupEnd()
{
#ifdef BUILD_MULTI_GPU
    NCCLCHECK(ncclGroupEnd());
#endif
}

void ftNcclStreamSynchronize(NcclParam tensor_para, NcclParam pipeline_para, cudaStream_t stream)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    cudaError_t  cudaErr;
    ncclResult_t tensor_ncclErr = ncclSuccess, tensor_ncclAsyncErr = ncclSuccess, pipeline_ncclErr = ncclSuccess,
                 pipeline_ncclAsyncErr = ncclSuccess;
    ncclComm_t tensor_comm             = tensor_para.nccl_comm_;
    ncclComm_t pipeline_comm           = pipeline_para.nccl_comm_;
    if (tensor_para.world_size_ == 1 && pipeline_para.world_size_ == 1) {
        check_cuda_error(cudaStreamSynchronize(stream));
        return;
    }
    while (1) {
        cudaErr = cudaStreamQuery(stream);
        if (cudaErr == cudaSuccess) {
            TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
            return;
        }

        if (cudaErr != cudaErrorNotReady) {
            std::string error_msg = "CUDA Error : cudaStreamQuery returned " + std::to_string(cudaErr);
            throw std::runtime_error(error_msg);
        }
        if (tensor_para.world_size_ > 1) {
            tensor_ncclErr = ncclCommGetAsyncError(tensor_comm, &tensor_ncclAsyncErr);
        }
        if (pipeline_para.world_size_ > 1) {
            pipeline_ncclErr = ncclCommGetAsyncError(pipeline_comm, &pipeline_ncclAsyncErr);
        }

        if (tensor_ncclErr != ncclSuccess || pipeline_ncclErr != ncclSuccess) {
            std::string error_msg = "NCCL Error : ncclCommGetAsyncError returned " + std::to_string(tensor_ncclErr)
                                    + " (tensor_para) " + std::to_string(pipeline_ncclErr) + " (pipeline_para)";
            throw std::runtime_error(error_msg);
        }

        if (tensor_ncclAsyncErr != ncclSuccess) {
            // An asynchronous error happened. Stop the operation and destroy
            // the communicator
            tensor_ncclErr = ncclCommAbort(tensor_comm);
            if (tensor_ncclErr != ncclSuccess) {
                std::string error_msg = "NCCL Error : ncclCommDestroy returned " + std::to_string(tensor_ncclErr);
                throw std::runtime_error(error_msg);
            }
        }

        if (pipeline_ncclAsyncErr != ncclSuccess) {
            // An asynchronous error happened. Stop the operation and destroy
            // the communicator
            pipeline_ncclErr = ncclCommAbort(pipeline_comm);
            if (pipeline_ncclErr != ncclSuccess) {
                std::string error_msg = "NCCL Error : ncclCommDestroy returned " + std::to_string(pipeline_ncclErr);
                throw std::runtime_error(error_msg);
            }
        }
    }
#endif
}

void ftNcclGetUniqueId(NcclUid& uid)
{
#ifdef BUILD_MULTI_GPU
    NCCLCHECK(ncclGetUniqueId(&uid.nccl_uid_));
#endif
}

void ftNcclCommInitRank(NcclParam& param, const int rank, const int world_size, const NcclUid uid)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    // Initialize a nccl communicator.
    if (param.nccl_comm_ != nullptr) {
        TM_LOG_WARNING("NcclParam is already initialized.");
        return;
    }
    param.rank_       = rank;
    param.world_size_ = world_size;
    param.nccl_uid_   = uid.nccl_uid_;
    NCCLCHECK(ncclCommInitRank(&param.nccl_comm_, param.world_size_, param.nccl_uid_, param.rank_));
#endif
    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void ftNcclParamDestroy(NcclParam& param)
{
#ifdef BUILD_MULTI_GPU
    if (param.nccl_comm_ != nullptr) {
        ncclCommDestroy(param.nccl_comm_);
    }
#endif
}

void ftNcclInitialize(NcclParam& tensor_para,
                      NcclParam& pipeline_para,
                      const int  tensor_para_size,
                      const int  pipeline_para_size)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    // Initialize nccl communication grid of tensor and pipeline parallel groups.
#ifndef BUILD_MULTI_GPU
    FT_CHECK_WITH_INFO(tensor_para_size == 1,
                       fmtstr("tensor_para_size=%d although BUILD_MULTI_GPU is disabled. "
                              "Please use the cmake flag -DBUILD_MULTI_GPU=ON if you want "
                              "to use tensor/pipeline parallelism.",
                              tensor_para_size));
    FT_CHECK_WITH_INFO(pipeline_para_size == 1,
                       fmtstr("pipeline_para_size=%d although BUILD_MULTI_GPU is disabled. "
                              "Please use the cmake flag -DBUILD_MULTI_GPU=ON if you want "
                              "to use tensor/pipeline parallelism.",
                              pipeline_para_size));
    tensor_para.rank_         = 0;
    tensor_para.world_size_   = tensor_para_size;
    pipeline_para.rank_       = 0;
    pipeline_para.world_size_ = pipeline_para_size;
#else
    // Initialize a nccl communicator.
    if (tensor_para.nccl_comm_ != nullptr && pipeline_para.nccl_comm_ != nullptr) {
        TM_LOG_WARNING("NcclParam is already initialized. Skip NCCL initialization.");
        return;
    }
    FT_CHECK(tensor_para.nccl_comm_ == nullptr);
    FT_CHECK(pipeline_para.nccl_comm_ == nullptr);
    FT_CHECK(tensor_para_size > 0);
    FT_CHECK(pipeline_para_size > 0);

    if (tensor_para_size == 1 && pipeline_para_size == 1) {
        TM_LOG_WARNING("Skip NCCL initialization since requested tensor/pipeline parallel sizes are equals to 1.");
        tensor_para.rank_         = 0;
        tensor_para.world_size_   = tensor_para_size;
        pipeline_para.rank_       = 0;
        pipeline_para.world_size_ = pipeline_para_size;
        return;
    }

    int mpi_initialized;
    MPICHECK(MPI_Initialized(&mpi_initialized));
    FT_CHECK_WITH_INFO(mpi_initialized, "Fail to nccl initialization because MPI is not initialized.");

    int rank, world_size;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    FT_CHECK_WITH_INFO(tensor_para_size * pipeline_para_size <= world_size,
                       fmtstr("tensor_para_size (%d) * pipeline_para_size (%d) should equal to the world size (%d).",
                              tensor_para_size,
                              pipeline_para_size,
                              world_size));

    // Convert WORLD communicator into 2D grid (k * n) communicator.
    //  row = a tensor parallel group, col = a pipeline parallel group.
    MPI_Comm grid_comm, tp_comm, pp_comm;

    int dims[2]    = {pipeline_para_size, tensor_para_size};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

    // Split 2D communicator into rows and cols.
    int tp_remain_dims[2] = {false, true};
    int pp_remain_dims[2] = {true, false};
    MPI_Cart_sub(grid_comm, tp_remain_dims, &tp_comm);
    MPI_Cart_sub(grid_comm, pp_remain_dims, &pp_comm);

    int tp_rank, pp_rank;
    MPI_Comm_rank(tp_comm, &tp_rank);
    MPI_Comm_rank(pp_comm, &pp_rank);

    ncclUniqueId tp_uid;
    ncclUniqueId pp_uid;
    // The root of each group creates a nccl uid.
    if (tp_rank == 0) {
        TM_LOG_DEBUG("rank %d pp rank %d creates nccl uid.", rank, tp_rank);
        NCCLCHECK(ncclGetUniqueId(&tp_uid));
    }
    if (pp_rank == 0) {
        TM_LOG_DEBUG("rank %d pp rank %d creates nccl uid.", rank, pp_rank);
        NCCLCHECK(ncclGetUniqueId(&pp_uid));
    }
    // Broadcast nccl uid to share the same nccl uid across gpus in the same group.
    TM_LOG_DEBUG("Broadcast nccl uid to the others in the same parallel groups.");
    MPI_Bcast(&tp_uid, sizeof(tp_uid), MPI_BYTE, 0, tp_comm);
    MPI_Bcast(&pp_uid, sizeof(pp_uid), MPI_BYTE, 0, pp_comm);

    TM_LOG_DEBUG("Initialize NCCL communicators.");
    ncclComm_t tp_nccl_comm, pp_nccl_comm;
    NCCLCHECK(ncclCommInitRank(&tp_nccl_comm, tensor_para_size, tp_uid, tp_rank));
    NCCLCHECK(ncclCommInitRank(&pp_nccl_comm, pipeline_para_size, pp_uid, pp_rank));

    tensor_para.world_size_   = tensor_para_size;
    tensor_para.rank_         = tp_rank;
    tensor_para.nccl_uid_     = tp_uid;
    tensor_para.nccl_comm_    = tp_nccl_comm;
    pipeline_para.world_size_ = pipeline_para_size;
    pipeline_para.rank_       = pp_rank;
    pipeline_para.nccl_uid_   = pp_uid;
    pipeline_para.nccl_comm_  = pp_nccl_comm;
    TM_LOG_INFO("NCCL initialized rank=%d world_size=%d tensor_para=%s pipeline_para=%s",
                rank,
                world_size,
                tensor_para.toString().c_str(),
                pipeline_para.toString().c_str());
#endif
    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

static std::atomic<int>& ncclGroupCount()
{
    static std::atomic<int> value{};
    return value;
}

int ftNcclNextGroupId()
{
    return ncclGroupCount()++;
}

int ftNcclGroupCount()
{
    return ncclGroupCount();
}

size_t getLocalBatchSize(const size_t batch_size, const size_t seq_len, const size_t pipeline_para_size)
{
    size_t local_batch_size = batch_size;
    if (pipeline_para_size == 1) {
        return local_batch_size;
    }
    if (local_batch_size % pipeline_para_size == 0) {
        local_batch_size /= pipeline_para_size;
    }
    while (local_batch_size * seq_len > 1024 && local_batch_size % 2 == 0) {
        local_batch_size /= 2;
    }
    return local_batch_size;
}

}  // namespace turbomind

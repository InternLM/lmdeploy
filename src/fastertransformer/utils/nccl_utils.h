/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/mpi_utils.h"

#include <cuda_runtime.h>
#ifdef BUILD_MULTI_GPU
#include <mpi.h>
#include <nccl.h>
#endif
#include <stdio.h>
#include <string>

#if defined(NCCL_VERSION_CODE) && (NCCL_VERSION_CODE >= 21003)
#define ENABLE_BF16_NCCL
#endif

namespace fastertransformer {
#ifdef BUILD_MULTI_GPU
#define NCCLCHECK(cmd)                                                                                                 \
    do {                                                                                                               \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess) {                                                                                        \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                      \
            __builtin_trap();                                                                                          \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#else
#define NCCLCHECK(cmd) printf("[WARNING} No NCCL");
#endif

struct NcclUid {
#ifndef BUILD_MULTI_GPU
    NcclUid(){};
    NcclUid(NcclUid const& uid){};
#else
    ncclUniqueId nccl_uid_;
    NcclUid(){};
    NcclUid(NcclUid const& uid): nccl_uid_(uid.nccl_uid_){};
#endif
};

struct NcclParam {
    int rank_{0};
    int world_size_{1};
    int group_id_{0};
#ifdef BUILD_MULTI_GPU
    ncclUniqueId nccl_uid_{};
    ncclComm_t   nccl_comm_ = nullptr;
#endif

#ifdef BUILD_MULTI_GPU
    NcclParam(): rank_(0), world_size_(1), nccl_comm_(nullptr){};
    NcclParam(int rank, int world_size): rank_(rank), world_size_(world_size){};
    NcclParam(NcclParam const& param):
        rank_(param.rank_),
        world_size_(param.world_size_),
        group_id_(param.group_id_),
        nccl_uid_(param.nccl_uid_),
        nccl_comm_(param.nccl_comm_){};
    std::string toString()
    {
        return fmtstr(
            "NcclParam[rank=%d, world_size=%d, nccl_comm=%p, group_id=%d]", rank_, world_size_, nccl_comm_, group_id_);
    }
#else
    NcclParam(): rank_(0), world_size_(1){};
    NcclParam(int rank, int world_size): rank_(rank), world_size_(world_size){};
    NcclParam(NcclParam const& param): rank_(param.rank_), world_size_(param.world_size_){};
    std::string toString()
    {
        return fmtstr("NcclParam[rank=%d, world_size=%d]", rank_, world_size_);
    }
#endif
};

// New APIs
template<typename T>
void ftNcclAllReduceSum(const T* send_buf, T* recv_buf, const int data_size, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclAllGather(
    const T* send_buf, T* recv_buf, const int data_size, const int rank, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclBroadCast(T* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclRecv(T* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclSend(const T* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);

// nccl stream synchronize, abort nccl comms and throw errors when nccl async errors detected
void ftNcclStreamSynchronize(NcclParam tensor_para, NcclParam pipeline_para_, cudaStream_t stream);

void ftNcclGroupStart();
void ftNcclGroupEnd();
void ftNcclGetUniqueId(NcclUid& uid);
void ftNcclCommInitRank(NcclParam& param, const int rank, const int world_size, const NcclUid uid);
void ftNcclParamDestroy(NcclParam& param);

void ftNcclInitialize(NcclParam& tensor_para,
                      NcclParam& pipeline_para,
                      const int  tensor_para_size,
                      const int  pipeline_para_size);

int ftNcclNextGroupId();
int ftNcclGroupCount();

size_t getLocalBatchSize(const size_t batch_size, const size_t seq_len, const size_t pipeline_para_size);

}  // namespace fastertransformer

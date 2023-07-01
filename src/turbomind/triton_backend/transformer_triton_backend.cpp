/*
 * Copyright (c) OpenMMLab. All rights reserved.
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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/triton_backend/transformer_triton_backend.cpp

#include "src/turbomind/triton_backend/transformer_triton_backend.hpp"
#include "src/turbomind/utils/nccl_utils.h"

std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>>
AbstractTransformerModel::createNcclParams(const int node_id, const int device_id_start, const bool multi_node)
{
    const int gpu_count          = ft::getDeviceCount();
    const int tensor_para_size   = getTensorParaSize();
    const int pipeline_para_size = getPipelineParaSize();
    const int local_comm_size    = multi_node ? gpu_count : tensor_para_size * pipeline_para_size;
    ft::FT_CHECK(tensor_para_size > 0 && pipeline_para_size > 0);
    ft::FT_CHECK(device_id_start + (int)local_comm_size <= gpu_count);

    std::vector<ft::NcclUid> nccl_ids;
    if (tensor_para_size > 1 || pipeline_para_size > 1) {
        nccl_ids.resize(tensor_para_size + pipeline_para_size);
        if (node_id == 0) {
            for (uint32_t i = 0; i < nccl_ids.size(); i++) {
                ft::ftNcclGetUniqueId(nccl_ids[i]);
            }
        }
        for (size_t i = 0; i < nccl_ids.size(); i++) {
            ft::mpi::bcast(&nccl_ids[i], sizeof(nccl_ids[i]), ft::mpi::MPI_TYPE_BYTE, 0, ft::mpi::COMM_WORLD);
        }
    }

    std::vector<ft::NcclParam> tensor_para_params(local_comm_size);
    std::vector<ft::NcclParam> pipeline_para_params(local_comm_size);
    // Don't init comm when size == 1
    if (tensor_para_size > 1) {
        const auto group_id = ft::ftNcclNextGroupId();
        ft::ftNcclGroupStart();
        for (int gid = device_id_start; gid < device_id_start + local_comm_size; gid++) {
            int rank               = node_id * gpu_count + gid - device_id_start;
            int tensor_para_rank   = rank % tensor_para_size;
            int pipeline_para_rank = rank / tensor_para_size;

            ft::NcclUid tensor_para_nccl_uid = nccl_ids[pipeline_para_rank];
            ft::check_cuda_error(cudaSetDevice(gid));
            ft::ftNcclCommInitRank(
                tensor_para_params[gid - device_id_start], tensor_para_rank, tensor_para_size, tensor_para_nccl_uid);
            tensor_para_params[gid - device_id_start].group_id_ = group_id;
        }
        ft::ftNcclGroupEnd();
    }
    if (pipeline_para_size > 1) {
        const auto group_id = ft::ftNcclNextGroupId();
        ft::ftNcclGroupStart();
        for (int gid = device_id_start; gid < device_id_start + local_comm_size; gid++) {
            int rank               = node_id * gpu_count + gid - device_id_start;
            int tensor_para_rank   = rank % tensor_para_size;
            int pipeline_para_rank = rank / tensor_para_size;

            ft::NcclUid pipeline_para_nccl_uid = nccl_ids[pipeline_para_size + tensor_para_rank];
            ft::check_cuda_error(cudaSetDevice(gid));
            ft::ftNcclCommInitRank(pipeline_para_params[gid - device_id_start],
                                   pipeline_para_rank,
                                   pipeline_para_size,
                                   pipeline_para_nccl_uid);
            pipeline_para_params[gid - device_id_start].group_id_ = group_id;
        }
        ft::ftNcclGroupEnd();
    }
    return std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>>(tensor_para_params, pipeline_para_params);
}

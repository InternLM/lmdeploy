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

#include "src/turbomind/utils/mpi_utils.h"

namespace fastertransformer {
namespace mpi {

#ifdef BUILD_MULTI_GPU
MPI_Datatype getMpiDtype(MpiType dtype)
{
    static const std::unordered_map<MpiType, MPI_Datatype> dtype_map{
        {MPI_TYPE_BYTE, MPI_BYTE},
        {MPI_TYPE_CHAR, MPI_CHAR},
        {MPI_TYPE_INT, MPI_INT},
        {MPI_TYPE_INT64_T, MPI_INT64_T},
        {MPI_TYPE_UINT32_T, MPI_UINT32_T},
        {MPI_TYPE_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG},
    };
    return dtype_map.at(dtype);
}
#endif

void initialize(int* argc, char*** argv)
{
#ifdef BUILD_MULTI_GPU
    MPICHECK(MPI_Init(argc, argv));
#endif
}

void finalize()
{
#ifdef BUILD_MULTI_GPU
    MPICHECK(MPI_Finalize());
#endif
}

bool isInitialized()
{
    int mpi_initialized = 0;
#ifdef BUILD_MULTI_GPU
    MPICHECK(MPI_Initialized(&mpi_initialized));
#endif
    return static_cast<bool>(mpi_initialized);
}

void initThread(int* argc, char*** argv, MpiThreadSupport required, int* provided)
{
#ifdef BUILD_MULTI_GPU
    switch (required) {
        case THREAD_SINGLE:
            MPICHECK(MPI_Init_thread(argc, argv, MPI_THREAD_SINGLE, provided));
            break;
        case THREAD_FUNNELED:
            MPICHECK(MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, provided));
            break;
        case THREAD_SERIALIZED:
            MPICHECK(MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, provided));
            break;
        case THREAD_MULTIPLE:
            MPICHECK(MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, provided));
            break;
        default:
            break;
    }
#endif
}

int getCommWorldRank()
{
    int rank = 0;
#ifdef BUILD_MULTI_GPU
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    return rank;
}

int getCommWorldSize()
{
    int world_size = 1;
#ifdef BUILD_MULTI_GPU
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif
    return world_size;
}

void barrier(MpiComm comm)
{
#ifdef BUILD_MULTI_GPU
    MPICHECK(MPI_Barrier(comm.group));
#endif
}

void barrier()
{
#ifdef BUILD_MULTI_GPU
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
#endif
}

void bcast(void* buffer, size_t size, MpiType dtype, int root, MpiComm comm)
{
#ifdef BUILD_MULTI_GPU
    MPICHECK(MPI_Bcast(buffer, size, getMpiDtype(dtype), root, comm.group));
#endif
}

}  // namespace mpi
}  // namespace fastertransformer

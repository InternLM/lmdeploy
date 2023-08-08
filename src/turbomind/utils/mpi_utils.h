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

#pragma once

#include "src/turbomind/utils/logger.h"

#ifdef BUILD_MULTI_GPU
#include <mpi.h>
#endif
#include <stdio.h>
#include <unordered_map>

namespace turbomind {

#ifdef BUILD_MULTI_GPU
#define MPICHECK(cmd)                                                                                                  \
    do {                                                                                                               \
        int e = cmd;                                                                                                   \
        if (e != MPI_SUCCESS) {                                                                                        \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                                           \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#else
#define MPICHECK(cmd) printf("[WARNING] No MPI\n");
#endif

// A wrapper module of the MPI library.
namespace mpi {

// A wrapper of MPI data type. MPI_TYPE_{data_type}
enum MpiType {
    MPI_TYPE_BYTE,
    MPI_TYPE_CHAR,
    MPI_TYPE_INT,
    MPI_TYPE_INT64_T,
    MPI_TYPE_UINT32_T,
    MPI_TYPE_UNSIGNED_LONG_LONG,
};

// A wrapper of the level of MPI thread support
enum MpiThreadSupport {
    THREAD_SINGLE,
    THREAD_FUNNELED,
    THREAD_SERIALIZED,
    THREAD_MULTIPLE
};

struct MpiComm {
#ifdef BUILD_MULTI_GPU
    MPI_Comm group;
    MpiComm(){};
    MpiComm(MPI_Comm g): group(g){};
#endif
};

#ifdef BUILD_MULTI_GPU
#define COMM_WORLD MpiComm(MPI_COMM_WORLD)
#else
#define COMM_WORLD MpiComm()
#endif

#ifdef BUILD_MULTI_GPU
MPI_Datatype getMpiDtype(MpiType dtype);
#endif

void initialize(int* argc, char*** argv);
void initThread(int* argc, char*** argv, MpiThreadSupport required, int* provided);
void finalize();
bool isInitialized();
void barrier(MpiComm comm);
void barrier();

int getCommWorldRank();
int getCommWorldSize();

void bcast(void* buffer, size_t size, MpiType dtype, int root, MpiComm comm);

}  // namespace mpi
}  // namespace turbomind

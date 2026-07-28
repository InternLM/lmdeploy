# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(memory)
add_subdirectory(kernels)
add_subdirectory(comm)
add_subdirectory(generation)
add_subdirectory(models)
add_subdirectory(engine)

add_library(turbomind STATIC
        turbomind.cc
        rust/c_api.cc)
set_property(TARGET turbomind PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(turbomind PUBLIC
        engine
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,models>"
        device_comm
        host_comm
        core
        memory
        copy
        memory_utils
        nvtx_utils
        CUDA::cublasLt
        CUDA::cudart
        )

if(BUILD_RUST_API_SERVER)
    find_program(CARGO_EXECUTABLE cargo
                 HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin")
    if(NOT CARGO_EXECUTABLE)
        message(FATAL_ERROR
                "BUILD_RUST_API_SERVER=ON requires Cargo. Install Rust with rustup and ensure cargo is on PATH.")
    endif()
    file(GLOB_RECURSE RUST_API_SERVER_SOURCES CONFIGURE_DEPENDS
         "${PROJECT_SOURCE_DIR}/rust/Cargo.toml"
         "${PROJECT_SOURCE_DIR}/rust/Cargo.lock"
         "${PROJECT_SOURCE_DIR}/rust/crates/*.toml"
         "${PROJECT_SOURCE_DIR}/rust/crates/*.rs")
    set(RUST_TARGET_DIR "${CMAKE_BINARY_DIR}/rust-target")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(RUST_PROFILE_DIR "debug")
        set(RUST_PROFILE_ARG)
    else()
        set(RUST_PROFILE_DIR "release")
        set(RUST_PROFILE_ARG --release)
    endif()
    set(RUST_API_SERVER_LIBRARY
        "${RUST_TARGET_DIR}/${RUST_PROFILE_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}rust_api_server${CMAKE_STATIC_LIBRARY_SUFFIX}")
    add_custom_command(
        OUTPUT "${RUST_API_SERVER_LIBRARY}"
        COMMAND "${CMAKE_COMMAND}" -E env "CARGO_TARGET_DIR=${RUST_TARGET_DIR}"
                "${CARGO_EXECUTABLE}" build --locked -p rust-api-server --features ffi ${RUST_PROFILE_ARG}
        DEPENDS ${RUST_API_SERVER_SOURCES}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/rust"
        COMMENT "Building Rust API server"
        VERBATIM)
    add_custom_target(rust_api_server_build DEPENDS "${RUST_API_SERVER_LIBRARY}")
    add_library(rust_api_server STATIC IMPORTED GLOBAL)
    set_target_properties(rust_api_server PROPERTIES IMPORTED_LOCATION "${RUST_API_SERVER_LIBRARY}")
    add_dependencies(rust_api_server rust_api_server_build)
endif()

if(BUILD_PY_FFI)
    add_subdirectory(python)
endif()

# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.25)

block()
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

    set(BUILD_SHARED OFF)
    set(BUILD_STATIC ON)
    set(BUILD_TESTS OFF)
    set(WITH_PERF_TOOL OFF)
    set(ENABLE_DRAFTS OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    FetchContent_Declare(
        libzmq
        GIT_REPOSITORY https://github.com/zeromq/libzmq.git
        GIT_TAG v4.3.5
        GIT_SHALLOW ON)
    FetchContent_MakeAvailable(libzmq)

    set(CPPZMQ_BUILD_TESTS OFF)
    FetchContent_Declare(
        cppzmq
        GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
        GIT_TAG v4.11.0
        GIT_SHALLOW ON)
    FetchContent_MakeAvailable(cppzmq)

    set(MSGPACK_CXX11 OFF)
    set(MSGPACK_CXX17 ON)
    set(MSGPACK_USE_BOOST OFF)
    set(MSGPACK_BUILD_TESTS OFF)
    set(MSGPACK_BUILD_DOCS OFF)
    set(MSGPACK_BUILD_EXAMPLES OFF)
    FetchContent_Declare(
        msgpack_cxx
        GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
        GIT_TAG cpp-8.0.0
        GIT_SHALLOW ON)
    FetchContent_MakeAvailable(msgpack_cxx)
endblock()

add_library(lmcache_connector STATIC connector.cc lookup.cc store.cc retrieve.cc mq.cc)
set_property(TARGET lmcache_connector PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(lmcache_connector PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(lmcache_connector PUBLIC msgpack-cxx PRIVATE core cppzmq-static)

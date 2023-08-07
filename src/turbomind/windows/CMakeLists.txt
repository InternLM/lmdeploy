cmake_minimum_required(VERSION 3.14)

if (NOT MSVC)
    return ()
endif ()

include(FetchContent)

# libpthreadVC3
SET(PTHREAD_WIN32_SRC ${CMAKE_BINARY_DIR}/_deps/pthread-win32-src/)
SET(PTHREAD_WIN32_BINARY ${CMAKE_BINARY_DIR}/_deps/pthread-win32-build/)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PTHREAD_WIN32_SRC}/cmake")

FetchContent_Declare(
    pthread-win32
    GIT_REPOSITORY https://github.com/GerHobbelt/pthread-win32
    GIT_TAG version-3.1.0-release
    PATCH_COMMAND git apply --ignore-whitespace "${CMAKE_CURRENT_LIST_DIR}/patch.diff"
    GIT_SHALLOW ON
    UPDATE_DISCONNECTED 1
)

FetchContent_MakeAvailable(pthread-win32)
target_include_directories(libpthreadVC3 PRIVATE ${PTHREAD_WIN32_SRC})

cmake_minimum_required(VERSION 3.14)

if (NOT MSVC)
    return ()
endif ()

# dirent.h
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(FetchContent)

# libpthreadVC3
SET(PTHREAD_WIN32_SRC ${CMAKE_BINARY_DIR}/_deps/pthread-win32-src/)
SET(PTHREAD_WIN32_BINARY ${CMAKE_BINARY_DIR}/_deps/pthread-win32-build/)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PTHREAD_WIN32_SRC}/cmake")

FetchContent_Declare(
    pthread-win32
    GIT_REPOSITORY https://github.com/GerHobbelt/pthread-win32
    GIT_TAG version-3.1.0-release
    GIT_SHALLOW ON
)

file(READ "${PTHREAD_WIN32_SRC}/cmake/get_version.cmake" FILE_CONTENT)
string(REPLACE "CMAKE_SOURCE_DIR" "CMAKE_CURRENT_SOURCE_DIR" FILE_CONTENT "${FILE_CONTENT}" )
file(WRITE "${PTHREAD_WIN32_SRC}/cmake/get_version.cmake" "${FILE_CONTENT}")

file(READ "${PTHREAD_WIN32_SRC}/cmake/config_h_in.cmake" FILE_CONTENT)
string(REPLACE "\${CMAKE_BINARY_DIR}" "${PTHREAD_WIN32_BINARY}" FILE_CONTENT "${FILE_CONTENT}" )
file(WRITE "${PTHREAD_WIN32_SRC}/cmake/config_h_in.cmake" "${FILE_CONTENT}")

file(READ "${PTHREAD_WIN32_SRC}/cmake/target_arch.cmake" FILE_CONTENT)
string(REPLACE "\${CMAKE_BINARY_DIR}" "${PTHREAD_WIN32_BINARY}" FILE_CONTENT "${FILE_CONTENT}" )
file(WRITE "${PTHREAD_WIN32_SRC}/cmake/target_arch.cmake" "${FILE_CONTENT}")


file(READ "${PTHREAD_WIN32_SRC}/CMakeLists.txt" FILE_CONTENT)
string(REPLACE "\${CMAKE_SOURCE_DIR}/cmake/version.rc.in \${CMAKE_BINARY_DIR}/version.rc"
    "${PTHREAD_WIN32_SRC}/cmake/version.rc.in ${PTHREAD_WIN32_BINARY}/version.rc"
    FILE_CONTENT "${FILE_CONTENT}" )
string(REPLACE "\${CMAKE_SOURCE_DIR}"
    "${PTHREAD_WIN32_SRC}"
    FILE_CONTENT "${FILE_CONTENT}" )
string(REPLACE "\${CMAKE_BINARY_DIR}"
    "${PTHREAD_WIN32_BINARY}"
    FILE_CONTENT "${FILE_CONTENT}" )
string(REPLACE "install(FILES _ptw32.h pthread.h sched.h semaphore.h TYPE INCLUDE)"
    "" FILE_CONTENT "${FILE_CONTENT}" )
file(WRITE "${CMAKE_BINARY_DIR}/_deps/pthread-win32-src/CMakeLists.txt" "${FILE_CONTENT}")


FetchContent_MakeAvailable(pthread-win32)
target_include_directories(libpthreadVC3 PUBLIC ${PTHREAD_WIN32_SRC})

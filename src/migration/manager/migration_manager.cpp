#include <string>
#include <iostream>

#include "migration_manager.cuh"

std::vector<char> serialize_cuda_ipc_handle(const cudaIpcMemHandle_t& handle) {
    // 将 cudaIpcMemHandle_t 转换为字节流
    std::vector<char> buffer(reinterpret_cast<const char*>(&handle), reinterpret_cast<const char*>(&handle) + sizeof(cudaIpcMemHandle_t));
    return buffer;
}

cudaIpcMemHandle_t deserialize_from_vector(const std::vector<char>& buffer) {
    if (buffer.size() != sizeof(cudaIpcMemHandle_t)) {
        throw std::runtime_error("Invalid buffer size: expected " + std::to_string(sizeof(cudaIpcMemHandle_t)) +
                                 " bytes, got " + std::to_string(buffer.size()) + " bytes");
    }

    cudaIpcMemHandle_t result;
    std::memcpy(&result, buffer.data(), sizeof(cudaIpcMemHandle_t));
    return result;
}
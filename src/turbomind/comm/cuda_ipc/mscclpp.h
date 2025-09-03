// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace mscclpp {

// Copied from
// https://github.com/microsoft/mscclpp/blob/591276f9d07d2df8e2a45a16738e27867e468ca3/include/mscclpp/packet_device.hpp#L19
union alignas(16) LL16Packet {
    // Assume data is written with an atomicity of 8 bytes (IB/RDMA).
    struct {
        uint32_t data1;
        uint32_t flag1;
        uint32_t data2;
        uint32_t flag2;
    };
    using Payload = uint2;

    ulonglong2 raw_;

    __device__ LL16Packet() {}

    __device__ LL16Packet(uint2 val, uint32_t flag)
    {
        data1 = val.x;
        flag1 = flag;
        data2 = val.y;
        flag2 = flag;
    }

    /// Write 8 bytes of data to the packet.
    /// @param val1 The first 4-byte data to write.
    /// @param val2 The second 4-byte data to write.
    /// @param flag The flag to write.
    __device__ void write(uint32_t val1, uint32_t val2, uint32_t flag)
    {
        asm volatile(
            "st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(&raw_), "r"(val1), "r"(flag), "r"(val2), "r"(flag));
    }

    /// Write 8 bytes of data to the packet.
    /// @param val The 8-byte data to write.
    /// @param flag The flag to write.
    __device__ void write(uint64_t val, uint32_t flag)
    {
        write((uint32_t)val, (uint32_t)(val >> 32), flag);
    }

    /// Write 8 bytes of data to the packet.
    /// @param val The 8-byte data to write.
    /// @param flag The flag to write.
    __device__ void write(uint2 val, uint32_t flag)
    {
        write(val.x, val.y, flag);
    }

    /// Helper of @ref read().
    /// @param flag The flag to read.
    /// @param data The 8-byte data read.
    /// @return True if the flag is not equal to the given flag.
    __device__ bool readOnce(uint32_t flag, uint2& data) const
    {
        uint32_t flag1, flag2;
        asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(data.x), "=r"(flag1), "=r"(data.y), "=r"(flag2)
                     : "l"(&raw_));
        return (flag1 != flag) || (flag2 != flag);
    }

    /// Read 8 bytes of data from the packet.
    /// @param flag The flag to read.
    /// @return The 8-byte data read.
    __device__ uint2 read(uint32_t flag) const
    {
        uint2 data;
        while (readOnce(flag, data)) {}
        return data;
    }

    /// Clear the packet.
    __device__ void clear()
    {
        raw_ = make_ulonglong2(0, 0);
    }
};

using LLPacket = LL16Packet;

}  // namespace mscclpp

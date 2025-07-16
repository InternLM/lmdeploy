#pragma once

#include "src/turbomind/kernels/core/array.h"

namespace turbomind::comm {

inline constexpr int kMaxRanks        = 8;
static constexpr int kPacketBuffSize  = 8 << 20;  // 8 MB
static constexpr int kScratchBuffSize = 8 << 20;  // 8 MB
static constexpr int kMaxChannels     = 64;

template<class T>
struct SymmetricPtr_V2 {
    Array<T*, kMaxRanks> uc;
    T*                   mc;
};

}  // namespace turbomind::comm

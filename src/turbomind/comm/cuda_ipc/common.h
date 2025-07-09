#pragma once

#include "src/turbomind/kernels/core/array.h"

namespace turbomind::comm {

inline constexpr int kMaxRanks        = 8;
inline constexpr int kMaxNearPeers    = 7;
static constexpr int kPacketBuffSize  = 8 << 20;  // 8 MB
static constexpr int kScratchBuffSize = 8 << 20;  // 8 MB
static constexpr int kMaxChannels     = 64;

template<class T>
struct SymmetricPtr {
    Array<T*, kMaxNearPeers> uc;
    T*                       mc;
};

template<class T>
struct SymmetricPtr_V2 {
    Array<T*, kMaxRanks> uc;
    T*                   mc;
};

}  // namespace turbomind::comm
// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::gemm::config {

template<int... Dimensions>
struct Shape;

template<int M_, int N_>
struct Shape<M_, N_> {
    static constexpr int M = M_;
    static constexpr int N = N_;
};

template<int M_, int N_, int K_>
struct Shape<M_, N_, K_> {
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int K = K_;
};

template<int Producer_, int Math_>
struct Registers {
    static constexpr int Producer = Producer_;
    static constexpr int Math = Math_;
};

template<class... Parts>
struct Config;

template<class Tile_, class Groups_>
struct Config<Tile_, Groups_> {
    using Tile = Tile_;
    using Groups = Groups_;
};

template<class Tile_, class Groups_, class Registers_>
struct Config<Tile_, Groups_, Registers_>: Config<Tile_, Groups_> {
    using RegisterConfig = Registers_;
};

}  // namespace turbomind::gemm::config

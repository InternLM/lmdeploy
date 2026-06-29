// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/sub_byte_ptr.h"
#include <iostream>
#include <type_traits>

namespace turbomind {

namespace block {

template<class T, class Tkv, int HeadDim, bool ShareKV = false>
struct Config {
    int head_num_;
    int block_len_;

    TM_HOST_DEVICE constexpr int t_bits() const
    {
        if constexpr (std::is_same_v<T, Tkv>) {
            return 0;
        }
        else {
            return bitsof<T>;
        }
    }

    TM_HOST_DEVICE constexpr int q_bits() const
    {
        return bitsof<Tkv>;
    }

    TM_HOST_DEVICE constexpr int head_dim() const
    {
        return HeadDim;
    }

    TM_HOST_DEVICE int head_num() const
    {
        return head_num_;
    }

    TM_HOST_DEVICE constexpr int block_len() const
    {
        return block_len_;
    }

    TM_HOST_DEVICE constexpr bool is_share_kv() const
    {
        return ShareKV;
    }
};

// Layout -> byte offset -> head id -> token id -> [block] -> (k_data, v_data, k_param, v_param)

template<class T, class Tkv, class Layout>
class Head {
public:
    TM_HOST_DEVICE Head(Layout layout, int offset, int head_id): layout_{layout}, offset_{offset}, head_id_{head_id} {}

    TM_HOST_DEVICE auto k_data(char* block, int ti) const
    {
        if constexpr (std::is_same_v<Tkv, uint4_t>) {
            return SubBytePtr<Tkv>{block + offset_ + layout_.k_data(head_id_, ti)};
        }
        else {
            return reinterpret_cast<Tkv*>(block + offset_ + layout_.k_data(head_id_, ti));
        }
    }

    TM_HOST_DEVICE auto v_data(char* block, int ti) const
    {
        if constexpr (std::is_same_v<Tkv, uint4_t>) {
            return SubBytePtr<Tkv>{block + offset_ + layout_.v_data(head_id_, ti)};
        }
        else {
            return reinterpret_cast<Tkv*>(block + offset_ + layout_.v_data(head_id_, ti));
        }
    }

    TM_HOST_DEVICE T* k_param(char* block, int ti) const
    {
        return reinterpret_cast<T*>(block + offset_ + layout_.k_param(head_id_, ti));
    }

    TM_HOST_DEVICE T* v_param(char* block, int ti) const
    {
        return reinterpret_cast<T*>(block + offset_ + layout_.v_param(head_id_, ti));
    }

    TM_HOST_DEVICE void get_block_coord(int seq_ti, int& block_idx, int& block_ti) const
    {
        block_idx = seq_ti / block_len();
        block_ti  = seq_ti % block_len();
    }

    TM_HOST_DEVICE auto block_len() const
    {
        return layout_.config().block_len();
    }

    template<class Func>
    TM_HOST_DEVICE auto with(char** block_ptrs, int ti, Func&& func) const
    {
        int block_id;
        int block_ti;
        get_block_coord(ti, block_id, block_ti);

        char* block = block_ptrs[block_id];

        return ((Func&&)func)(
            k_data(block, block_ti), v_data(block, block_ti), k_param(block, block_ti), v_param(block, block_ti));
    }

private:
    Layout layout_;

    int offset_;
    int head_id_;
};

// L(H2SDQ+H2S2T)
template<class Config_>
struct Layout {

    using Config = Config_;

    Config config_;

    // This trivial ctor is defined for CTAD
    TM_HOST_DEVICE Layout(Config config): config_{config} {}

    TM_HOST_DEVICE const Config& config() const
    {
        return config_;
    }

    TM_HOST_DEVICE constexpr bool is_share_kv() const
    {
        // return 0;
        return config().is_share_kv();
    }

    TM_HOST_DEVICE constexpr int kv_num() const
    {
        // return 2;
        return is_share_kv() ? 1 : 2;
    }

    TM_HOST_DEVICE int token_data_size() const
    {
        return config().q_bits() * config().head_dim() / 8;
    }

    TM_HOST_DEVICE int token_param_size() const
    {
        return config().t_bits() * 2 / 8;  // 2 for scales/zeros
    }

    TM_HOST_DEVICE int head_data_size() const
    {
        return config().block_len() * token_data_size();
    }

    TM_HOST_DEVICE int head_param_size() const
    {
        return config().block_len() * token_param_size();
    }

    TM_HOST_DEVICE int layer_size() const
    {
        // TODO: enforce alignment
        return config().head_num() * kv_num() * head_data_size() + config().head_num() * kv_num() * head_param_size();
    }

    TM_HOST_DEVICE int k_data(int head, int token) const
    {
        return head_data(head) + token_data(token);
    }

    TM_HOST_DEVICE int v_data(int head, int token) const
    {
        return k_data(head, token) + (is_share_kv() ? 0 : head_data_size());
    }

    TM_HOST_DEVICE int k_param(int head, int token) const
    {
        return head_param(head) + token_param(token);
    }

    TM_HOST_DEVICE int v_param(int head, int token) const
    {
        return k_param(head, token) + (is_share_kv() ? 0 : head_param_size());
    }

    TM_HOST_DEVICE int head_data(int head) const
    {
        return head * kv_num() * head_data_size();
    }

    TM_HOST_DEVICE int head_param(int head) const
    {
        return head * kv_num() * head_param_size();
    }

    TM_HOST_DEVICE int token_data(int ti) const
    {
        return ti * token_data_size();
    }

    TM_HOST_DEVICE int token_param(int ti) const
    {
        return ti * token_param_size();
    }
};

template<class Config>
void dump(const Layout<Config>& layout)
{
    std::cout << "head_dim: " << layout.config().head_dim() << "\n";
    std::cout << "head_num: " << layout.config().head_num() << "\n";
    std::cout << "block_len: " << layout.config().block_len() << "\n";
    std::cout << "q_bits: " << layout.config().q_bits() << "\n";
    std::cout << "t_bits: " << layout.config().t_bits() << "\n";
    std::cout << "token_data_size: " << layout.token_data_size() << "\n";
    std::cout << "token_param_size: " << layout.token_param_size() << "\n";
    std::cout << "head_data_size: " << layout.head_data_size() << "\n";
    std::cout << "head_param_size: " << layout.head_param_size() << "\n";
    std::cout << "layer_size: " << layout.layer_size() << "\n";
}

}  // namespace block

}  // namespace turbomind

// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/logger.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <queue>
#include <unordered_map>
#include <vector>

namespace fastertransformer {

// k-cache layout [L, H, D/x, S[s:], x]
// v-cache layout [L, H, S[s:], D/x, x]

class LlamaCacheManager {
public:
    LlamaCacheManager(size_t      layer_num,
                      size_t      head_num,
                      size_t      size_per_head,
                      size_t      max_seq_len,
                      size_t      elem_bits,
                      size_t      max_entry_count,
                      size_t      chunk_size,
                      int         rank,
                      IAllocator* allocator):
        layer_num_(layer_num),
        head_num_(head_num),
        size_per_head_(size_per_head),
        max_seq_len_(max_seq_len),
        elem_bits_(elem_bits),
        cache_byte_size_(layer_num_ * head_num_ * max_seq_len_ * size_per_head_ * elem_bits_ / 8),
        max_entry_count_(max_entry_count),
        chunk_size_(chunk_size),
        rank_(rank),
        allocator_(allocator)
    {
        if (rank == 0) {
            FT_LOG_INFO("[LlamaCacheManager] max_entry_count = %d", (int)max_entry_count_);
            FT_LOG_INFO("[LlamaCacheManager] chunk_size = %d", (int)chunk_size_);
        }
        allocate(true);
    }

    ~LlamaCacheManager();

    struct Sequence {
        // header
        uint64_t id;
        size_t   max_seq_len;

        // payloads
        std::vector<int> token_ids;  // all token ids
        size_t           cache_len;  // cache_len == 0 -> cache miss
        void*            k_cache;
        void*            v_cache;

        std::vector<uint8_t> random_state_;  // states for RNGs

        // for LRU policy
        uint64_t timestamp;
    };

    Sequence create(uint64_t id, cudaStream_t stream);

    Sequence fetch(uint64_t id, cudaStream_t stream);

    void update(const Sequence& seq, cudaStream_t stream);

    void erase(uint64_t id);

    bool contains(uint64_t id) const noexcept;

private:
    std::vector<Sequence>::iterator getEntryOrThrow(uint64_t id);

    void* allocate(bool is_preallocte);

    void* evict();

private:
    const size_t layer_num_{};
    const size_t head_num_{};
    const size_t size_per_head_{};
    const size_t max_seq_len_{};
    const size_t elem_bits_{};
    const size_t cache_byte_size_{};
    const size_t max_entry_count_{};
    const size_t chunk_size_{};
    const int    rank_{};
    IAllocator*  allocator_{};

    std::queue<void*>  device_free_;
    std::vector<void*> device_mem_;
    int                entry_count_{};

    uint64_t timestamp_{};

    std::vector<Sequence> device_cache_;
};

}  // namespace fastertransformer

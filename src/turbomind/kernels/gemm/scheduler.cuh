// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "cutlass/fast_math.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

namespace turbomind::gemm {

TM_DEVICE void mbarrier_arrive_cluster(uint64_t* mbar, int cta_id, int pred)
{
    uint32_t smem_addr = cast_smem_ptr_to_uint(mbar);
    if (pred) {
        asm volatile("{\n"
                     ".reg .b32 remAddr32;\n"
                     "mapa.shared::cluster.u32  remAddr32, %0, %1;\n"
                     "mbarrier.arrive.release.cluster.shared::cluster.b64  _, [remAddr32];\n"
                     "}"
                     :
                     : "r"(smem_addr), "r"(cta_id));
    }
}

TM_DEVICE void mbarrier_wait_cluster(uint64_t* mbar, uint32_t phase)
{
    uint32_t smem_addr = cast_smem_ptr_to_uint(mbar);
    uint32_t ticks     = 0x989680;
    asm volatile("{\n"
                 ".reg .pred       P1; \n"
                 "LAB_WAIT: \n"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1, %2; \n"
                 "@P1 bra DONE; \n"
                 "bra     LAB_WAIT; \n"
                 "DONE: \n"
                 "}"
                 :
                 : "r"(smem_addr), "r"(phase), "r"(ticks));
}

TM_DEVICE void* map_to_cta(void* ptr, int cta_id)
{
    void* ret;
    asm volatile("mapa.u64 %0, %1, %2;\n" : "=l"(ret) : "l"(ptr), "r"(cta_id));
    return ret;
}

TM_DEVICE void st_shared_cluster(uint32_t ptr, int value)
{
    asm volatile("st.shared::cluster.s32 [%0], %1;\n" ::"r"(ptr), "r"(value));
}

template<class T, class M>
constexpr int member_offset(M T::*member)
{
    return reinterpret_cast<std::size_t>(&(reinterpret_cast<T*>(0)->*member));
}

template<Order order,
         class Cluster,
         int  striped_m,
         bool striped_n,
         int  tile_m,
         int  tile_n,
         int  Stages_,
         bool is_grouped_gemm>
struct TileScheduler {

    static constexpr bool is_dynamic = 1;  // is_grouped_gemm;
    static constexpr int  Stages     = Stages_;

    static constexpr int2 tile_{tile_m, tile_n};
    static constexpr int2 cluster_tile_{tile_m * Cluster::M, tile_n* Cluster::N};

    int4 gemm_shape_;
    int2 tiled_shape_;

    int log_tile_;
    int k_iters_;

    int2 tile_offset_;
    int2 iter_k_range_;

    int clusters_;

    //////// v2 /////
    int2 swizzle_unit_;
    int2 cluster_tiles_;
    int2 padded_cluster_tiles_;
    int2 swizzled_cluster_tiles_;

    cutlass::FastDivmod swizzle_tile_x_;
    /////////////

    const int* offsets_;

    int* next_cluster_id_;

    using PipelineState = cutlass::PipelineState<Stages>;

    struct Tile0 {
        int is_valid_cta;
        int is_valid_cluster;
        int offset_m;
        int offset_n;
        int alive;
    };

    struct Tile1 {
        int is_valid_cta;
        int is_valid_cluster;
        int offset_m;
        int offset_n;
        int alive;
        int group_idx;
        int m0;
        int m1;
    };

    using Tile = std::conditional_t<is_grouped_gemm, Tile1, Tile0>;

    struct Storage {
        Tile tile[Stages];
        __align__(8) uint64_t producer_bar[Stages];
        __align__(8) uint64_t consumer_bar[Stages];
    };

    struct ConsumerState {
        PipelineState  pipe;
        Storage&       store;
        TileScheduler& sched;

        TM_DEVICE bool acquire(Tile*& tile)
        {
            return sched.acquire(*this, tile);
        }

        TM_DEVICE void release(int step = 1)
        {
            return sched.release(*this, step);
        }
    };

    struct ProducerState {
        PipelineState  pipe;
        int            group_id_offset;
        int            cluster_idx;
        Storage&       store;
        TileScheduler& sched;

        TM_DEVICE bool next()
        {
            return sched.next(*this);
        }
    };

public:
    TM_DEVICE void init_dyanmic(Storage& store, int consumer_num)
    {
        for (int i = 0; i < Stages; ++i) {
            cutlass::arch::ClusterBarrier::init(&store.producer_bar[i], 1);
            cutlass::arch::ClusterBarrier::init(&store.consumer_bar[i], consumer_num);
        }
        // cutlass::arch::ClusterBarrier::init(&store.sync_bar, 1);
    }

    TM_HOST_DEVICE void init(int4 gemm_shape, int log_tile, int3 tile_shape)
    {
        gemm_shape_ = gemm_shape;

        // printf("gemm shape: %d %d %d\n", gemm_shape.x, gemm_shape.y, gemm_shape.z);

        log_tile_ = log_tile;
        k_iters_  = cdiv(gemm_shape_.z, tile_shape.z);

        tiled_shape_.x = cdiv(gemm_shape.x, tile_.x);
        tiled_shape_.y = cdiv(gemm_shape.y, tile_.y);

        cluster_tiles_.x = cdiv(gemm_shape.x, cluster_tile_.x);  // useless
        cluster_tiles_.y = cdiv(gemm_shape.y, cluster_tile_.y);

        // printf("cluster tiles: %d %d\n", cluster_tiles_.x, cluster_tiles_.y);

        if constexpr (is_grouped_gemm) {
            {
                int2 unit     = get_swizzled_shape({1, 1}, log_tile);
                swizzle_unit_ = order == kColMajor ? int2{unit.y, unit.x} : int2{unit.x, unit.y};
            }

            // col {8, 1}, row {1, 8}
            // printf("swizzle unit: %d %d\n", swizzle_unit_.x, swizzle_unit_.y);

            swizzle_tile_x_ = cluster_tile_.x * swizzle_unit_.x;

            int num = gemm_shape_.w;

            // num of tiles won't change after swizzle
            padded_cluster_tiles_.x = (num + gemm_shape.x / (cluster_tile_.x * swizzle_unit_.x)) * swizzle_unit_.x;
            padded_cluster_tiles_.y = cdiv(gemm_shape.y, cluster_tile_.y * swizzle_unit_.y) * swizzle_unit_.y;

            // printf("padded   cluster tiles: %d %d\n", padded_cluster_tiles_.x, padded_cluster_tiles_.y);

            swizzled_cluster_tiles_ = get_swizzled_shape(padded_cluster_tiles_, log_tile);

            // printf("swizzled cluster tiles: %d %d\n", swizzled_cluster_tiles_.x, swizzled_cluster_tiles_.y);

            clusters_ = padded_cluster_tiles_.x * padded_cluster_tiles_.y;

            // printf("clusters = %d\n", clusters_);
            // M is runtime value
        }
        else {
            tiled_shape_.x = cdiv(gemm_shape.x, tile_.x);
            tiled_shape_.y = cdiv(gemm_shape.y, tile_.y);

            cluster_tiles_.x = cdiv(gemm_shape.x, cluster_tile_.x);
            cluster_tiles_.y = cdiv(gemm_shape.y, cluster_tile_.y);

            swizzled_cluster_tiles_ = get_swizzled_shape(cluster_tiles_, log_tile);

            swizzle_tile_x_ = swizzled_cluster_tiles_.x;

            clusters_ = swizzled_cluster_tiles_.x * swizzled_cluster_tiles_.y;
        }
    }

    TM_HOST_DEVICE static int get_log_tile(int2 tiled_mn, int tile_size)
    {
        return gemm::get_log_tile(order == kColMajor ? tiled_mn.y : tiled_mn.x, tile_size);
    }

    TM_HOST_DEVICE static int2 get_swizzled_shape(int2 tiled_shape, int log_tile)
    {
        const int tile = 1 << log_tile;

        if constexpr (order == kColMajor) {
            return {tiled_shape.x * tile, (tiled_shape.y + tile - 1) >> log_tile};
        }
        else {
            return {tiled_shape.y * tile, (tiled_shape.x + tile - 1) >> log_tile};
        }
    }

    TM_DEVICE ProducerState init_producer(Storage& store)
    {
        int cluster_id = 0;
        if constexpr (!is_dynamic) {
            cluster_id = (int)cute::cluster_id_in_grid().x;
        }
        return {
            PipelineState{0, 1, 0},
            0,
            cluster_id,
            store,
            *this,
        };
    }

    TM_DEVICE ConsumerState init_consumer(Storage& store)
    {
        return {
            PipelineState{},
            store,
            *this,
        };
    }

    TM_DEVICE void
    unswizzle(Tile& tile, int cluster_idx, int cta_id, int2 cta_tiles, int2 cluster_tiles, int2 swizzle_tiles) const
    {
        int cluster_idx_x, cluster_idx_y;

        if constexpr (is_grouped_gemm) {
            cluster_idx_x = cluster_idx % swizzle_tiles.x;
            cluster_idx_y = cluster_idx / swizzle_tiles.x;
        }
        else {
            swizzle_tile_x_(cluster_idx_y, cluster_idx_x, cluster_idx);
        }

        auto [cluster_cta_m, cluster_cta_n] = Cluster::cta_mn(cta_id);

        const int offset_x = cluster_cta_m * (striped_m ? cluster_tiles.x : 1);
        const int offset_y = cluster_cta_n * (striped_n ? cluster_tiles.y : 1);

        int2 cluster_tile_offset;

        if constexpr (order == kColMajor) {
            cluster_tile_offset = {(cluster_idx_x >> log_tile_),
                                   (cluster_idx_y << log_tile_) + (cluster_idx_x & ((1 << log_tile_) - 1))};
        }
        else {
            cluster_tile_offset = {(cluster_idx_y << log_tile_) + (cluster_idx_x & ((1 << log_tile_) - 1)),
                                   (cluster_idx_x >> log_tile_)};
        }

        // `tile` may be on DSMEM
        int tile_idx_x        = offset_x + cluster_tile_offset.x * (striped_m ? 1 : Cluster::M);
        int tile_idx_y        = offset_y + cluster_tile_offset.y * (striped_n ? 1 : Cluster::N);
        tile.offset_m         = tile_idx_x * tile_.x;
        tile.offset_n         = tile_idx_y * tile_.y;
        int valid_cluster_p   = cluster_tile_offset.x < cluster_tiles.x && cluster_tile_offset.y < cluster_tiles.y;
        tile.is_valid_cta     = valid_cluster_p && tile_idx_x < cta_tiles.x && tile_idx_y < cta_tiles.y;
        tile.is_valid_cluster = valid_cluster_p;
    }

    TM_DEVICE int get_start_index(int g) const
    {
        // return (__ldg(&offsets_[g]) / (cluster_tile_.x * swizzle_unit_.x) + g) * swizzle_unit_.x
        //        * padded_cluster_tiles_.y;
        return (swizzle_tile_x_.div(__ldg(&offsets_[g])) + g) * swizzle_unit_.x * padded_cluster_tiles_.y;
    }

    TM_DEVICE bool update_sync(int   cluster_idx,
                               int&  group_id_offset,
                               int&  group_idx,
                               int&  group_beg,
                               int&  group_m0,
                               int&  group_m1,
                               int2& tiled_shape,
                               int2& cluster_tiles,
                               int2& swizzled_tiles) const
    {
        const int lane_id = threadIdx.x % WARP_SIZE;

        uint32_t mask;
        while (true) {
            int e    = group_id_offset + lane_id;
            int pred = e > gemm_shape_.w || cluster_idx < get_start_index(e);
            mask     = __ballot_sync((uint32_t)-1, pred);
            if (mask) {
                break;
            }
            group_id_offset += WARP_SIZE;
        }

        // 32 - clz(~mask) - 1
        group_idx = group_id_offset + 31 - __clz(~mask);

        group_m0 = __ldg(&offsets_[group_idx]);
        group_m1 = __ldg(&offsets_[group_idx + 1]);
        int m    = group_m1 - group_m0;

        group_beg = get_start_index(group_idx);

        tiled_shape.x   = cdiv(m, tile_.x);
        cluster_tiles.x = cdiv(m, cluster_tile_.x);

        swizzled_tiles = get_swizzled_shape(cluster_tiles, log_tile_);

        return true;
    }

    TM_DEVICE bool next(ProducerState& state)
    {
        const int lane_id = cutlass::canonical_lane_idx();

        auto& store = state.store;
        auto& pipe  = state.pipe;

        int cluster_idx{};

        if constexpr (is_dynamic) {
            if (lane_id == 0) {
                cutlass::arch::ClusterBarrier::wait(&store.consumer_bar[pipe.index()], pipe.phase());
                cluster_idx = atomicAdd(next_cluster_id_, 1);
            }
            cluster_idx = __shfl_sync((uint32_t)-1, cluster_idx, 0);
        }
        else {
            cutlass::arch::ClusterBarrier::wait(&store.consumer_bar[pipe.index()], pipe.phase());
            cluster_idx = state.cluster_idx;
            state.cluster_idx += (int)cute::cluster_grid_dims().x;
        }

        Tile* tile{};

        if constexpr (Cluster::size == 1) {
            tile = &store.tile[pipe.index()];
        }
        else {
            if (lane_id < Cluster::size) {
                tile = (Tile*)map_to_cta(&store.tile[pipe.index()], lane_id);
            }
        }

        const int alive = cluster_idx < clusters_;

        if (alive) {
            int  group_id      = 0;
            int  group_beg     = 0;
            int  group_m0      = 0;
            int  group_m1      = 0;
            auto cta_tiles     = tiled_shape_;
            auto cluster_tiles = cluster_tiles_;
            auto swizzle_tiles = swizzled_cluster_tiles_;
            if constexpr (is_grouped_gemm) {
                update_sync(cluster_idx,  //
                            state.group_id_offset,
                            group_id,
                            group_beg,
                            group_m0,
                            group_m1,
                            cta_tiles,
                            cluster_tiles,
                            swizzle_tiles);
            }
            if (lane_id < Cluster::size) {
                unswizzle(*tile,  //
                          cluster_idx - group_beg,
                          lane_id,
                          cta_tiles,
                          cluster_tiles,
                          swizzle_tiles);
                if constexpr (is_grouped_gemm) {
                    tile->group_idx = group_id;
                    tile->m0        = group_m0;
                    tile->m1        = group_m1;
                }
            }
        }

        if (lane_id < Cluster::size) {
            tile->alive = alive;
        }

        if constexpr (Cluster::size == 1) {
            if (lane_id == 0) {
                cutlass::arch::ClusterBarrier::arrive(&store.producer_bar[pipe.index()]);
            }
        }
        else {
            mbarrier_arrive_cluster(&store.producer_bar[pipe.index()], lane_id, lane_id < Cluster::size);
        }

        ++pipe;

        return alive;
    }

    TM_DEVICE void tail(ProducerState& state)
    {
        if constexpr (Cluster::size > 1) {
            for (int i = 0; i < Stages; ++i) {
                cutlass::arch::ClusterBarrier::wait(&state.store.consumer_bar[state.pipe.index()], state.pipe.phase());
                ++state.pipe;
            }
        }
    }

    TM_DEVICE bool acquire(ConsumerState& state, Tile*& tile)
    {
        auto& store = state.store;
        auto& pipe  = state.pipe;

        if constexpr (Cluster::size == 1) {
            cutlass::arch::ClusterBarrier::wait(&store.producer_bar[pipe.index()], pipe.phase());
        }
        else {
            mbarrier_wait_cluster(&store.producer_bar[pipe.index()], pipe.phase());
        }

        tile = &store.tile[pipe.index()];

        return tile->alive;
    }

    TM_DEVICE void release(ConsumerState& state, int step)
    {
        auto& store = state.store;
        auto& pipe  = state.pipe;

        __syncwarp();

        if constexpr (Cluster::size == 1) {
            if (cutlass::elect_one_sync()) {
                cutlass::arch::ClusterBarrier::arrive(&store.consumer_bar[pipe.index()]);
            }
        }
        else {
            cutlass::arch::ClusterBarrier::arrive(&store.consumer_bar[pipe.index()], 0, cutlass::elect_one_sync());
        }

        pipe.advance(step);
    }

    TM_DEVICE int4 gemm_shape() const
    {
        return gemm_shape_;
    }

    TM_DEVICE int2 tiled_shape() const
    {
        return tiled_shape_;
    }
};

}  // namespace turbomind::gemm

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

#include <cooperative_groups.h>

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

template<Order order, class Cluster, int striped_m, bool striped_n, int tile_m, int tile_n, bool is_grouped_gemm>
struct TileScheduler {
    int4 gemm_shape_;
    int2 tiled_shape_;
    // int3 tile_shape_;
    int log_tile_;

    int k_iters_;

    int2 tile_offset_;
    int2 iter_k_range_;

    int cluster_idx_;

    static constexpr int2 tile_{tile_m, tile_n};
    static constexpr int2 cluster_tile_{tile_m * Cluster::M, tile_n* Cluster::N};

    int clusters_;

    //////// v2 /////
    int2 swizzle_unit_;
    int2 cluster_tiles_;
    int2 padded_cluster_tiles_;
    int2 swizzled_cluster_tiles_;

    cutlass::FastDivmod swizzle_tile_x_;
    /////////////

    int2 is_valid_;  // {is_valid_cta_tile, is_valid_cluster_tile}

    const int* offsets_;

    int group_idx_ = -1;

    int group_beg_ = 0;
    int group_end_ = 0;

    bool is_next_ = true;

    int* next_cluster_id_;

    static constexpr int Stages = 4;

    using PipelineState = cutlass::PipelineState<Stages>;

    struct Tile {
        int group_idx;
        int is_valid_cta;
        int is_valid_cluster;
        int tile_idx_x;
        int tile_idx_y;
        int dim;
    };

    struct Storage {
        int cluster_idx[Stages];

        __align__(128) uint64_t producer_bar[Stages];
        __align__(128) uint64_t consumer_bar[Stages];
        // Tile tile[Stages];
    };

    // cutlass::FastDivmod swizzled_shape_x;

public:
    TM_DEVICE void init_dyanmic(Storage& storage, int consumer_num)
    {
        for (int i = 0; i < Stages; ++i) {
            cutlass::arch::ClusterBarrier::init(&storage.producer_bar[i], 1);
            cutlass::arch::ClusterBarrier::init(&storage.consumer_bar[i], consumer_num);
        }
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

    TM_DEVICE void grid_init(int n = 1)
    {
        cluster_idx_ = (int)cute::cluster_id_in_grid().x - n * (int)cute::cluster_grid_dims().x;
    }

    TM_DEVICE void unswizzle(int cluster_idx)
    {
        int cluster_idx_x, cluster_idx_y;

        if constexpr (is_grouped_gemm) {
            cluster_idx_x = cluster_idx % swizzled_cluster_tiles_.x;
            cluster_idx_y = cluster_idx / swizzled_cluster_tiles_.x;
        }
        else {
            swizzle_tile_x_(cluster_idx_y, cluster_idx_x, cluster_idx);
        }

        auto [cluster_cta_m, cluster_cta_n] = Cluster::cta_mn(cute::block_id_in_cluster().x);

        const int offset_x = cluster_cta_m * (striped_m ? cluster_tiles_.x : 1);
        const int offset_y = cluster_cta_n * (striped_n ? cluster_tiles_.y : 1);

        int2 cluster_tile_offset;

        if constexpr (order == kColMajor) {
            cluster_tile_offset = {(cluster_idx_x >> log_tile_),
                                   (cluster_idx_y << log_tile_) + (cluster_idx_x & ((1 << log_tile_) - 1))};
        }
        else {
            cluster_tile_offset = {(cluster_idx_y << log_tile_) + (cluster_idx_x & ((1 << log_tile_) - 1)),
                                   (cluster_idx_x >> log_tile_)};
        }

        tile_offset_ = {offset_x + cluster_tile_offset.x * (striped_m ? 1 : Cluster::M),
                        offset_y + cluster_tile_offset.y * (striped_n ? 1 : Cluster::N)};

        iter_k_range_ = {0, k_iters_};

        is_valid_.x = tile_offset_.x < tiled_shape_.x && tile_offset_.y < tiled_shape_.y;
        is_valid_.y = cluster_tile_offset.x < cluster_tiles_.x && cluster_tile_offset.y < cluster_tiles_.y;
    }

    TM_DEVICE int get_start_index(int g)
    {
        // return (__ldg(&offsets_[g]) / (cluster_tile_.x * swizzle_unit_.x) + g) * swizzle_unit_.x
        //        * padded_cluster_tiles_.y;
        return (swizzle_tile_x_.div(__ldg(&offsets_[g])) + g) * swizzle_unit_.x * padded_cluster_tiles_.y;
    }

    TM_DEVICE bool update()
    {
        const int lane_id = threadIdx.x % WARP_SIZE;

        uint32_t mask = 0;

        for (int g = lane_id + group_idx_; mask == 0; g += WARP_SIZE) {
            int pred   = g < gemm_shape_.w && cluster_idx_ < get_start_index(g + 1);
            mask       = __ballot_sync((uint32_t)-1, pred);
            group_idx_ = g;
        }

        // NOTE: __ffs = BREV + FLO
        group_idx_ = __shfl_sync((uint32_t)-1, group_idx_, __ffs(mask) - 1);

        gemm_shape_.x = __ldg(&offsets_[group_idx_ + 1]) - __ldg(&offsets_[group_idx_]);

        group_beg_ = get_start_index(group_idx_);
        group_end_ = get_start_index(group_idx_ + 1);

        tiled_shape_.x   = cdiv(gemm_shape_.x, tile_.x);
        cluster_tiles_.x = cdiv(gemm_shape_.x, cluster_tile_.x);

        swizzled_cluster_tiles_ = get_swizzled_shape(cluster_tiles_, log_tile_);

        return true;
    }

    TM_DEVICE bool next(int n = 1)
    {
        cluster_idx_ += n * (int)cute::cluster_grid_dims().x;

        if (cluster_idx_ >= clusters_) {
            return false;
        }

        if constexpr (is_grouped_gemm) {
            update();
            unswizzle(cluster_idx_ - group_beg_);
        }
        else {
            unswizzle(cluster_idx_);
        }

        return true;
    }

    TM_DEVICE bool produce_next(Storage& store, cutlass::PipelineState<Stages>& pipe)
    {

        if constexpr (Cluster::size == 1) {
            if (threadIdx.x % WARP_SIZE == 0) {
                // printf("PWAIT %d\n", pipe.index());
                cutlass::arch::ClusterBarrier::wait(&store.consumer_bar[pipe.index()], pipe.phase());
                cluster_idx_ = atomicAdd(next_cluster_id_, 1);
                // printf("PGET %d\n", cluster_idx_);
                store.cluster_idx[pipe.index()] = cluster_idx_;
                // printf("PPOST %d\n", pipe.index());
                cutlass::arch::ClusterBarrier::arrive(&store.producer_bar[pipe.index()]);
                ++pipe;
            }
            cluster_idx_ = __shfl_sync((uint32_t)-1, cluster_idx_, 0);
        }
        else {
            if (cute::block_id_in_cluster().x == 0) {
                const int lane_id = cutlass::canonical_lane_idx();
                if (lane_id == 0) {
                    cutlass::arch::ClusterBarrier::wait(&store.consumer_bar[pipe.index()], pipe.phase());
                    cluster_idx_ = atomicAdd(next_cluster_id_, 1);
                    // printf("%d PGET %d\n", clus, cluster_idx_);
                    store.cluster_idx[pipe.index()] = cluster_idx_;
                }
                cluster_idx_  = __shfl_sync((uint32_t)-1, cluster_idx_, 0);
                if (lane_id < Cluster::size) {
                    *(int*)map_to_cta(&store.cluster_idx[pipe.index()], lane_id) = cluster_idx_;
                }
                mbarrier_arrive_cluster(&store.producer_bar[pipe.index()], lane_id, lane_id < Cluster::size);
                ++pipe;
            }
            else {
                return false;
            }
        }

        return cluster_idx_ < clusters_;
    }

    // TM_DEVICE void producer_tail(Storage& store, cutlass::PipelineState<Stages>& pipe, int iters)
    // {
    //     iters = min(iters, 2);
    //     for (int i = 0; i < iters; ++i) {
    //         cutlass::arch::ClusterBarrier::wait(&store.consumer_bar[pipe.index()], pipe.phase());
    //         ++pipe;
    //     }
    // }

    TM_DEVICE bool consume_next(Storage& store, cutlass::PipelineState<Stages>& pipe)
    {
        if constexpr (Cluster::size == 1) {
            // if (threadIdx.x == 256) {
            //     printf("CWAIT %d\n", pipe.index());
            // }
            cutlass::arch::ClusterBarrier::wait(&store.producer_bar[pipe.index()], pipe.phase());
            cluster_idx_ = store.cluster_idx[pipe.index()];
            // if (threadIdx.x == 256) {
            //     printf("CGET %d\n", cluster_idx_);
            //     printf("CPOST %d\n", pipe.index());
            // }
            if (cutlass::elect_one_sync()) {
                cutlass::arch::ClusterBarrier::arrive(&store.consumer_bar[pipe.index()]);
            }
            ++pipe;
        }
        else {
            // cutlass::arch::ClusterBarrier::wait(&store.producer_bar[pipe.index()], pipe.phase());
            mbarrier_wait_cluster(&store.producer_bar[pipe.index()], pipe.phase());
            cluster_idx_ = store.cluster_idx[pipe.index()];
            __syncwarp();
            cutlass::arch::ClusterBarrier::arrive(&store.consumer_bar[pipe.index()], 0, cutlass::elect_one_sync());
            ++pipe;
        }

        // if (threadIdx.x == 0) {
        //     printf("READ %d\n", cluster_idx_);
        // }

        if (cluster_idx_ >= clusters_) {
            return false;
        }

        if constexpr (is_grouped_gemm) {
            update();
            unswizzle(cluster_idx_ - group_beg_);
        }
        else {
            unswizzle(cluster_idx_);
        }

        return true;
    }

    TM_DEVICE explicit operator bool() const
    {
        return cluster_idx_ < clusters_;
    }

    TM_DEVICE int2 is_valid_tile() const
    {
        return is_valid_;
    }

    TM_DEVICE int4 gemm_shape() const
    {
        return gemm_shape_;
    }

    TM_DEVICE int2 tiled_shape() const
    {
        return tiled_shape_;
    }

    TM_DEVICE int2 tile_offset() const
    {
        return tile_offset_;
    }

    TM_DEVICE int2 iter_k_range() const
    {
        return iter_k_range_;
    }

    TM_DEVICE int tile_id() const
    {
        return tile_offset_.x * tiled_shape_.y + tile_offset_.y;
    }
};

}  // namespace turbomind::gemm
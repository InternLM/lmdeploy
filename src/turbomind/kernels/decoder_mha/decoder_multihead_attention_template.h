#pragma once

#include "array_ops.h"
#include "iterator.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "thread_map.h"
#include <cuda_pipeline_primitives.h>

#include "decoder_multihead_attention_params.h"

namespace turbomind {

template<typename T, int HeadPerCta, int MaxHeadDim, int KeyPerIter, int HeadDim, int SliceLen, int Stages>
struct DecoderMultiHeadAttentionKernel {
    using Dtype     = T;
    using ParamType = DecoderMultiHeadAttentionParams<T>;

    static constexpr int kWarpCount  = 4;
    static constexpr int kHeadPerCta = HeadPerCta;
    static constexpr int kMaxHeadDim = MaxHeadDim;
    static constexpr int kKeyPerIter = KeyPerIter;
    static constexpr int kHeadDim    = HeadDim;
    static constexpr int kStages     = Stages;

    static constexpr int kSliceLen     = SliceLen;
    static constexpr int kIterPerSlice = kSliceLen / kKeyPerIter;

    static constexpr int kVecKvSize    = sizeof(uint4) / sizeof(T);
    static constexpr int kThreadPerKey = 8;

    using VecKv      = Array<Dtype, kVecKvSize>;
    using VecKvFloat = Array<float, kVecKvSize>;

    static constexpr bool kUseBlockIter = true;

    using MapKv  = ThreadMapKv<kMaxHeadDim, kKeyPerIter, kVecKvSize, kThreadPerKey, kWarpCount>;
    using IterKv = turbomind::Iterator<T, MapKv, SliceLen, kStages, kUseBlockIter>;

    static size_t GetDynamicSmemSize(int)
    {
        size_t smem_kv_cache = IterKv::kSmemByteSize;
        size_t smem_kv_align = 128;
        size_t smem_qk       = sizeof(float) * kHeadPerCta * kSliceLen;
        size_t smem_pr       = sizeof(float) * kHeadPerCta * kSliceLen;
        return smem_kv_align + smem_kv_cache + std::max(smem_qk, smem_pr);
    }

    // using AccumType   = float;
    // using ComputeType = float;

    using QkAccumType   = float;
    using QkComputeType = float;

    using PvAccumType   = float;
    using PvComputeType = float;

    struct SharedStorage {
        __align__(16) Dtype Q[kHeadPerCta * kMaxHeadDim];
        __align__(16) float O[kHeadPerCta * kMaxHeadDim];
        float M[kHeadPerCta];  // max{dot(Q,  K^T  )}
        float L[kHeadPerCta];  // sum{exp(s - S_max)}
        float red_max[kHeadPerCta * kWarpCount];
        float red_sum[kHeadPerCta * kWarpCount];
    };

    const ParamType& params_;

    int head_idx_;
    int batch_idx_;
    int warp_id_;
    int lane_id_;

    int timestep_;
    T*  k_cache_;  // [S, D]
    T*  v_cache_;  // [S, D]

    const void** k_cache_ptrs_;
    const void** v_cache_ptrs_;

    Dtype* smem_Kv_;
    float* smem_S_;
    float* smem_P_;
    Dtype* smem_Q_;
    float* smem_M_;
    float* smem_L_;
    float* smem_O_;
    float* smem_red_max_;
    float* smem_red_sum_;

    __device__ bool thread0()
    {
        return blockIdx.x == 0 && threadIdx.x == 0;
    }

    __device__ DecoderMultiHeadAttentionKernel(const ParamType& params, SharedStorage& smem, uint8_t* dsmem):
        params_(params)
    {
        smem_Kv_      = (Dtype*)dsmem;
        smem_S_       = (float*)(smem_Kv_ + IterKv::kSizePerTile * kStages);  // [HeadPerCta * kSliceLen]
        smem_P_       = smem_S_;  // ! reusing only works when S and P has same dtype
        smem_Q_       = smem.Q;
        smem_M_       = smem.M;
        smem_L_       = smem.L;
        smem_O_       = smem.O;
        smem_red_max_ = smem.red_max;
        smem_red_sum_ = smem.red_sum;

        head_idx_  = blockIdx.x;
        batch_idx_ = blockIdx.y;
        warp_id_   = threadIdx.x / WARP_SIZE;
        lane_id_   = threadIdx.x % WARP_SIZE;

        timestep_ = params_.per_sample_length[batch_idx_];

        if constexpr (kUseBlockIter) {
            k_cache_ptrs_ = params_.k_cache_block_ptrs + params_.cu_block_cnts[batch_idx_];
            v_cache_ptrs_ = params_.v_cache_block_ptrs + params_.cu_block_cnts[batch_idx_];
        }
        else {
            k_cache_ = (T*)params_.per_sample_k_cache[batch_idx_] + params.layer_offset
                       + head_idx_ * params_.max_seq_len * params_.size_per_head;
            v_cache_ = (T*)params_.per_sample_v_cache[batch_idx_] + params.layer_offset
                       + head_idx_ * params_.max_seq_len * params_.size_per_head;
        }
    }

    // [kkkk][vvvv][kkkk][vvvv][kkkk][vvvv][k][v]
    // __device__ int is_last_iter_of_slice(int iter, int full, int partial)
    // {
    //     if (iter < full) {
    //         return (iter + 1) % kIterPerSlice == 0;
    //     }
    //     else {
    //         return (iter - full + 1) % partial == 0;
    //     }
    // }

    __device__ void Prolugue()
    {
        // - Each warp is handling a row of Q
        // - K/V are loaded redundantly only for the current step
        static_assert(kMaxHeadDim % WARP_SIZE == 0);
        static constexpr int kVecQSize = kMaxHeadDim / WARP_SIZE;

        using VecQ = Array<T, kVecQSize>;

        using MapQ = ThreadMapQ<kMaxHeadDim, kHeadPerCta, kVecQSize, kWarpCount>;

        static constexpr int kQVecPerThread  = MapQ::kIterC;
        static constexpr int kQHeadPerThread = MapQ::kIterS;  // > 1 when #warp < #head

        static_assert(kQVecPerThread == 1);

        int2 offset   = MapQ::get_offset(warp_id_, lane_id_);
        bool is_valid = offset.x < kMaxHeadDim && offset.y < kHeadPerCta;

        if (!is_valid) {
            return;
        }

        VecQ frag_Q[kQHeadPerThread];
        VecQ frag_K;
        VecQ frag_V;

        // load qkv
        PRAGMA_UNROLL
        for (int s = 0; s < kQHeadPerThread; ++s) {
            int di = offset.x;
            int qi = offset.y + s;
            Ldg(frag_Q[s], &params_.q[batch_idx_ * params_.stride + (head_idx_ + qi) * kHeadDim + di]);
        }
        Ldg(frag_K, &params_.k[batch_idx_ * params_.stride + head_idx_ * kHeadDim + offset.x]);
        Ldg(frag_V, &params_.v[batch_idx_ * params_.stride + head_idx_ * kHeadDim + offset.x]);

        if (params_.q_bias) {
            // load biases
            VecQ bias_Q[kQHeadPerThread];
            PRAGMA_UNROLL
            for (int s = 0; s < kQHeadPerThread; ++s) {
                int di = offset.x;
                int qi = offset.y + s;
                Ldg(bias_Q[s], &params_.q_bias[(head_idx_ + qi) * kHeadDim + di]);
            }
            VecQ bias_K;
            VecQ bias_V;
            Ldg(bias_K, &params_.k_bias[head_idx_ * kHeadDim + offset.x]);
            Ldg(bias_V, &params_.v_bias[head_idx_ * kHeadDim + offset.x]);

            using namespace ops;
            // apply biases
            PRAGMA_UNROLL
            for (int s = 0; s < kQHeadPerThread; ++s) {
                frag_Q[s] = frag_Q[s] + bias_Q[s];
            }
            frag_K = frag_K + bias_K;
            frag_V = frag_V + bias_V;
        }

        // Apply rotary embedding
        RotaryEmbedding<kVecQSize> rotary_emb(
            params_.rotary_embedding_base, params_.rotary_embedding_dim, timestep_, offset);

        PRAGMA_UNROLL
        for (int s = 0; s < kQHeadPerThread; ++s) {
            rotary_emb.apply(frag_Q[s]);
        }
        rotary_emb.apply(frag_K);

        PRAGMA_UNROLL
        for (int s = 0; s < kQHeadPerThread; ++s) {
            int         qi = offset.y + s;
            QkAccumType qk = qk_dot<QkAccumType, QkComputeType, WARP_SIZE>(frag_Q[s], frag_K);
            if (lane_id_ == 0) {
                qk *= params_.inv_sqrt_dh;
                // printf("qk_last[%d]=%f\n", head_idx_, qk);
                smem_M_[qi] = qk;
                smem_L_[qi] = 1.f;
            }
            // write Q and O
            Store(&smem_Q_[qi * kMaxHeadDim + offset.x], frag_Q[s]);
            Store(&smem_O_[qi * kMaxHeadDim + offset.x], cast<float>(frag_V));
        }

        // store
        if (warp_id_ == 0) {
            if constexpr (kUseBlockIter) {
                int block_index  = timestep_ / params_.kv_cache_block_size;
                int block_offset = timestep_ % params_.kv_cache_block_size;
                // if (thread0()) {
                //     printf("%d %d %p %p\n", block_index, block_offset, k_cache_ptrs_, v_cache_ptrs_);
                // }
                k_cache_ = (T*)k_cache_ptrs_[block_index] + params_.layer_offset
                           + head_idx_ * params_.kv_cache_block_size * kHeadDim;
                v_cache_ = (T*)v_cache_ptrs_[block_index] + params_.layer_offset
                           + head_idx_ * params_.kv_cache_block_size * kHeadDim;
                Store(&k_cache_[block_offset * kHeadDim + offset.x], frag_K);
                Store(&v_cache_[block_offset * kHeadDim + offset.x], frag_V);
            }
            else {
                Store(&k_cache_[timestep_ * kHeadDim + offset.x], frag_K);
                Store(&v_cache_[timestep_ * kHeadDim + offset.x], frag_V);
            }
        }
    }

    __device__ void PrefetchKvCache(IterKv& iter)
    {
        PRAGMA_UNROLL
        for (int stage = 0; stage < kStages - 1; ++stage) {
            iter.PrefetchStage();
            CpAsyncCommit();
        }
    }

    __device__ void CpAsyncWait()
    {
        __pipeline_wait_prior(kStages - 2);
        // __syncwarp();
        // __syncthreads();
    }

    __device__ void CpAsyncCommit()
    {
        __pipeline_commit();
    }

    __device__ void CpAsyncFlush()
    {
        __pipeline_commit();
        __pipeline_wait_prior(0);
    }

    static constexpr int kKvVecPerThread = MapKv::kIterC;
    static constexpr int kKvKeyPerThread = MapKv::kIterS;

    struct FragmentQ {
        VecKv data[kHeadPerCta][kKvVecPerThread];
    };

    struct State {
        // Double buffering to hide smem/dequant latency
        VecKv frag_Kv_buf[2][kKvVecPerThread];
    };

    static constexpr int kPrefetchCount = (IterKv::kIterCount + MapKv::kIterS - 1) / MapKv::kIterS;

    __device__ void ComputeSlice(FragmentQ& frag_Q, State& state, const int2& offset, int step, int iter_length)
    {

        Array<float, kHeadPerCta> frag_M;
        PRAGMA_UNROLL
        for (int i = 0; i < kHeadPerCta; ++i) {
            frag_M[i] = smem_M_[i];
        }

        IterKv iter_K;

        if constexpr (kUseBlockIter) {
            iter_K = {k_cache_ptrs_,
                      params_.kv_cache_block_size,
                      params_.layer_offset,
                      head_idx_,
                      smem_Kv_,
                      step,
                      step + iter_length,
                      warp_id_,
                      lane_id_};
        }
        else {
            iter_K = {k_cache_, smem_Kv_, step, step + iter_length, warp_id_, lane_id_};
        }

        PrefetchKvCache(iter_K);
        CpAsyncWait();

        iter_K.Load(state.frag_Kv_buf[0]);
        iter_K.PrefetchBatch(0, kPrefetchCount);
        if (kKvKeyPerThread == 1) {
            CpAsyncCommit();
            CpAsyncWait();
            iter_K.AdvancePrefetchStage();
            iter_K.AdvanceComputeStage();
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        /// Compute QK(Q, S) = Q(Q, D) * K^T(D, S)

        PRAGMA_NO_UNROLL
        for (int _it = 0; _it < iter_length; _it += kKeyPerIter) {
            PRAGMA_UNROLL
            for (int si = 0; si < kKvKeyPerThread; ++si) {
                // smem -> rmem for next iter
                iter_K.Load(state.frag_Kv_buf[(si + 1) % 2]);

                // current iter's K fragment
                auto& frag_K = state.frag_Kv_buf[si % 2];

                const int local_offset = offset.y + _it + si * MapKv::kWarpAccessS;

                PRAGMA_UNROLL
                for (int qi = 0; qi < kHeadPerCta; ++qi) {

                    auto qk = qk_dot<QkAccumType, QkComputeType, kThreadPerKey>(frag_Q.data[qi], frag_K);

                    // if (ti == 16) {
                    //     for (int vi = 0; vi < kKvVecPerThread; ++vi) {
                    //         for (int i = 0; i < kVecKvSize; ++i) {
                    //             printf("frag_Q = %f, frag_K[%d] = %f\n",
                    //                    (float)frag_Q.data[qi][vi][i],
                    //                    offset.x + vi * kVecKvSize + i,
                    //                    (float)frag_K[vi][i]);
                    //         }
                    //     }
                    // }

                    qk *= params_.inv_sqrt_dh;

                    if (step + local_offset < timestep_) {

                        // group leader writes to smem
                        if (threadIdx.x % kThreadPerKey == 0) {
                            // printf("qk_%d = %f\n", step + local_offset, (float)qk);

                            smem_S_[kSliceLen * qi + local_offset] = qk;

                            // local max
                            frag_M[qi] = fmaxf(frag_M[qi], qk);
                        }
                    }
                }

                iter_K.PrefetchBatch((si + 1) % kKvKeyPerThread, kPrefetchCount);

                if (kKvKeyPerThread == 1 || si == kKvKeyPerThread - 2) {
                    CpAsyncCommit();
                    CpAsyncWait();
                    iter_K.AdvancePrefetchStage();
                    iter_K.AdvanceComputeStage();
                }
            }

            // handle special case
            if (kKvKeyPerThread == 1) {
                for (int vi = 0; vi < kKvVecPerThread; ++vi) {
                    state.frag_Kv_buf[0][vi] = state.frag_Kv_buf[1][vi];
                }
            }
        }

        CpAsyncFlush();

        __syncthreads();

        Array<float, kHeadPerCta> exp_M_diff;
        PRAGMA_UNROLL
        for (int i = 0; i < kHeadPerCta; ++i) {
            exp_M_diff[i] = smem_M_[i];
        }

        /// block synchronization
        frag_M = qk_max<MapKv>(frag_M, smem_red_max_, warp_id_, lane_id_);

        if (threadIdx.x == 0 && step == timestep_ - kSliceLen) {
            // printf("frag_M[%d] = %f\n", head_idx_, (float)frag_M[0]);
        }

        // wait while smem_red_ is being used.
        // __syncthreads();

        PRAGMA_UNROLL
        for (int i = 0; i < kHeadPerCta; ++i) {
            // if (thread0()) {
            //     printf("%f %f %f\n", (float)exp_M_diff[i], (float)frag_M[i], (float)__expf(exp_M_diff[i] -
            //     frag_M[i]));
            // }
            // exp(m1 - m2)
            exp_M_diff[i] = __expf(exp_M_diff[i] - frag_M[i]);

            if (threadIdx.x == 0) {
                smem_M_[i] = frag_M[i];
            }
        }

        // __syncthreads();  // DEBUG

        /////////////////////////////////////////////////////////////////////////////////////////
        // / Compute softmax P(Q, S)
        Array<float, kHeadPerCta> frag_L{};

        for (int ti = threadIdx.x; ti < iter_length; ti += kWarpCount * WARP_SIZE) {
            PRAGMA_UNROLL
            for (int qi = 0; qi < kHeadPerCta; ++qi) {
                int   idx = qi * kSliceLen + ti;
                float qk  = smem_S_[idx];
                float pr  = expf(qk - frag_M[qi]);
                // printf("smem_P[%d] = %f\n", ti, pr);
                smem_P_[idx] = pr;
                frag_L[qi] += pr;
            }
        }

        // if (thread0()) {
        // printf("frag_L0 = %f\n", (float)frag_L[0]);
        // }

        /// block synchronization
        frag_L = blockSum<kWarpCount>(frag_L, smem_red_sum_, warp_id_, lane_id_);

        if (thread0()) {
            // printf("frag_L = %f\n", (float)frag_L[0]);
        }

        for (int qi = 0; qi < kHeadPerCta; ++qi) {
            // exp(m1 - m2) * l1
            frag_L[qi] += exp_M_diff[qi] * smem_L_[qi];
        }

        __syncthreads();

        for (int qi = 0; qi < kHeadPerCta; ++qi) {
            if (threadIdx.x == 0) {
                smem_L_[qi] = frag_L[qi];
            }
        }

        if (threadIdx.x == 0 && step == timestep_ - kSliceLen) {
            // printf("frag_L'[%d] = %f\n", head_idx_, (float)frag_L[0]);
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        // / Compute O[H,D] = P[H,S] * V[S,D]
        VecKvFloat frag_O[kHeadPerCta][kKvVecPerThread]{};  // value initialize
                                                            // float      frag_Pr_buf[2][kHeadPerCta];

        // ti = step + offset.y;

        // int ti = step + offset.y;

        // PRAGMA_UNROLL
        // for (int qi = 0; qi < kHeadPerCta; ++qi) {
        //     // prefetch Pr for first warp iter
        //     frag_Pr_buf[0][qi] = smem_P_[qi * kSliceLen + ti];
        // }

        IterKv iter_V;

        if constexpr (kUseBlockIter) {
            iter_V = {v_cache_ptrs_,
                      params_.kv_cache_block_size,
                      params_.layer_offset,
                      head_idx_,
                      smem_Kv_,
                      step,
                      step + iter_length,
                      warp_id_,
                      lane_id_};
        }
        else {
            iter_V = {v_cache_, smem_Kv_, step, step + iter_length, warp_id_, lane_id_};
        }

        PrefetchKvCache(iter_V);
        CpAsyncWait();

        iter_V.Load(state.frag_Kv_buf[0]);
        iter_V.PrefetchBatch(0, kPrefetchCount);
        if (kKvKeyPerThread == 1) {
            CpAsyncCommit();
            CpAsyncWait();
            iter_V.AdvancePrefetchStage();
            iter_V.AdvanceComputeStage();
        }

        PRAGMA_NO_UNROLL
        for (int _it = 0; _it < iter_length; _it += kKeyPerIter) {
            PRAGMA_UNROLL
            for (int si = 0; si < kKvKeyPerThread; ++si) {
                // Load value cache for next warp iter
                iter_V.Load(state.frag_Kv_buf[(si + 1) % 2]);

                // Load Pr for next warp iter
                // PRAGMA_UNROLL
                // for (int qi = 0; qi < kHeadPerCta; ++qi) {
                //     frag_Pr_buf[(si + 1) % 2][qi] = smem_P_[qi * kSliceLen + (ti + MapKv::kWarpAccessS)];
                // }

                auto& frag_V = state.frag_Kv_buf[si % 2];
                // auto& frag_P = frag_Pr_buf[si % 2];

                const int local_offset = offset.y + _it + si * MapKv::kWarpAccessS;

                float frag_P[kHeadPerCta];
                PRAGMA_UNROLL
                for (int qi = 0; qi < kHeadPerCta; ++qi) {
                    frag_P[qi] = smem_P_[qi * kSliceLen + local_offset];
                }

                if (step + local_offset < timestep_) {
                    PRAGMA_UNROLL
                    for (int qi = 0; qi < kHeadPerCta; ++qi) {
                        fma_pv<PvComputeType>(frag_P[qi], frag_V, frag_O[qi]);
                    }
                    // for (int i = 0; i < kKvVecPerThread; ++i) {
                    //     for (int j = 0; j < kVecKvSize; ++j) {
                    //         printf("frag_V %f\n", (float)frag_V[i][j]);
                    //     }
                    // }
                    // if (threadIdx.x % MapKv::kWarpThreadC == 0) {
                    //     printf("frag_P[%d] %f\n", ti, frag_P[0]);
                    // }
                }

                iter_V.PrefetchBatch((si + 1) % kKvKeyPerThread, kPrefetchCount);

                if (kKvKeyPerThread == 1 || si == kKvKeyPerThread - 2) {
                    CpAsyncCommit();
                    CpAsyncWait();
                    iter_V.AdvancePrefetchStage();
                    iter_V.AdvanceComputeStage();
                }
            }

            // handle special case
            if (kKvKeyPerThread == 1) {
                for (int vi = 0; vi < kKvVecPerThread; ++vi) {
                    state.frag_Kv_buf[0][vi] = state.frag_Kv_buf[1][vi];
                }
                // PRAGMA_UNROLL
                // for (int qi = 0; qi < kHeadPerCta; ++qi) {
                //     frag_Pr_buf[0][qi] = frag_Pr_buf[1][qi];
                // }
            }
        }

        /// warp reduce over S dim
        PRAGMA_UNROLL
        for (int qi = 0; qi < kHeadPerCta; ++qi) {
            PRAGMA_UNROLL
            for (int vi = 0; vi < kKvVecPerThread; ++vi) {
                PRAGMA_UNROLL
                for (int i = 0; i < kVecKvSize; ++i) {
                    // reduce over warp thread S
                    PRAGMA_UNROLL
                    for (int mask = WARP_SIZE / 2; mask >= MapKv::kWarpThreadC; mask /= 2) {
                        frag_O[qi][vi][i] += __shfl_xor_sync(uint32_t(-1), frag_O[qi][vi][i], mask);
                    }
                }
            }
        }

        // __syncthreads();

        PRAGMA_UNROLL
        for (int gi = 0; gi < MapKv::kS; gi += MapKv::kFootprintS) {
            PRAGMA_UNROLL
            for (int qi = 0; qi < kHeadPerCta; ++qi) {
                PRAGMA_UNROLL
                for (int vi = 0; vi < kKvVecPerThread; ++vi) {
                    if (offset.y == gi) {
                        // ! 2-way bank conflict
                        auto& smem_O = (VecKvFloat&)smem_O_[qi * kMaxHeadDim + offset.x + vi * MapKv::kDeltaC];
                        using namespace ops;
                        auto tmp_O = smem_O;
                        if (offset.y == 0) {
                            tmp_O = tmp_O * exp_M_diff[qi];
                        }
                        // ! 2-way bank conflict
                        smem_O = tmp_O + frag_O[qi][vi];
                    }
                }
            }
            __syncthreads();
        }

        CpAsyncFlush();
    }

    __device__ void LoopKv()
    {
        const int2 offset = MapKv::get_offset(warp_id_, lane_id_);

        ///////////////////////////////////////////////////////////////////////////////////////////
        /// Load Q from shared memory.
        /// NOTE: There will be bank-conflict when sizeof(VecKv) > 16 (e.g. KV is quantized)
        FragmentQ frag_Q;

        PRAGMA_UNROLL
        for (int qi = 0; qi < kHeadPerCta; ++qi) {
            PRAGMA_UNROLL
            for (int c = 0; c < kKvVecPerThread; ++c) {
                const int di       = offset.x + MapKv::kDeltaC * c;
                frag_Q.data[qi][c] = (VecKv&)smem_Q_[qi * kMaxHeadDim + di];
            }
        }

        State state;

        PRAGMA_NO_UNROLL
        for (int step = 0; step < timestep_; step += kSliceLen) {
            int iter_length = min(timestep_ - step, kSliceLen);
            ComputeSlice(frag_Q, state, offset, step, iter_length);
        }
    }

    __device__ void Run()
    {
        if constexpr (0) {
            for (int i = threadIdx.x; i < kStages * IterKv::kSizePerTile; i += blockDim.x) {
                smem_Kv_[i] = Dtype(0);
            }
            __syncthreads();
        }

        // early exit if finished flag is set
        if (params_.finished[batch_idx_]) {
            return;
        }

        // Compute attention for current step
        Prolugue();

        __syncthreads();

        // Iterate over K/V
        LoopKv();

        __syncthreads();

        // Normalize outputs & write to device memory
        Epilogue();
    }

    __device__ void Epilogue()
    {
        static constexpr int kVecQSize = kMaxHeadDim / WARP_SIZE;

        using VecQ      = Array<T, kVecQSize>;
        using VecQFloat = Array<float, kVecQSize>;

        using MapQ = ThreadMapQ<kMaxHeadDim, kHeadPerCta, kVecQSize, kWarpCount>;

        static constexpr int kQkvHeadPerThread = MapQ::kIterS;
        static_assert(kQkvHeadPerThread == 1);

        int2 offset = MapQ::get_offset(warp_id_, lane_id_);

        bool is_valid = offset.x < kMaxHeadDim && offset.y < kHeadPerCta;
        if (!is_valid) {
            return;
        }

        PRAGMA_UNROLL
        for (int s = 0; s < kQkvHeadPerThread; ++s) {
            int   di    = offset.x;
            int   qi    = offset.y + s;
            float scale = __fdividef(1.f, smem_L_[qi] + 1e-6f);
            // float scale = 1.f;
            using namespace ops;
            VecQFloat frag_O = (VecQFloat&)smem_O_[qi * kMaxHeadDim + di] * scale;
            /// FIXME: `(head_idx_ + qi)` doesn't look right
            Store(&params_.out[batch_idx_ * params_.num_heads * kHeadDim + (head_idx_ + qi) * kHeadDim + di],
                  cast<Dtype>(frag_O));
        }
    }
};

extern __shared__ uint8_t dynamic_smem[];

template<typename MHAType, typename ParamType = typename MHAType::ParamType>
__global__ void decoder_multihead_attention(ParamType params)
{
    __shared__ typename MHAType::SharedStorage shared_storage;

    uint8_t* smem_ptr = dynamic_smem;

    // Align dynamic smem ptr to 128 byte boundary, this eliminates excessive wavefronts from smem to L1
    // but it does not improve performance
    if constexpr (0) {
        int misalign = (uintptr_t)smem_ptr % 128;
        if (misalign) {
            smem_ptr += 128 - misalign;
        }
    }

    MHAType{params, shared_storage, smem_ptr}.Run();
}

}  // namespace turbomind

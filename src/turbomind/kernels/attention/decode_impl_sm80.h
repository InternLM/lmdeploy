#include "array_ops.h"
#include "attention_impl.h"
#include "iterator_sm80.h"
#include "src/turbomind/kernels/custom_ar_kernels.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

#include "thread_map.h"

namespace turbomind {

struct sm80_decode_t {};

template<class T, class Tkv, class BlockSeqLen, int CTA_Q, int CTA_S, int HeadDim>
struct AttentionImpl<sm80_decode_t, T, Tkv, BlockSeqLen, CTA_Q, CTA_S, HeadDim> {

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 8;
    static constexpr int OP_K = 16;

    static constexpr int WARP_S = OP_M;  // 16
    static constexpr int WARP_Q = OP_N;  // 8

    static constexpr int kWarpCntQ = CTA_Q / WARP_Q;
    static constexpr int kWarpCntS = CTA_S / WARP_S;

    static constexpr int kWarpCount = kWarpCntS * kWarpCntQ;

    static constexpr int kM = WARP_S / OP_M;   // 1
    static constexpr int kN = WARP_Q / OP_N;   // 1
    static constexpr int kK = HeadDim / OP_K;  // 8

    static constexpr int vM = HeadDim / OP_M;  // 8
    static constexpr int vN = WARP_Q / OP_N;   // 1
    static constexpr int vK = WARP_S / OP_K;   // 1

    using FragK = Array<T, 8>[kK][kM];      // (s8,d4) (Dk,Sm) (d2,s2,d2)
                                            //   1  2   16 16    8  8  1
    using FragQ = Array<T, 4>[kN][kK];      // (q8,d4) (Qn,Dk) (d2,d2)
                                            //   1  2    8 16    8  1
    using FragS = Array<float, 4>[kM][kN];  // (s8,q4) (Sm,Qn) (s2,q2)
                                            //   1  2   16  8    8  1
    using FragV = Array<T, 8>[kK][kM];      // (d8,s4) (Sk,Dm) (s2,d2,s2)
                                            //   1  2   16 16    8  8  1
    using FragP = Array<T, 4>[vK][vN];      // (q8,s4) (Sk,Qn) (s2,s2)
                                            //   8  2   16  8    8  1
    using FragO = Array<float, 4>[vM][vN];  // (d8,q4) (Dm,Qn) (d2,q2)
                                            //   1  2   16  8    8  1
    using FragM = Array<float, 2>[kN];      // (_8,q4)    (Qn)    (q2)
    using FragL = FragM;

    using SmemM = Array<float, 2>[kN][kWarpCntQ][kWarpCntS][4];
    using SmemO = Array<float, 4>[kM][kN][kWarpCntQ][WARP_SIZE];

    struct Swizzle {
        __device__ int operator()(int index) const
        {
            // sssSSSdDDDddd
            // DDD ^= SSS
            constexpr int mask = 0x7 << 7;
            return index ^ ((index & mask) >> 4);
        }
    };

    using SwizzleQ = Swizzle;
    using SwizzleK = Swizzle;
    using SwizzleV = Swizzle;

    static constexpr int kPadQ = 0;
    static constexpr int kPadK = 0;
    static constexpr int kPadP = 0;
    static constexpr int kPadV = 0;

    static constexpr bool kUseSmemQ = false;
    static constexpr bool kUseSmemP = false;

    struct SharedStorage {
        union {
            __align__(16) T smem_Q[CTA_Q][HeadDim + kPadQ];
            struct {
                __align__(16) Tkv smem_K[CTA_S][HeadDim + kPadK];
                __align__(16) Tkv smem_V[CTA_S][HeadDim + kPadV];
            };
            struct {
                __align__(16) SmemM smem_M;
                __align__(16) SmemO smem_O;
            };
        };
    };

    using ThreadMap = RakedThreadMap<HeadDim, CTA_S, sizeof(uint4) / sizeof(T), kWarpCount>;

    using SmemLayoutK = SmemLayout<HeadDim, Swizzle>;
    using SmemLayoutV = SmemLayout<HeadDim, Swizzle>;

    using GmemIterK = Sm80GmemIterator<Tkv, ThreadMap, BlockSeqLen, SmemLayoutK, 2>;
    using GmemIterV = Sm80GmemIterator<Tkv, ThreadMap, BlockSeqLen, SmemLayoutV, 2>;

    using SmemIterK = Sm80SmemIterK<Tkv, SmemLayoutK, kN>;
    using SmemIterV = Sm80SmemIterV<Tkv, SmemLayoutV, vN>;

    template<class Fragment, class Func>
    __device__ static void ForeachS(Fragment& S, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < kM; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < kN; ++n) {
                PRAGMA_UNROLL
                for (int s = 0; s < 2; ++s) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        const int si = m * OP_M + lane_id / 4 + s * 8 + warp_id / kWarpCntQ * WARP_S;
                        const int qi = n * OP_N + lane_id % 4 + q * 1 + warp_id % kWarpCntQ * WARP_Q;
                        ((Func&&)func)(qi, si, S[m][n][s * 2 + q]);
                    }
                }
            }
        }
    }

    __device__ void TransformQ(const T* smem_Q, FragQ& frag_Q)
    {
        if constexpr (!kUseSmemQ) {
            uint32_t  smem_int_ptr = cast_smem_ptr_to_uint(smem_Q);
            const int warp_id      = threadIdx.x / WARP_SIZE;
            const int lane_id      = threadIdx.x % WARP_SIZE;
            SwizzleQ  swizzle;
            PRAGMA_UNROLL
            for (int n = 0; n < kN; ++n) {
                PRAGMA_UNROLL
                for (int k = 0; k < kK; k += 2) {
                    auto&     Q   = (Array<uint32_t, 4>&)frag_Q[n][k];
                    const int qi  = n * OP_N + lane_id % 8 + warp_id % kWarpCntQ * WARP_Q;
                    const int di  = k * OP_K + lane_id / 8 * 8;
                    const int idx = swizzle(qi * (HeadDim + kPadQ) + di);
                    ldmatrix_m8n8_x4_b16(Q[0], Q[1], Q[2], Q[3], smem_int_ptr + sizeof(T) * idx);
                }
            }
        }
    }

    template<class SmemQ, class SmemK>
    __device__ static void ComputeQK(SmemQ& smem_Q, SmemK& smem_K, FragQ& frag_Q, FragS& frag_S)
    {
        FragK frag_K;
        smem_K.Load(frag_K[0], 0);
        PRAGMA_UNROLL
        for (int k = 0; k < kK; ++k) {
            if (k < kK - 1) {
                smem_K.Load(frag_K[k + 1], k + 1);
            }
            PRAGMA_UNROLL
            for (int m = 0; m < kM; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < kN; ++n) {
                    mma_m16n8k16_row_col(frag_S[m][n], frag_K[k][m], frag_Q[n][k], frag_S[m][n]);
                }
            }
        }
    }

    template<class SmemP, class SmemV>
    __device__ static void ComputePV(SmemP& smem_P, SmemV& smem_V, FragP& frag_P, FragO& frag_O)
    {
        FragV frag_V;
        smem_V.Load(frag_V[0], 0);
        PRAGMA_UNROLL
        for (int k = 0; k < vK; ++k) {
            if (k < vK - 1) {
                smem_V.Load(frag_V[k + 1], k + 1);
            }
            PRAGMA_UNROLL
            for (int m = 0; m < vM; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < vN; ++n) {
                    mma_m16n8k16_row_col(frag_O[m][n], frag_V[k][m], frag_P[n][k], frag_O[m][n]);
                }
            }
        }
    }

    template<bool is_residue>
    __device__ static void
    Softmax(FragS& frag_S, FragM& frag_M, FragL& frag_L, FragO& frag_O, float qk_scale, T* smem_P, FragP& frag_P)
    {
        FragM prev_M;
        copy(frag_M, prev_M);

        PRAGMA_UNROLL
        for (int n = 0; n < kN; ++n) {
            PRAGMA_UNROLL
            for (int m = 0; m < kM; ++m) {
                PRAGMA_UNROLL
                for (int s = 0; s < 2; ++s) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        frag_M[n][q] = fmaxf(frag_M[n][q], frag_S[m][n][s * 2 + q]);
                    }
                }
            }
        }

        PRAGMA_UNROLL
        for (int n = 0; n < kN; ++n) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                frag_M[n][q] = fmaxf(frag_M[n][q], __shfl_xor_sync(uint32_t(-1), frag_M[n][q], 4));
                frag_M[n][q] = fmaxf(frag_M[n][q], __shfl_xor_sync(uint32_t(-1), frag_M[n][q], 8));
                frag_M[n][q] = fmaxf(frag_M[n][q], __shfl_xor_sync(uint32_t(-1), frag_M[n][q], 16));
            }
        }

        PRAGMA_UNROLL
        for (int n = 0; n < kN; ++n) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                float expdiff_M = exp2f((prev_M[n][q] - frag_M[n][q]) * qk_scale);
                if (is_residue && frag_M[n][q] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M = 0.f;
                }
                PRAGMA_UNROLL
                for (int m = 0; m < kM; ++m) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        frag_O[m][n][s * 2 + q] *= expdiff_M;
                    }
                }
                frag_L[n][q] *= expdiff_M;
            }
        }

        PRAGMA_UNROLL
        for (int n = 0; n < kN; ++n) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                float tmp_L{};
                PRAGMA_UNROLL
                for (int m = 0; m < kM; ++m) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        float p = exp2f(frag_S[m][n][s * 2 + q] * qk_scale - frag_M[n][q] * qk_scale);
                        if (is_residue && frag_M[n][q] == -std::numeric_limits<float>::infinity()) {
                            p = 0.f;
                        }
                        tmp_L += p;
                        frag_S[m][n][s * 2 + q] = p;
                    }
                }
                tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 4);
                tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 8);
                tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 16);
                frag_L[n][q] = frag_L[n][q] + tmp_L;
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < kM; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < kN; ++n) {
                PRAGMA_UNROLL
                for (int s = 0; s < 2; ++s) {
                    Array<T, 2> tmp_P;
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        tmp_P[q] = static_cast<T>(frag_S[m][n][s * 2 + q]);
                    }
                    // (s8,q4),(q2) -> (q8,s4),(s2)
                    transpose_m8n8_b16((uint&)tmp_P, 0);
                    (Array<T, 2>&)frag_P[m][n][s * 2] = tmp_P;
                }
            }
        }
    }

    template<class Func>
    __device__ static void
    StoreO(FragO& frag_O, FragL& frag_L, Func&& func, FragM& frag_M, SmemM& smem_M, SmemO& smem_O, float qk_scale)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_q = warp_id / kWarpCntS;

        if constexpr (kWarpCntS > 1) {

            const int warp_s = warp_id % kWarpCntS;

            // Store local maximum to shared memory
            PRAGMA_UNROLL
            for (int n = 0; n < kN; ++n) {
                if (lane_id < 4) {
                    Store(&smem_M[n][warp_q][warp_s][lane_id], frag_M[n]);
                }
            }

            __syncthreads();

            FragM prev_M;
            copy(frag_M, prev_M);

            PRAGMA_UNROLL
            for (int n = 0; n < kN; ++n) {
                // Compute global maximum
                PRAGMA_UNROLL
                for (int s = 0; s < kWarpCntS - 1; ++s) {
                    Array<float, 2> tmp_M = smem_M[n][warp_q][(warp_s + s + 1) % kWarpCntS][lane_id % 4];
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        frag_M[n][q] = fmaxf(frag_M[n][q], tmp_M[q]);
                    }
                }
                // Scale with global maximum
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    float expdiff_M = exp2f((prev_M[n][q] - frag_M[n][q]) * qk_scale);
                    PRAGMA_UNROLL
                    for (int m = 0; m < vM; ++m) {
                        PRAGMA_UNROLL
                        for (int d = 0; d < 2; ++d) {
                            frag_O[m][n][d * 2 + q] *= expdiff_M;  // rescore O
                        }
                    }
                    frag_L[n][q] *= expdiff_M;  // rescale L
                }
            }

            __syncthreads();

            PRAGMA_UNROLL
            for (int s = 0; s < kWarpCntS; ++s) {
                PRAGMA_UNROLL
                for (int n = 0; n < vN; ++n) {
                    if (warp_s == s) {
                        using namespace ops;
                        PRAGMA_UNROLL
                        for (int m = 0; m < vM; ++m) {
                            if (s == 0) {
                                clear(smem_O[m][n][warp_q][lane_id]);
                            }
                            smem_O[m][n][warp_q][lane_id] = smem_O[m][n][warp_q][lane_id] + frag_O[m][n];
                        }
                        if (lane_id < 4) {
                            if (s == 0) {
                                clear(smem_M[n][warp_q][0][lane_id]);
                            }
                            smem_M[n][warp_q][0][lane_id] = smem_M[n][warp_q][0][lane_id] + frag_L[n];
                        }
                    }
                }
                __syncthreads();
            }

            if (warp_s == 0) {
                PRAGMA_UNROLL
                for (int n = 0; n < vN; ++n) {
                    Lds(frag_L[n], smem_M[n][warp_q][0][lane_id % 4]);
                    PRAGMA_UNROLL
                    for (int m = 0; m < vM; ++m) {
                        Lds(frag_O[m][n], smem_O[n][n][warp_q][lane_id]);
                    }
                }
            }
            else {
                return;
            }
        }

        FragL tmp_L;
        PRAGMA_UNROLL
        for (int n = 0; n < vN; ++n) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                tmp_L[n][q] = fdividef(1.f, frag_L[n][q] + 1e-8f);
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < vM; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < vN; ++n) {
                PRAGMA_UNROLL
                for (int d = 0; d < 2; ++d) {
                    Array<T, 2> tmp_O;
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        tmp_O = frag_O[m][n][d * 2 + q] * tmp_L[n][q];
                    }
                    // (d8,q4),(q2) -> (q8,d4),(d2)
                    (uint&)tmp_O = transpose_m8n8_b16((uint&)tmp_O, 0);
                    const int qi = n * OP_N + lane_id / 4 + warp_q * WARP_Q;
                    const int di = m * OP_M + lane_id % 4 * 2 + d * 8;
                    ((Func&&)func)(qi, di, tmp_O);
                }
            }
        }
    }
};

}  // namespace turbomind
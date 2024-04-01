#include "array_ops.h"
#include "impl.h"
#include "iterator.h"
#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "thread_map.h"
#include <type_traits>

namespace turbomind::attention {

template<class T_,
         class Tkv_,
         int CTA_H_,
         int CTA_Q_,
         int CTA_S_,
         int WARP_H_,
         int WARP_Q,
         int WARP_S,
         int HeadDim,
         int Stages>
struct Impl<MMA_81616, T_, Tkv_, CTA_H_, CTA_Q_, CTA_S_, WARP_H_, WARP_Q, WARP_S, HeadDim, Stages> {
    using T   = T_;
    using Tkv = Tkv_;

    static constexpr int kQuantKV = !std::is_same_v<T, Tkv>;

    static constexpr int CTA_H = CTA_H_;
    static constexpr int CTA_Q = CTA_Q_;
    static constexpr int CTA_S = CTA_S_;

    static_assert(CTA_Q == 1);

    static constexpr int WARP_H = WARP_H_;

    static constexpr int kHeadDim = HeadDim;

    static constexpr int kWarpCntH = CTA_H / WARP_H;
    static constexpr int kWarpCntQ = CTA_Q / WARP_Q;
    static constexpr int kWarpCntS = CTA_S / WARP_S;

    static constexpr int kWarpCount = kWarpCntQ * kWarpCntS;

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 8;
    static constexpr int OP_K = 16;

    static constexpr int K_M = WARP_S / OP_M;               // 1
    static constexpr int K_N = (WARP_H + OP_N - 1) / OP_N;  // 1
    static constexpr int K_K = HeadDim / OP_K;              // 8

    static constexpr int V_M = HeadDim / OP_M;              // 8
    static constexpr int V_N = (WARP_H + OP_N - 1) / OP_N;  // 1
    static constexpr int V_K = WARP_S / OP_K;               // 1

    using FragK = Array<T, 8>[K_K][K_M];      // (s8,d4) (Dk,Sm) (d2,s2,d2)
                                              //   1  2   16 16    8  8  1
    using FragQ = Array<T, 4>[K_N][K_K];      // (q8,d4) (Qn,Dk) (d2,d2)
                                              //   1  2    8,16    8  1
    using FragS = Array<float, 4>[K_M][K_N];  // (s8,q4) (Sm,Qn) (s2,q2)
                                              //   1  2   16  8    8  1
    using FragV = Array<T, 8>[V_M][V_K];      // (d8,s4) (Dm,Sk) (s2,d2,s2)
                                              //   1  2   16 16    8  8  1
    using FragP = Array<T, 4>[V_K][V_N];      // (q8,s4) (Sk,Qn) (s2,s2)
                                              //   1  2   16  8    8  1
    using FragO = Array<float, 4>[V_M][V_N];  // (d8,q4) (Dm,Qn) (d2,q2)
                                              //   1  2   16  8    8  1
    using FragM = Array<float, 2>[K_N];       // (_8,q4)    (Qn)    (q2)
                                              //      2       8       1

    static constexpr int X = 16 / bitsof<Tkv>;

    using DataK = Array<Tkv, 8 * X>[K_K / X][K_M];  // {s8,d4} [Dk/x,Sm] (d2,s2,dx,d2)
                                                    //   1 2x    16x 16   8x  8  2  1
    using ParamK = Array<T, 2>[K_M][2];             // {s8,_4} [     Sm] (   s2      )
                                                    //   1  0        16       8
    using DataV = Array<Tkv, 8 * X>[V_M / X][V_K];  // {s8,d4} [Dm/x,Sk] (s2,d2,dx,d2)
                                                    //   1 2x    16x 16    8 8x  2  1
    using ParamV = Array<T, 2>[V_K][2];             // {s8,_4} [     Sk] (s2         )
                                                    //   1  0        16    8

    using FragL = FragM;

    using SmemM = Array<float, 2>[K_N][kWarpCntS][4];

    using SmemO = Array<float, 4>[V_M][V_N][kWarpCntS][WARP_SIZE];

    static constexpr bool kUseSmemQ = false;
    static constexpr bool kUseSmemP = false;

    static constexpr int CTA_H1 = (CTA_H + OP_N - 1) / OP_N * OP_N;

    static constexpr auto _SmemLayoutKV(std::integral_constant<int, 16>)
    {
        return SmemLayoutV2<CTA_S, HeadDim, 16, 64, Swizzle<3, 3, 3>>{};
    }
    static constexpr auto _SmemLayoutKV(std::integral_constant<int, 8>)
    {
        return SmemLayoutV2<CTA_S, HeadDim, 32, 64, Swizzle<3, 4, 3>>{};
    }
    static constexpr auto _SmemLayoutKV(std::integral_constant<int, 4>)
    {
        return SmemLayoutV2<CTA_S, HeadDim, 32, 128, Swizzle<2, 5, 3>>{};
    }

    using SmemLayoutQ = SmemLayoutV2<CTA_H1, HeadDim, CTA_H1, HeadDim, Swizzle<3, 3, 4>>;
    using SmemLayoutK = decltype(_SmemLayoutKV(bitsof<Tkv>));
    using SmemLayoutV = decltype(_SmemLayoutKV(bitsof<Tkv>));

    using SmemLayoutKVp = SmemLayoutV2<CTA_S, 2, CTA_S, 2, Identity>;

    using PointerKV = get_pointer_type<Tkv>;

    union SharedStorage {
        __align__(16) T Q[SmemLayoutQ::kSize];

        struct {
            __align__(16) Array<Tkv, Stages * SmemLayoutK::kSize> KV;
            __align__(16) T KVp[Stages * SmemLayoutKVp::kSize];
        };

        struct {
            __align__(16) SmemM M;
            __align__(16) SmemM L;
            __align__(16) SmemO O;
        };

        __align__(16) float O1[CTA_H1][kHeadDim];
    };

    using ThreadMapQ  = RakedThreadMap<HeadDim, CTA_H1, 8, kWarpCount>;
    using ThreadMapKV = RakedThreadMap<HeadDim, CTA_S, 128 / bitsof<Tkv>, kWarpCount>;
    // `WARP_SIZE / WARP_S` is chosen to achieve minimum kIterS w/o introducing partial S iter
    using ThreadMapKVp = RakedThreadMap<2, CTA_S, 2, kWarpCount, WARP_SIZE / WARP_S>;

    static constexpr int kBatchK = ThreadMapKV::kIterS;
    static constexpr int kBatchV = ThreadMapKV::kIterS;

    static constexpr bool kDeferReduceL = true;

    __device__ static void Sync()
    {
        if constexpr (kQuantKV) {  // Thread layout of KV & KVp is different within warp boundary
            __syncwarp();
        }
    }

    template<class GmemIterK, class GmemIterV>
    __device__ static void SetSmemKV(GmemIterK& gmem_K, GmemIterV& gmem_V, SharedStorage& storage, bool offset_kv)
    {
        int pred = offset_kv;
        if constexpr (kQuantKV) {
            gmem_K.SetSmem(storage.KV.data(), storage.KVp);
            gmem_V.SetSmem(storage.KV.data() + pred * SmemLayoutK::kSize, storage.KVp + pred * SmemLayoutKVp::kSize);
        }
        else {
            gmem_K.SetSmem(storage.KV.data());
            gmem_V.SetSmem(storage.KV.data() + pred * SmemLayoutK::kSize);
        }
    }

    template<class Fragment, class Func>
    __device__ static void ForeachS(Fragment& S, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int s = 0; s < 2; ++s) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        const int si = m * OP_M + lane_id / 4 * 1 + s * 8 + warp_id * WARP_S;
                        const int hi = n * OP_N + lane_id % 4 * 2 + q * 1;
                        ((Func&&)func)(hi, /*qi*/ 0, si, /*ri*/ 0, S[m][n][s * 2 + q]);
                    }
                }
            }
        }
    }

    template<class Func>
    __device__ static void ForeachML(FragM& frag_M, FragL& frag_L, Func&& func)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {  // Q
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                const int hi = lane_id % 4 * 2 + n * OP_N + q * 1;
                const int ri = lane_id / 4 * 1;
                ((Func&&)func)(hi, /*qi*/ 0, ri, frag_M[n][q], frag_L[n][q]);
            }
        }
    }

    __device__ static void TransformQ(T* smem_Q, FragQ& frag_Q)
    {
        static_assert(K_K % 2 == 0);
        SmemAccessor<T, SmemLayoutQ> sQ{smem_Q};

        const int lane_id = threadIdx.x % WARP_SIZE;

        if constexpr (!kQuantKV) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_K; k += 2) {  // 16x16 tile
                    const int hi = n * OP_N + lane_id % 8;
                    const int di = k * OP_K + lane_id / 8 * 8;
                    ldsm_x4((Array<uint32_t, 4>&)frag_Q[n][k], cast_smem_ptr_to_uint(&sQ(hi, di)));
                }
            }
        }
        else {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_K; k += X) {
                    PRAGMA_UNROLL
                    for (int x = 0; x < X; ++x) {
                        PRAGMA_UNROLL
                        for (int d = 0; d < 2; ++d) {  // (s8,d8)
                            const int hi = n * OP_N + lane_id / 4;
                            const int di = k * OP_K + lane_id % 4 * 2 * X + x * 2 + d * 8 * X;
                            Load((Array<T, 2>&)frag_Q[n][k + x][d * 2], &sQ(hi, di));
                        }
                    }
                }
            }
        }
    }

    struct StateQK {
        PointerKV smem_K;
        T*        smem_K_param;
        FragQ     frag_Q;
        ParamK    param_K;
        DataK     data_K;
        FragK     frag_K;

        __device__ StateQK(SharedStorage& storage, FragQ frag_Q_)
        {
            smem_K       = storage.KV.data();
            smem_K_param = storage.KVp;
            static_assert(!kUseSmemQ, "not implemented");
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_K; ++k) {
                    frag_Q[n][k] = frag_Q_[n][k];
                }
            }
        }

        __device__ void Load(int k, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;

            if (kQuantKV && k == 0) {
                static_assert(K_M == 1);
                const int m = 0;
                PRAGMA_UNROLL
                for (int s = 0; s < 2; ++s) {
                    const int si = m * 16 + lane_id / 4 * 1 + s * 8 + warp_id * WARP_S;
                    Lds(param_K[m][s], &smem_K_param[pipe_iter * SmemLayoutKVp::kSize + SmemLayoutKVp::apply(si, 0)]);
                }
            }

            if (k % X == 0) {
                const int offset_s = lane_id % 16 * 1 + warp_id * WARP_S;
                const int offset_c = lane_id / 16 * 8 * X;
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
                    const int s = m * 16 + offset_s;  // Q
                    const int c = k * 16 + offset_c;  // D
                    static_assert(sizeof(data_K[k / X][m]) == 16);
                    ldsm_x4((Array<uint32_t, 4>&)data_K[k / X][m],
                            cast_smem_ptr_to_uint(&smem_K[pipe_iter * SmemLayoutK::kSize + SmemLayoutK::apply(s, c)]));
                }
            }
        }

        __device__ void Transform(int k)
        {
            if constexpr (!kQuantKV) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
                    frag_K[k][m] = data_K[k][m];
                }
            }
            else {  // this also covers non-quantized case, but it's too convolved to read
                static_assert(K_M == 1);
                if (k % X == 0) {
                    using Converter = ConvertKvCache<Tkv, T>;
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        PRAGMA_UNROLL
                        for (int d = 0; d < 2; ++d) {
                            auto dx_d2 =
                                Converter::convert((Array<Tkv, X * 2>&)data_K[k / X][0][d * 4 * X + s * 2 * X]);
                            PRAGMA_UNROLL
                            for (int x = 0; x < X; ++x) {
                                (Array<short, 2>&)frag_K[k + x][0][d * 4 + s * 2] = (Array<short, 2>&)dx_d2[x * 2];
                            }
                        }
                    }
                }
                PRAGMA_UNROLL
                for (int s = 0; s < 2; ++s) {
                    PRAGMA_UNROLL
                    for (int d = 0; d < 2; ++d) {
                        auto& d2 = (Array<T, 2>&)frag_K[k][0][d * 4 + s * 2];
                        PRAGMA_UNROLL
                        for (int i = 0; i < 2; ++i) {
                            d2[i] = __hfma(d2[i], param_K[0][s][0], param_K[0][s][1]);
                        }
                    }
                }
            }
        }
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputeQK(StateQK state_QK, FragS& frag_S, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        if constexpr (K_K == 1) {
            ((Prefetch&&)prefetch)(0);
        }

        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                state_QK.Load(k + 1, offset);
            }
            else {
                ((Preload&&)preload)();
            }

            state_QK.Transform(k);

            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    mma_m16n8k16_row_col(frag_S[m][n], state_QK.frag_K[k][m], state_QK.frag_Q[n][k], frag_S[m][n]);
                }
            }
            if (k < K_K - 1) {
                ((Prefetch&&)prefetch)(k);
            }
            if (k == K_K - 2) {
                ((Prefetch&&)prefetch)(K_K - 1);
            }
        }
    }

    struct StatePV {
        PointerKV smem_V;
        T*        smem_V_param;
        ParamV    param_V;
        DataV     data_V;
        FragP     frag_P;
        FragV     frag_V;

        __device__ StatePV(SharedStorage& storage, bool offset = false)
        {
            smem_V       = storage.KV.data() + (offset ? SmemLayoutK::kSize : 0);
            smem_V_param = storage.KVp + (offset ? SmemLayoutKVp::kSize : 0);
        }

        __device__ void Load(int m, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            if (kQuantKV && m == 0) {
                static_assert(V_K == 1);
                const int k = 0;
                PRAGMA_UNROLL
                for (int s = 0; s < 2; ++s) {
                    const int si = k * 16 + lane_id / 4 * 1 + s * 8 + warp_id * WARP_S;
                    Lds(param_V[k][s], &smem_V_param[pipe_iter * SmemLayoutKVp::kSize + SmemLayoutKVp::apply(si, 0)]);
                }
            }

            if (m % X == 0) {
                const int offset_s = lane_id / 16 * 8 + lane_id % 8 + warp_id * WARP_S;
                const int offset_c = lane_id % 16 / 8 * 8 * X;
                PRAGMA_UNROLL
                for (int k = 0; k < V_K; ++k) {
                    const int s = k * 16 + offset_s;
                    const int c = m * 16 + offset_c;
                    static_assert(sizeof(data_V[m / X][k]) == 16);
                    if constexpr (!kQuantKV) {
                        ldsm_x4_trans(
                            (Array<uint32_t, 4>&)data_V[m / X][k],
                            cast_smem_ptr_to_uint(&smem_V[pipe_iter * SmemLayoutV::kSize + SmemLayoutV::apply(s, c)]));
                    }
                    else {
                        ldsm_x4(
                            (Array<uint32_t, 4>&)data_V[m / X][k],
                            cast_smem_ptr_to_uint(&smem_V[pipe_iter * SmemLayoutV::kSize + SmemLayoutV::apply(s, c)]));
                    }
                }
            }
        }

        __device__ void Transform(int m)
        {
            if constexpr (!kQuantKV) {
                PRAGMA_UNROLL
                for (int k = 0; k < V_K; ++k) {
                    frag_V[m][k] = data_V[m][k];
                }
            }
            else {
                static_assert(V_K == 1);
                if (m % X == 0) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        PRAGMA_UNROLL
                        for (int d = 0; d < 2; ++d) {
                            auto dx_d2 = ConvertKvCache<Tkv, T>::convert(
                                (Array<Tkv, 2 * X>&)data_V[m / X][0][s * 4 * X + d * 2 * X]);
                            PRAGMA_UNROLL
                            for (int x = 0; x < X; ++x) {
                                (Array<T, 2>&)frag_V[m + x][0][s * 4 + d * 2] = (Array<T, 2>&)dx_d2[x * 2];
                            }
                        }
                    }
                }
                PRAGMA_UNROLL
                for (int s = 0; s < 2; ++s) {
                    PRAGMA_UNROLL
                    for (int d = 0; d < 2; ++d) {
                        auto& d2 = (Array<T, 2>&)frag_V[m][0][s * 4 + d * 2];
                        PRAGMA_UNROLL
                        for (int i = 0; i < 2; ++i) {
                            d2[i] = __hfma(d2[i], param_V[0][s][0], param_V[0][s][1]);
                        }
                        (uint32_t&)d2 = transpose_m8n8_b16((uint32_t&)d2);
                    }
                }
            }
        }
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputePV(StatePV state_PV, FragO& frag_O, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            if (m < V_M - 1) {
                state_PV.Load(m + 1, offset);
            }
            else {
                ((Preload&&)preload)();
            }

            state_PV.Transform(m);

            PRAGMA_UNROLL
            for (int k = 0; k < V_K; ++k) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    mma_m16n8k16_row_col(frag_O[m][n], state_PV.frag_V[m][k], state_PV.frag_P[k][n], frag_O[m][n]);
                }
            }
            if (m < V_M - 1) {
                ((Prefetch&&)prefetch)(m);
            }
            if (m == V_M - 2) {
                ((Prefetch&&)prefetch)(V_M - 1);
            }
        }
    }

    template<bool is_residue>
    __device__ static void Softmax(FragS& frag_S, FragM& frag_M, FragM& frag_L, FragO& frag_O, float qk_scale)
    {
        FragM prev_M;
        copy(frag_M, prev_M);

        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {  // h
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {  // s
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
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                frag_M[n][q] = fmaxf(frag_M[n][q], __shfl_xor_sync(uint32_t(-1), frag_M[n][q], 4));
                frag_M[n][q] = fmaxf(frag_M[n][q], __shfl_xor_sync(uint32_t(-1), frag_M[n][q], 8));
                frag_M[n][q] = fmaxf(frag_M[n][q], __shfl_xor_sync(uint32_t(-1), frag_M[n][q], 16));
            }
        }

        FragM expdiff_M;
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                expdiff_M[n][q] = exp2f((prev_M[n][q] - frag_M[n][q]) * qk_scale);
                if (is_residue && frag_M[n][q] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M[n][q] = 0.f;
                }
                frag_L[n][q] *= expdiff_M[n][q];
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                PRAGMA_UNROLL
                for (int d = 0; d < 2; ++d) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        frag_O[m][n][d * 2 + q] *= expdiff_M[n][q];  // Rescale previous output
                    }
                }
            }
        }

        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                float tmp_L{};
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
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
                if constexpr (!kDeferReduceL) {
                    tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 4);
                    tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 8);
                    tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 16);
                }
                frag_L[n][q] += tmp_L;  // update L
            }
        }
    }

    __device__ static void ConvertStoP(FragS& frag_S, FragP& frag_P, SharedStorage&)
    {
        static_assert(K_M == V_K);

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int s = 0; s < 2; ++s) {
                    Array<T, 2> tmp_P;
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        tmp_P[q] = static_cast<T>(frag_S[m][n][s * 2 + q]);
                    }
                    // (s8,q4),(s2,q2) -> (q8,s4),(s2,s2)
                    //   1  2    8  1       1  2    8  1
                    (uint32_t&)tmp_P = transpose_m8n8_b16((uint32_t&)tmp_P);

                    (Array<T, 2>&)frag_P[m][n][s * 2] = tmp_P;
                }
            }
        }
    }

    __device__ static void Merge(FragO& frag_O, FragM& frag_M, FragL& frag_L, float qk_scale, SharedStorage& storage)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_id_s = warp_id % kWarpCntS;

        FragM prev_M;
        copy(frag_M, prev_M);

        __syncthreads();

        /////////////////////////////////////////////////////////////////////////
        //  global max
        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            if (lane_id < 4) {
                Store((float*)&storage.M[n][warp_id_s][lane_id], frag_M[n]);
            }
        }

        __syncthreads();

        PRAGMA_UNROLL
        for (int n = 0; n < K_N; ++n) {
            // Compute global maximum
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                PRAGMA_UNROLL
                for (int w = 0; w < kWarpCntS - 1; ++w) {
                    const int src_warp = (warp_id_s + w + 1) % kWarpCntS;
                    frag_M[n][q]       = fmaxf(frag_M[n][q], storage.M[n][src_warp][lane_id % 4][q]);
                }
                // if (lane_id < 4) {
                //     printf("M %d %d %f\n", lane_id % 4 * 2 + q, warp_id, frag_M[n][q]);
                // }
            }
        }

        // if (threadIdx.x == 0) {
        //     printf("M %d %f\n", 0, frag_M[0][0]);
        // }

        ///////////////////////////////////////////////////////////////////////////
        //  rescale & global sum

        FragM expdiff_M;
        PRAGMA_UNROLL
        for (int n = 0; n < V_N; ++n) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                expdiff_M[n][q] = exp2f((prev_M[n][q] - frag_M[n][q]) * qk_scale);
                if (frag_M[n][q] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M[n][q] = 0.f;
                }
            }
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int d = 0; d < 2; ++d) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        frag_O[m][n][d * 2 + q] *= expdiff_M[n][q];
                    }
                }
                Store((float*)&storage.O[m][n][warp_id_s][lane_id], frag_O[m][n]);
            }
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                frag_L[n][q] *= expdiff_M[n][q];
                if constexpr (kDeferReduceL) {
                    frag_L[n][q] += __shfl_xor_sync(uint32_t(-1), frag_L[n][q], 4);
                    frag_L[n][q] += __shfl_xor_sync(uint32_t(-1), frag_L[n][q], 8);
                    frag_L[n][q] += __shfl_xor_sync(uint32_t(-1), frag_L[n][q], 16);
                }
            }
            if (lane_id < 4) {
                Store((float*)&storage.L[n][warp_id_s][lane_id], frag_L[n]);
            }
        }

        __syncthreads();

        clear(frag_O);
        clear(frag_L);

        PRAGMA_UNROLL
        for (int n = 0; n < V_N; ++n) {
            PRAGMA_UNROLL
            for (int w = 0; w < kWarpCntS; ++w) {
                using namespace ops;
                PRAGMA_UNROLL
                for (int m = 0; m < V_M; ++m) {
                    Array<float, 4> tmp_O;
                    Load(tmp_O, storage.O[m][n][w][lane_id].data());
                    frag_O[m][n] = frag_O[m][n] + tmp_O;
                }
                frag_L[n] = frag_L[n] + storage.L[n][w][lane_id % 4];
            }
            // PRAGMA_UNROLL
            // for (int q = 0; q < 2; ++q) {
            //     if (lane_id < 4) {
            //         printf("L %d %d %f\n", lane_id % 4 * 2 + q, warp_id, frag_L[n][q]);
            //     }
            // }

            // if (threadIdx.x == 0) {
            //     printf("L %d %f\n", 0, frag_L[0][0]);
            // }
        }
    }

    template<bool is_norm, class Func>
    __device__ static void StoreO(FragO& frag_O, const FragL& frag_L, SharedStorage& storage, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        FragL inv_L;
        PRAGMA_UNROLL
        for (int n = 0; n < V_N; ++n) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                inv_L[n][q] = fdividef(1.f, frag_L[n][q]);
            }
        }

        __syncthreads();

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; m += X) {
            PRAGMA_UNROLL
            for (int x = 0; x < X; ++x) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    PRAGMA_UNROLL
                    for (int d = 0; d < 2; ++d) {
                        if constexpr (is_norm) {
                            using namespace ops;
                            (Array<float, 2>&)frag_O[m + x][n][d * 2] =
                                (Array<float, 2>&)frag_O[m + x][n][d * 2] * inv_L[n];
                        }
                        PRAGMA_UNROLL
                        for (int q = 0; q < 2; ++q) {
                            const int hi = n * OP_N + lane_id % 4 * 2 + q * 1;
                            // [43][2][10]
                            //   2  1
                            //   4  1
                            //   8  1
                            const int di = m * OP_M + lane_id / 4 % 2 + d * 8 * X + x * 2 + lane_id / 8 * X * 2;
                            if (warp_id == 0) {
                                storage.O1[hi][di] = frag_O[m + x][n][d * 2 + q];
                                // if (hi == 0) {
                                //     printf("O %4d %4d %f\n", hi, di, frag_O[m][n][d * 2 + q]);
                                // }
                            }
                        }
                    }
                }
            }
        }

        __syncthreads();

        using Map = RakedThreadMap<kHeadDim, CTA_H1, 4, kWarpCount>;
        Array<float, 4> tmp_O[Map::kIterS][Map::kIterC];
        const int2      offset = Map::get_offset(warp_id, lane_id);
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                const int hi = offset.y + s * Map::kDeltaS;
                const int di = offset.x + c * Map::kDeltaC;
                Load(tmp_O[s][c], &storage.O1[hi][di]);
                ((Func&&)func)(hi, 0, di, tmp_O[s][c]);
            }
        }
    }
};

}  // namespace turbomind::attention

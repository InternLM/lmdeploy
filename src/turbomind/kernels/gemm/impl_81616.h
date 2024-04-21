// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/mma.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/impl.h"
#include "src/turbomind/kernels/gemm/thread_map.h"

namespace turbomind::gemm {

template<class T_,
         class Tb_,
         class Tq_,
         int CTA_M_,
         int CTA_N_,
         int CTA_K_,
         int WARP_M_,
         int WARP_N_,
         int WARP_K_,
         int Stages_,
         int Flag_>
struct Impl<MMA_81616, T_, Tb_, Tq_, CTA_M_, CTA_N_, CTA_K_, WARP_M_, WARP_N_, WARP_K_, Stages_, Flag_> {

    using T  = T_;
    using Tb = Tb_;
    using Tq = Tq_;

    static constexpr int G = 128;

    static constexpr int CTA_M = CTA_M_;
    static constexpr int CTA_N = CTA_N_;
    static constexpr int CTA_K = CTA_K_;
    static constexpr int CTA_G = (CTA_K + G - 1) / G;

    static constexpr int G_CTA = (G + CTA_K - 1) / CTA_K;

    static constexpr int WARP_M = WARP_M_;
    static constexpr int WARP_N = WARP_N_;
    static constexpr int WARP_K = WARP_K_;

    static constexpr int Stages = Stages_;

    static constexpr int WARP_CNT_M = CTA_M / WARP_M;
    static constexpr int WARP_CNT_N = CTA_N / WARP_N;
    static constexpr int WARP_CNT_K = CTA_K / WARP_K;

    static constexpr int WARP_CNT = WARP_CNT_M * WARP_CNT_N * WARP_CNT_K;

    // - M batch size
    // - N output dims
    // - K input dims

    static constexpr int OP_M = 8;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 16;

    static constexpr int ITER_M = WARP_M / OP_M;
    static constexpr int ITER_N = WARP_N / OP_N;
    static constexpr int ITER_K = WARP_K / OP_K;

    //                               32~512   16
    static constexpr int MMA_CNT_K = CTA_K / OP_K;
    //                               32-256   16
    static constexpr int MMA_CNT_N = CTA_N / OP_N;
    // static constexpr int CNT_NK    = MMA_CNT_N * MMA_CNT_K;

    static constexpr int P = 16 / bitsof<Tb>;
    // static constexpr int P_N = P >= 2 ? 2 : 1;
    static constexpr int P_N = P;
    static constexpr int P_K = P / P_N;

    // cta
    static constexpr int kPackedC = MMA_CNT_K * P_N * OP_N * OP_K;
    static constexpr int kPackedS = MMA_CNT_N / P_N;

    static_assert(P == P_N * P_K);

    static constexpr int kPackedN = Flag_ ? P_N * OP_N : 1;

    static constexpr int P_Q   = Flag_ ? 16 / sizeof(Array<Tq, 2>) : 1;
    static constexpr int P_Q_N = P_Q;
    static constexpr int P_Q_K = 1;

    // static constexpr int kPacked_Q_C = CTA_N;
    // static constexpr int kPacked_Q_S = CTA_G;

    using FragA = Array<T, 4>[ITER_K][ITER_M];      // {m8,k4}, [iK,iM], (k2,k2)
                                                    //   1  2    16  8     8  1
    using FragB = Array<T, 8>[ITER_K][ITER_N];      // {n8,k4}, [iK,iN], (k2,n2,k2)
                                                    //   1  2    16 16     8  8  1
    using FragC = Array<float, 4>[ITER_M][ITER_N];  // {n8,m4}, [iM,iN], (n2,m2)
                                                    //   1  2     8 16     8

    using DataB = Array<Tb, 8 * P>[ITER_K / P_K][ITER_N / P_N];

    // static constexpr int ITER_G = (WARP_K + G - 1) / G;

    using FragQ = Array<Tq, 2>[ITER_K][ITER_N];

    using SmemLayoutA = SmemLayoutV2<CTA_M, CTA_K, 16, 32, Swizzle<2, 3, 3>>;
    // using SmemLayoutA = SmemLayoutV2<CTA_M, CTA_K + 8>;
    // using SmemLayoutA = SmemLayoutV2<CTA_M, CTA_K, 16, 64, Swizzle<3, 3, 3>>;
    using ThreadMapA = gemm::ThreadMap<CTA_K, CTA_M, 8, WARP_CNT>;

    using SmemLayoutB = std::conditional_t<Flag_,
                                           SmemLayoutV2<kPackedS, kPackedC, kPackedS, kPackedC, Identity>,
                                           SmemLayoutV2<CTA_N, CTA_K, 16, 32, Swizzle<2, 3, 3>>>;
    using ThreadMapB  = std::conditional_t<Flag_,
                                          gemm::ThreadMap<kPackedC, kPackedS, 128 / bitsof<Tb>, WARP_CNT, WARP_SIZE>,
                                          gemm::ThreadMap<CTA_K, CTA_N, 8, WARP_CNT>>;

    // [CTA_K / G, CTA_N]
    using SmemLayoutQ = SmemLayoutV2<CTA_G, CTA_N>;
    using ThreadMapQ  = gemm::ThreadMap<CTA_N, CTA_G, 1, WARP_CNT, 32>;

    union SharedStorage {
        struct {
            __align__(16) T A[Stages_ * SmemLayoutA::kSize];
            __align__(16) Array<Tb, Stages_ * SmemLayoutB::kSize> B;
            __align__(16) Array<Tq, Stages_ * SmemLayoutQ::kSize> Q;
        };
    };

    struct StateA {
        SmemAccessor<T, SmemLayoutA> smem_A;
        FragA                        frag_A;
        int                          offset = 0;
        T*                           data;

        __device__ StateA(SharedStorage& storage): smem_A{storage.A}
        {
            data = storage.A;
        }

        __device__ void Load(int k, int)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;

            const int warp_offset_m = warp_id_m(warp_id) * WARP_M;

            if constexpr (ITER_M == 1) {
                // const int offset_s = lane_id % 8 + warp_offset_m;
                // const int offset_c = lane_id / 8 * 8;
                // if constexpr (ITER_K % 2 == 0) {
                //     if (k % 2 == 0) {
                //         const int s = offset_s;
                //         const int c = offset_c + k * 16;
                //         ldsm_x4((Array<uint32_t, 4>&)frag_A[k][0], cast_smem_ptr_to_uint(&smem_A(s, c, offset)));
                //     }
                // }
                // else {
                //     const int s = offset_s;
                //     const int c = offset_c % 16 + k * 16;
                //     ldsm_x2((Array<uint32_t, 2>&)frag_A[k][0], cast_smem_ptr_to_uint(&smem_A(s, c, offset)));
                // }
            }
            else {
                SmemAccessor<T, SmemLayoutA> smem{data};

                const int offset_s = lane_id % 8 + lane_id / 16 * 8 + warp_offset_m;
                const int offset_c = lane_id / 8 * 8 % 16;
                static_assert(ITER_M % 2 == 0);
                PRAGMA_UNROLL
                for (int m = 0; m < ITER_M; m += 2) {
                    const int s = m * 8 + offset_s;
                    const int c = k * 16 + offset_c;
                    ldsm_x4((Array<uint32_t, 4>&)frag_A[k][m], cast_smem_ptr_to_uint(&smem(s, c)));
                }
            }
        }

        __device__ void Advance()
        {
            offset += SmemLayoutA::kSize;
            if (offset == Stages * SmemLayoutA::kSize) {
                offset = 0;
            }
            data = smem_A.ptr_ + offset;
        }
    };

    struct StateQ {
        SmemAccessor<Tq, SmemLayoutQ> smem_Q;
        FragQ                         frag_Q;
        int                           offset = 0;
        Tq*                           data;
        int                           counting  = false;
        int                           g_counter = 0;
        bool                          g_mask    = true;

        template<class Storage>
        __device__ StateQ(Storage& storage): smem_Q{storage.Q.data()}
        {
            data = storage.Q.data();
        }

        __device__ void Load(int k, int)
        {
            // if constexpr (std::is_same_v<T, Tb>) {
            //     return;
            // }

            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;

            const int warp_idx_n    = warp_id_n(warp_id);
            const int warp_offset_n = warp_idx_n * WARP_N;

            // if (threadIdx.x == 0) {
            //     printf("[load_%d]   counter=%d, g_mask=%d\n", k, g_counter, g_mask);
            // }

            if constexpr (!Flag_) {
                PRAGMA_UNROLL
                for (int n = 0; n < ITER_N; ++n) {
                    PRAGMA_UNROLL
                    for (int i = 0; i < 2; ++i) {
                        //              16        1            8          WARP_N
                        const int c = n * 16 + lane_id / 4 + i * 8 + warp_offset_n;
                        const int s = k * 16 / G;
                        if (g_mask) {
                            Lds((Array<Tq, 1>&)frag_Q[k][n][i], &data[SmemLayoutQ::apply(s, c)]);
                        }
                    }
                }
            }
            else {
                constexpr int kAccessSize = 2 * P_Q;
                constexpr int PACKED_N    = MMA_CNT_N / P_Q_N;
                static_assert(PACKED_N * kAccessSize * 8 == CTA_N);
                PRAGMA_UNROLL
                for (int n = 0; n < ITER_N; n += P_Q_N) {
                    const int pack_idx_n = (n + warp_idx_n * ITER_N) / P_Q_N;
                    const int pack_idx_k = (k * 16 / G);
                    const Tq* ptr = data + ((pack_idx_k * PACKED_N + pack_idx_n) * 8 + lane_id / 4) * kAccessSize;
                    if (g_mask) {
                        Lds((Array<Tq, kAccessSize>&)frag_Q[k][n], ptr);
                    }
                }
            }
        }

        __device__ void Advance()
        {
            if constexpr (std::is_same_v<T, Tb>) {
                return;
            }

            offset += SmemLayoutQ::kSize;
            if (offset == Stages * SmemLayoutQ::kSize) {
                offset = 0;
            }
            data = smem_Q.ptr_ + offset;

            if (counting) {
                ++g_counter;
                g_mask = g_counter % G_CTA == 0;
            }
            // if (threadIdx.x == 0) {
            //     printf("[q]        counter=%d, g_mask=%d\n", g_counter, g_mask);
            // }
        }
    };

    struct StateB {
        SmemAccessor<Tb, SmemLayoutB> smem_B;
        DataB                         data_B;
        FragB                         frag_B;
        int                           offset = 0;
        get_pointer_type<Tb>          data;
        StateQ*                       state_Q;

        template<class Storage>
        __device__ StateB(Storage& storage): smem_B{storage.B.data()}
        {
            data = storage.B.data();
        }

        __device__ void Load(int k, int)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;

            const int warp_idx_k    = 0;
            const int warp_idx_n    = warp_id_n(warp_id);
            const int warp_offset_n = warp_idx_n * WARP_N;

            if constexpr (!Flag_) {
                const int offset_s = lane_id % 16 + warp_offset_n;
                const int offset_c = lane_id / 16 * 8;

                PRAGMA_UNROLL
                for (int n = 0; n < ITER_N; ++n) {
                    const int s = n * 16 + offset_s;
                    const int c = k * 16 + offset_c;
                    ldsm_x4((Array<uint32_t, 4>&)frag_B[k][n], cast_smem_ptr_to_uint(&smem_B(s, c, offset)));
                }
            }
            else {
                if (k % P_K == 0) {
                    Tb* lo = (Tb*)data;
                    Tb* hi = (Tb*)(data + SmemLayoutB::kSize);

                    static_assert(ITER_N % P_N == 0);
                    static constexpr int kAccessSize = 8 * P_N * P_K;
                    PRAGMA_UNROLL
                    for (int n = 0; n < ITER_N; n += P_N) {
                        const int pack_idx_k = (k + warp_idx_k * ITER_K) / P_K;
                        const int pack_idx_n = (n + warp_idx_n * ITER_N) / P_N;

                        constexpr int PACKED_K = MMA_CNT_K / P_K;

                        static_assert(WARP_SIZE * kAccessSize == P_K * P_N * OP_K * OP_N);

                        // auto ptr =
                        //     (Tb*)(data + ((pack_idx_n * PACKED_K + pack_idx_k) * WARP_SIZE + lane_id) * kAccessSize);

                        auto ptr = (Tb*)(data + (pack_idx_n * PACKED_K + pack_idx_k) * P_K * P_N * OP_K * OP_N
                                         + lane_id * kAccessSize);

                        Lds(data_B[k / P_K][n / P_N], ptr);
                    }
                }
                state_Q->Load(k, 0);
            }
        }

        __device__ void Transform(int k)
        {
            const int p_k = k % P_K;
            if constexpr (std::is_same_v<T, Tb>) {
                PRAGMA_UNROLL
                for (int n = 0; n < ITER_N; n += P_N) {
                    PRAGMA_UNROLL
                    for (int p_n = 0; p_n < P_N; ++p_n) {
                        frag_B[k][n + p_n] = (Array<Tb, 8>&)data_B[k / P_K][n / P_N][(p_k * P_N + p_n) * 8];
                    }
                }
            }
            else {
                using Converter = ConvertKvCache<Tb, T>;
                auto& frag_Q    = state_Q->frag_Q;
                PRAGMA_UNROLL
                for (int n = 0; n < ITER_N; n += P_N) {
                    PRAGMA_UNROLL
                    for (int p_n = 0; p_n < P_N; ++p_n) {
                        auto& frag = frag_B[k][n + p_n];
                        frag       = Converter::convert((Array<Tb, 8>&)data_B[k / P_K][n / P_N][(p_k * P_N + p_n) * 8]);
                        PRAGMA_UNROLL
                        for (int c = 0; c < 2; ++c) {
                            PRAGMA_UNROLL
                            for (int s = 0; s < 2; ++s) {
                                auto& k2    = (Array<T, 2>&)frag[c * 4 + s * 2];
                                auto& param = to_array(frag_Q[k][n + p_n][s]);
                                k2[0]       = apply_param(k2[0], param);
                                k2[1]       = apply_param(k2[1], param);
                            }
                        }
                    }
                }
            }
        }

        __device__ static Array<half, 2>& to_array(half2& x)
        {
            return reinterpret_cast<Array<half, 2>&>(x);
        }

        __device__ T apply_param(T b, const Array<T, 2>& p)
        {
            return __hfma(b, p[0], p[1]);
        }

        __device__ void Advance()
        {
            offset += SmemLayoutB::kSize;
            if (offset == Stages * SmemLayoutB::kSize) {
                offset = 0;
            }
            data = smem_B.ptr_ + offset;
        }
    };

    template<class Prefetch, class Advance>
    __device__ static void
    Compute(StateA& state_A, StateB& state_B, FragC& frag_C, int pipe_iter, Prefetch&& prefetch, Advance&& advance)
    {
        static_assert(ITER_K > 1);
        // if (threadIdx.x == 0) {
        //     printf("[compute] +++\n");
        // }
        PRAGMA_UNROLL
        for (int k = 0; k < ITER_K; ++k) {
            state_A.Load((k + 1) % ITER_K, pipe_iter);
            state_B.Load((k + 1) % ITER_K, pipe_iter);

            PRAGMA_UNROLL
            for (int n = 0; n < ITER_N; ++n) {
                PRAGMA_UNROLL
                for (int m = 0; m < ITER_M; ++m) {
                    const int mm = m ^ (n % 2 ? ITER_M - 1 : 0);
                    mma_m16n8k16_row_col(frag_C[mm][n], state_B.frag_B[k][n], state_A.frag_A[k][mm], frag_C[mm][n]);
                }
            }
            ((Prefetch&&)prefetch)((k + 1) % ITER_K);
            if (k + 1 == ITER_K - 1) {
                ((Advance&&)advance)();
            }
            // if (k < ITER_K - 1) {
            //     ((Prefetch&&)prefetch)(k);
            // }
            // if (k == ITER_K - 2) {
            //     ((Prefetch&&)prefetch)(ITER_K - 1);
            //     ((Advance&&)advance)();
            // }

            // ! Transform of k must come before prefetching k + 1
            state_B.Transform((k + 1) % ITER_K);
        }
        // if (threadIdx.x == 0) {
        //     printf("[compute] ---\n");
        // }
    }

    template<class Func>
    __device__ static void StoreC(FragC& frag_C, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_offset_m = warp_id_m(warp_id) * WARP_M;
        const int warp_offset_n = warp_id_n(warp_id) * WARP_N;

        PRAGMA_UNROLL
        for (int m = 0; m < ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < ITER_N; ++n) {
                PRAGMA_UNROLL
                for (int nn = 0; nn < 2; ++nn) {
                    auto tmp = cast<T>((Array<float, 2>&)frag_C[m][n][nn * 2]);
                    // {n8,m4},[iM,iN],(n2,m2) -> {m8,n4},[iM,iN],(n2,n2)
                    //   1  2    8 16    8  1       1  2    8 16    8  1
                    (uint32_t&)tmp = transpose_m8n8_b16((uint32_t&)tmp);
                    const int mi   = m * OP_M + lane_id / 4 * 1;
                    const int ni   = n * OP_N + lane_id % 4 * 2 + nn * 8;
                    ((Func&&)func)(warp_offset_m + mi, warp_offset_n + ni, tmp);
                }
            }
        }
    }

    __device__ static int warp_id_m(int warp_id)
    {
        return warp_id % WARP_CNT_M;
    }

    __device__ static int warp_id_n(int warp_id)
    {
        return warp_id / WARP_CNT_M % WARP_CNT_N;
    }

    __device__ static int warp_id_k(int warp_id)
    {
        return warp_id / WARP_CNT_M / WARP_CNT_N;
    }
};

}  // namespace turbomind::gemm
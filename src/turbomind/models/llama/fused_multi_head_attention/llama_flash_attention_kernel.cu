#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

#include "41_fused_multi_head_attention/kernel_forward.h"
#include <cuda_fp16.h>
#include <cutlass/arch/arch.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/half.h>
#include <cutlass/platform/platform.h>

// modified from:
// https://github.com/NVIDIA/cutlass/blob/main/examples/41_fused_multi_head_attention/kernel_forward.h

namespace turbomind {

template<
    // dtype of Q/K/V/M
    typename Element_,
    typename ArchTag,
    int kQueriesPerBlock,
    int kKeysPerBlock_,
    int kSingleValueIteration_ = false>
struct LlamaAttentionKernel:
    AttentionKernel<Element_,
                    ArchTag,
                    true,  // isAligned_
                    kQueriesPerBlock,
                    kKeysPerBlock_,
                    kSingleValueIteration_  // kSingleValueIteration_
                    > {
    using Base = AttentionKernel<Element_,
                                 ArchTag,
                                 true,  // isAligned_
                                 kQueriesPerBlock,
                                 kKeysPerBlock_,
                                 kSingleValueIteration_  // kSingleValueIteration_
                                 >;

    using scalar_t                                      = typename Base::scalar_t;
    using accum_t                                       = typename Base::accum_t;
    using output_t                                      = typename Base::output_t;
    using output_accum_t                                = typename Base::output_accum_t;
    using BaseParams                                    = typename Base::Params;
    static constexpr auto kSingleValueIteration         = kSingleValueIteration_;
    static constexpr bool kSupportsBias                 = true;
    static constexpr auto kKeysPerBlock                 = kKeysPerBlock_;
    static constexpr auto kNumThreads                   = Base::kNumThreads;
    static constexpr auto kMinBlocksPerSm               = Base::kMinBlocksPerSm;
    static constexpr auto kNumWarpsPerBlock             = Base::kNumWarpsPerBlock;
    static constexpr auto kWarpSize                     = Base::kWarpSize;
    static constexpr auto kKeepOutputInRF               = Base::kKeepOutputInRF;
    static constexpr bool kNeedsOutputAccumulatorBuffer = Base::kNeedsOutputAccumulatorBuffer;
    static constexpr auto kAlignLSE                     = Base::kAlignLSE;
    static constexpr auto kPreloadV                     = Base::kPreloadV;
    static constexpr auto kAlignmentQ                   = Base::kAlignmentQ;
    static constexpr auto kAlignmentK                   = Base::kAlignmentK;
    static constexpr auto kAlignmentV                   = Base::kAlignmentV;

    struct Params: BaseParams {
        scalar_t* attn_bias_ptr;
        int32_t   bias_strideM;
        int32_t   bias_strideH;
        int32_t   bias_strideB;

        bool q_use_seqlens = false;
        bool o_use_seqlens = false;

        scalar_t** q_batch_seqs_ptr = nullptr;
        scalar_t** k_batch_seqs_ptr = nullptr;
        scalar_t** v_batch_seqs_ptr = nullptr;
        output_t** o_batch_seqs_ptr = nullptr;

        size_t q_batch_seqs_offset = 0;
        size_t k_batch_seqs_offset = 0;
        size_t v_batch_seqs_offset = 0;
        size_t o_batch_seqs_offset = 0;

        int32_t group_size = 1;

        float scale;

        template<typename ptr_t>
        CUTLASS_DEVICE void
        update_batched_ptr(ptr_t& data_ptr, ptr_t* batch_seq_ptr, size_t batch_seq_offset, int batch_id, int strideB)
        {
            if (batch_seq_ptr != nullptr)
                data_ptr = batch_seq_ptr[batch_id] + batch_seq_offset;
            else
                data_ptr += batch_id * strideB;
        }

        CUTLASS_DEVICE bool advance_to_block()
        {

            auto& query_ptr        = BaseParams::query_ptr;
            auto& key_ptr          = BaseParams::key_ptr;
            auto& value_ptr        = BaseParams::value_ptr;
            auto& cu_seqlens_q_ptr = BaseParams::seqstart_q_ptr;
            auto& cu_seqlens_k_ptr = BaseParams::seqstart_k_ptr;

            auto& output_ptr       = BaseParams::output_ptr;
            auto& output_accum_ptr = BaseParams::output_accum_ptr;
            auto& logsumexp_ptr    = BaseParams::logsumexp_ptr;

            auto& head_dim       = BaseParams::head_dim;
            auto& head_dim_value = BaseParams::head_dim_value;
            auto& num_queries    = BaseParams::num_queries;
            auto& num_keys       = BaseParams::num_keys;

            auto& q_strideM = BaseParams::q_strideM;
            auto& k_strideM = BaseParams::k_strideM;
            auto& v_strideM = BaseParams::v_strideM;
            auto& o_strideM = BaseParams::o_strideM;

            // Everything below is only used in `advance_to_block`
            // and shouldn't use registers
            auto& q_strideH   = BaseParams::q_strideH;
            auto& k_strideH   = BaseParams::k_strideH;
            auto& v_strideH   = BaseParams::v_strideH;
            auto& q_strideB   = BaseParams::q_strideB;
            auto& k_strideB   = BaseParams::k_strideB;
            auto& v_strideB   = BaseParams::v_strideB;
            auto& num_batches = BaseParams::num_batches;
            auto& num_heads   = BaseParams::num_heads;

            auto batch_id    = blockIdx.z;
            auto head_id     = blockIdx.y;
            auto query_start = blockIdx.x * kQueriesPerBlock;

            auto o_strideB = o_strideM * num_queries;

            auto lse_dim = ceil_div((int32_t)num_queries, kAlignLSE) * kAlignLSE;

            int64_t q_start, k_start;

            if (kSupportsBias && attn_bias_ptr != nullptr) {
                attn_bias_ptr += (batch_id * bias_strideB) + (head_id * bias_strideH);
                attn_bias_ptr = warp_uniform(attn_bias_ptr);
            }

            // Advance to current batch - in case of different sequence lengths
            int qq_start, qo_start;
            if (cu_seqlens_q_ptr != nullptr) {
                cu_seqlens_q_ptr += batch_id;
                q_start              = cu_seqlens_q_ptr[0];
                int64_t q_next_start = cu_seqlens_q_ptr[1];
                num_queries          = q_next_start - q_start;

                if (query_start >= num_queries) {
                    return false;
                }
                if (!q_use_seqlens) {
                    update_batched_ptr(query_ptr, q_batch_seqs_ptr, q_batch_seqs_offset, batch_id, q_strideB);
                    qq_start = 0;
                }
                else {
                    qq_start = q_start;
                }
                if (!o_use_seqlens) {
                    update_batched_ptr(output_ptr, o_batch_seqs_ptr, o_batch_seqs_offset, batch_id, o_strideB);
                    qo_start = 0;
                }
                else {
                    qo_start = q_start;
                }
            }
            else {
                update_batched_ptr(query_ptr, q_batch_seqs_ptr, q_batch_seqs_offset, batch_id, q_strideB);
                update_batched_ptr(output_ptr, o_batch_seqs_ptr, o_batch_seqs_offset, batch_id, o_strideB);
                if (output_accum_ptr != nullptr) {
                    output_accum_ptr += batch_id * o_strideB;
                }
                q_start  = 0;
                qq_start = qo_start = q_start;
            }

            if (cu_seqlens_k_ptr != nullptr) {
                cu_seqlens_k_ptr += batch_id;
                k_start              = cu_seqlens_k_ptr[0];
                int64_t k_next_start = cu_seqlens_k_ptr[1];
                num_keys             = k_next_start - k_start;
            }
            else {
                update_batched_ptr(key_ptr, k_batch_seqs_ptr, k_batch_seqs_offset, batch_id, k_strideB);
                update_batched_ptr(value_ptr, v_batch_seqs_ptr, v_batch_seqs_offset, batch_id, v_strideB);
                k_start = 0;
            }

            // Advance to the current batch / head / query_start
            query_ptr += (qq_start + query_start) * q_strideM + head_id * q_strideH;
            key_ptr += k_start * k_strideM + int64_t(head_id / group_size) * k_strideH;
            value_ptr += k_start * v_strideM + int64_t(head_id / group_size) * v_strideH;
            output_ptr += int64_t(qo_start + query_start) * o_strideM + head_id * head_dim_value;

            if (output_accum_ptr != nullptr) {
                output_accum_ptr += int64_t(query_start) * o_strideM + head_id * head_dim_value;
            }
            else {
                // Accumulate directly in the destination buffer (eg for f32)
                output_accum_ptr = (accum_t*)output_ptr;
            }
            if (logsumexp_ptr != nullptr) {
                // lse[batch_id, head_id, query_start]
                logsumexp_ptr += batch_id * lse_dim * num_heads + head_id * lse_dim + query_start;
            }

            num_queries -= query_start;
            num_batches = 0;  // no longer used after

            // Make sure the compiler knows these variables are the same on all
            // the threads of the warp.
            query_ptr        = warp_uniform(query_ptr);
            key_ptr          = warp_uniform(key_ptr);
            value_ptr        = warp_uniform(value_ptr);
            output_ptr       = warp_uniform(output_ptr);
            output_accum_ptr = warp_uniform(output_accum_ptr);
            logsumexp_ptr    = warp_uniform(logsumexp_ptr);
            num_queries      = warp_uniform(num_queries);
            num_keys         = warp_uniform(num_keys);
            head_dim         = warp_uniform(head_dim);
            head_dim_value   = warp_uniform(head_dim_value);
            return true;
        }
    };

    using MM0                             = typename Base::MM0;
    using MM1                             = typename Base::MM1;
    using BaseSharedStorageEpilogueAtEnd  = typename Base::SharedStorageEpilogueAtEnd;
    using BaseSharedStorageEpilogueInLoop = typename Base::SharedStorageEpilogueInLoop;

    // TODO: find a way to optimize non aligned bias
    using BiasLoader = TileSmemLoader<scalar_t,
                                      cutlass::MatrixShape<kQueriesPerBlock, kKeysPerBlock>,
                                      MM0::MmaCore::kThreads,
                                      // input restriction: kv_len has to be a multiple of this value
                                      1>;  // 1 per load. unless bias is aligned.

    using AccumLambdaIterator =
        typename DefaultMmaAccumLambdaIterator<typename MM0::Mma::Operator::IteratorC, accum_t, kWarpSize>::Iterator;

    struct SharedStorageEpilogueAtEnd: BaseSharedStorageEpilogueAtEnd {
        struct SharedStorageAfterMM0 {
            // Everything here might be overwritten during MM0
            union {
                typename BiasLoader::SmemTile          bias;
                typename MM0::AccumulatorSharedStorage si;
            };
            typename MM1::SharedStorageMM1 mm1;
        };

        union {
            typename MM0::Mma::SharedStorage             mm0;
            SharedStorageAfterMM0                        after_mm0;
            typename MM1::DefaultEpilogue::SharedStorage epilogue;
        };
    };

    struct SharedStorageEpilogueInLoop: BaseSharedStorageEpilogueInLoop {
        struct SharedStorageAfterMM0 {
            // Everything here might be overwritten during MM0
            union {
                typename BiasLoader::SmemTile          bias;
                typename MM0::AccumulatorSharedStorage si;
            };
            typename MM1::SharedStorageMM1               mm1;
            typename MM1::DefaultEpilogue::SharedStorage epilogue;
        };

        union {
            typename MM0::Mma::SharedStorage mm0;
            SharedStorageAfterMM0            after_mm0;
        };
    };

    using SharedStorage = typename cutlass::platform::conditional<kSingleValueIteration || kKeepOutputInRF,
                                                                  SharedStorageEpilogueAtEnd,
                                                                  SharedStorageEpilogueInLoop>::type;

    static bool __host__ check_supported(Params const& p)
    {
        if (kSupportsBias) {
            CHECK_ALIGNED_PTR(p.attn_bias_ptr, kAlignmentQ);
            XFORMERS_CHECK(p.num_heads <= 1 || p.bias_strideH % kAlignmentQ == 0,
                           "attn_bias is not correctly aligned (strideH)");
        }
        return Base::check_supported(p);
    }

    static void CUTLASS_DEVICE attention_kernel(Params& p)
    {

        // In this block, we will only ever:
        // - read query[query_start:query_end, :]
        // - write to output[query_start:query_end, :]

        extern __shared__ char smem_buffer[];
        SharedStorage&         shared_storage = *((SharedStorage*)smem_buffer);
        auto&                  m_prime        = shared_storage.m_prime;
        auto&                  s_prime        = shared_storage.s_prime;
        auto&                  si             = shared_storage.after_mm0.si;
        auto&                  mi             = shared_storage.mi;
        const uint32_t         query_start    = blockIdx.x * kQueriesPerBlock;

        static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
        if (thread_id() < kQueriesPerBlock) {
            s_prime[thread_id()] = accum_t(0);
            m_prime[thread_id()] = -cutlass::platform::numeric_limits<accum_t>::infinity();
            mi[thread_id()]      = -cutlass::platform::numeric_limits<accum_t>::infinity();
        }
        typename MM1::Mma::FragmentC accum_o;
        accum_o.clear();

        auto createOutputIter = [&](int col) -> typename MM1::OutputTileIterator {
            using OutputTileIterator = typename MM1::OutputTileIterator;
            return OutputTileIterator(typename OutputTileIterator::Params{(int32_t)p.o_strideM},
                                      p.output_ptr,
                                      typename OutputTileIterator::TensorCoord{p.num_queries, p.head_dim_value},
                                      thread_id(),
                                      {0, col});
        };

        auto createOutputAccumIter = [&](int col) -> typename MM1::OutputTileIteratorAccum {
            using OutputTileIteratorAccum = typename MM1::OutputTileIteratorAccum;
            return OutputTileIteratorAccum(
                typename OutputTileIteratorAccum::Params{(int32_t)p.o_strideM},
                p.output_accum_ptr,
                typename OutputTileIteratorAccum::TensorCoord{p.num_queries, p.head_dim_value},
                thread_id(),
                {0, col});
        };

        // Iterate through keys
        for (int32_t iter_key_start = 0; iter_key_start < p.num_keys; iter_key_start += kKeysPerBlock) {
            int32_t        problem_size_0_m = cutlass::fast_min((int32_t)kQueriesPerBlock, p.num_queries);
            int32_t        problem_size_0_n = cutlass::fast_min(int32_t(kKeysPerBlock), p.num_keys - iter_key_start);
            int32_t const& problem_size_0_k = p.head_dim;
            int32_t const& problem_size_1_n = p.head_dim_value;
            int32_t const& problem_size_1_k = problem_size_0_n;

            auto prologueV = [&](int blockN) {
                typename MM1::Mma::IteratorB iterator_V(typename MM1::IteratorB::Params{MM1::LayoutB(p.v_strideM)},
                                                        p.value_ptr + iter_key_start * p.v_strideM,
                                                        {problem_size_1_k, problem_size_1_n},
                                                        thread_id(),
                                                        cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});
                MM1::Mma::prologue(shared_storage.after_mm0.mm1.mm, iterator_V, thread_id(), problem_size_1_k);
            };

            __syncthreads();  // Need to have shared memory initialized, and `m_prime`
                              // updated from end of prev iter

            // MATMUL: Q.K_t
            //
            // Computes the block-matrix product of:
            // (a) query[query_start:query_end, :]
            // with
            // (b) key[iter_key_start:iter_key_start + kKeysPerBlock]
            // and stores that into `shared_storage.si`
            //

            // Compute threadblock location
            cutlass::gemm::GemmCoord tb_tile_offset = {0, 0, 0};

            cutlass::MatrixCoord tb_offset_A{tb_tile_offset.m() * MM0::Mma::Shape::kM, tb_tile_offset.k()};

            cutlass::MatrixCoord tb_offset_B{tb_tile_offset.k(), tb_tile_offset.n() * MM0::Mma::Shape::kN};

            // Construct iterators to A and B operands
            typename MM0::IteratorA iterator_A(
                typename MM0::IteratorA::Params(typename MM0::MmaCore::LayoutA(p.q_strideM)),
                p.query_ptr,
                {problem_size_0_m, problem_size_0_k},
                thread_id(),
                tb_offset_A);

            typename MM0::IteratorB iterator_B(
                typename MM0::IteratorB::Params(typename MM0::MmaCore::LayoutB(p.k_strideM)),
                p.key_ptr + iter_key_start * p.k_strideM,
                {problem_size_0_k, problem_size_0_n},
                thread_id(),
                tb_offset_B);

            auto my_warp_id = warp_id();
            auto my_lane_id = lane_id();

            // Construct thread-scoped matrix multiply
            typename MM0::Mma mma(shared_storage.mm0, thread_id(), my_warp_id, my_lane_id);

            typename MM0::Mma::FragmentC accum;

            accum.clear();

            auto gemm_k_iterations = (problem_size_0_k + MM0::Mma::Shape::kK - 1) / MM0::Mma::Shape::kK;

            // Compute threadblock-scoped matrix multiply-add
            mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
            __syncthreads();

            if (kPreloadV) {
                prologueV(0);
            }

            typename MM0::Mma::Operator::IteratorC::TensorCoord iteratorC_tile_offset = {
                (tb_tile_offset.m() * MM0::Mma::WarpCount::kM) + (my_warp_id % MM0::Mma::WarpCount::kM),
                (tb_tile_offset.n() * MM0::Mma::WarpCount::kN) + (my_warp_id / MM0::Mma::WarpCount::kM)};

            // multiply by scaling factor
            if (kSupportsBias) {
                accum = cutlass::multiplies<typename MM0::Mma::FragmentC>()(p.scale, accum);
            }

            // apply attention bias if applicable
            if (kSupportsBias && p.attn_bias_ptr != nullptr) {
                // load bias tile Bij into shared memory
                typename BiasLoader::GmemTileIterator bias_iter(
                    {cutlass::layout::RowMajor(p.bias_strideM)},
                    // attn_bias_pointer points to matrix of size (n_queries, n_keys)
                    // for the relevant batch_id and head_id
                    p.attn_bias_ptr + query_start * p.bias_strideM + iter_key_start,
                    {problem_size_0_m, problem_size_0_n},
                    thread_id());
                cutlass::TensorRef<scalar_t, cutlass::layout::RowMajor> bias_tensor_ref(
                    shared_storage.after_mm0.bias.data(), cutlass::layout::RowMajor(MM0::ThreadblockShape::kN));
                typename BiasLoader::SmemTileIterator smem_tile_iter(bias_tensor_ref, thread_id());
                BiasLoader::load(bias_iter, smem_tile_iter);

                // Pij += Bij, Pij is in register fragment and Bij is in shared memory
                auto lane_offset = AccumLambdaIterator::get_lane_offset(lane_id(), warp_id(), iteratorC_tile_offset);
                AccumLambdaIterator::iterateRows(
                    lane_offset,
                    [&](int accum_m) {},
                    [&](int accum_m, int accum_n, int idx) {
                        if (accum_m < problem_size_0_m && accum_n < problem_size_0_n) {
                            accum[idx] += (1.0f - bias_tensor_ref.at({accum_m, accum_n})) * -1e5f;
                        }
                    },
                    [&](int accum_m) {});
            }

            DISPATCH_BOOL(iter_key_start == 0, kIsFirst, ([&] {
                              DISPATCH_BOOL(
                                  p.num_keys - iter_key_start >= kKeysPerBlock, kFullColumns, ([&] {
                                      // Update `mi` from accum stored in registers
                                      // Also updates `accum` with accum[i] <-
                                      // exp(accum[i] * scale
                                      // - mi)
                                      Base::iterative_softmax<MM0::Mma::Operator::IteratorC, kFullColumns, kIsFirst>(
                                          accum_o,
                                          accum,
                                          mi,
                                          m_prime,
                                          s_prime,
                                          lane_id(),
                                          thread_id(),
                                          warp_id(),
                                          p.num_keys - iter_key_start,
                                          iteratorC_tile_offset,
                                          kSupportsBias ? 1.0f : p.scale);
                                  }));
                          }));

            // Output results to shared-memory
            int  warp_idx_mn_0      = my_warp_id % (MM0::Mma::Base::WarpCount::kM * MM0::Mma::Base::WarpCount::kN);
            auto output_tile_coords = cutlass::MatrixCoord{warp_idx_mn_0 % MM0::Mma::Base::WarpCount::kM,
                                                           warp_idx_mn_0 / MM0::Mma::Base::WarpCount::kM};

            MM0::B2bGemm::accumToSmem(shared_storage.after_mm0.si, accum, my_lane_id, output_tile_coords);

            __syncthreads();

            //
            // MATMUL: Attn . V
            // Run the matmul `attn @ V` for a block of attn and V.
            // `attn` is read from shared memory (in `shared_storage_si`)
            // `V` is read from global memory (with iterator_B)
            //

            const int64_t nBlockN =
                kSingleValueIteration ? 1 : ceil_div((int64_t)problem_size_1_n, int64_t(MM1::ThreadblockShape::kN));
            for (int blockN = 0; blockN < nBlockN; ++blockN) {
                int gemm_k_iterations = (problem_size_1_k + MM1::Mma::Shape::kK - 1) / MM1::Mma::Shape::kK;

                // Compute threadblock-scoped matrix multiply-add and store it in accum
                // (in registers)
                if (!kPreloadV) {
                    __syncthreads();  // we share shmem between mma and epilogue
                }

                typename MM1::Mma::IteratorB iterator_V(typename MM1::IteratorB::Params{MM1::LayoutB(p.v_strideM)},
                                                        p.value_ptr + iter_key_start * p.v_strideM,
                                                        {problem_size_1_k, problem_size_1_n},
                                                        thread_id(),
                                                        cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});
                typename MM1::Mma            mma_pv(shared_storage.after_mm0.mm1.mm,
                                         shared_storage.after_mm0.si,
                                         (int)thread_id(),
                                         (int)warp_id(),
                                         (int)lane_id(),
                                         (int)problem_size_1_k);
                mma_pv.set_prologue_done(kPreloadV);
                if (!kKeepOutputInRF) {
                    accum_o.clear();
                }
                mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o);
                __syncthreads();

                if (kPreloadV && !kSingleValueIteration && blockN + 1 < nBlockN) {
                    prologueV(blockN + 1);
                }

                if (!kKeepOutputInRF) {
                    DISPATCH_BOOL(
                        iter_key_start == 0, kIsFirst, ([&] {
                            DISPATCH_BOOL(
                                (iter_key_start + kKeysPerBlock) >= p.num_keys, kIsLast, ([&] {
                                    using DefaultEpilogue = typename MM1::DefaultEpilogue;
                                    using DefaultOp       = typename MM1::DefaultConfig::EpilogueOutputOp;
                                    using ElementCompute  = typename DefaultOp::ElementCompute;
                                    using EpilogueOutputOp =
                                        typename cutlass::epilogue::thread::MemoryEfficientAttentionNormalize<
                                            typename cutlass::platform::conditional<kIsLast, output_t, output_accum_t>::
                                                type,
                                            output_accum_t,
                                            DefaultOp::kCount,
                                            typename DefaultOp::ElementAccumulator,
                                            ElementCompute,
                                            kIsFirst,
                                            kIsLast,
                                            cutlass::Array<ElementCompute, kQueriesPerBlock>>;
                                    using Epilogue = typename cutlass::epilogue::threadblock::EpiloguePipelined<
                                        typename DefaultEpilogue::Shape,
                                        typename MM1::Mma::Operator,
                                        DefaultEpilogue::kPartitionsK,
                                        typename cutlass::platform::conditional<
                                            kIsLast,
                                            typename MM1::OutputTileIterator,
                                            typename MM1::OutputTileIteratorAccum>::type,
                                        typename DefaultEpilogue::AccumulatorFragmentIterator,
                                        typename DefaultEpilogue::WarpTileIterator,
                                        typename DefaultEpilogue::SharedLoadIterator,
                                        EpilogueOutputOp,
                                        typename DefaultEpilogue::Padding,
                                        DefaultEpilogue::kFragmentsPerIteration,
                                        true,                                  // IterationsUnroll
                                        typename MM1::OutputTileIteratorAccum  // Read
                                                                               // iterator
                                        >;

                                    int  col         = blockN * MM1::Mma::Shape::kN;
                                    auto source_iter = createOutputAccumIter(col);
                                    auto dest_iter =
                                        call_conditional<kIsLast,
                                                         decltype(createOutputIter),
                                                         decltype(createOutputAccumIter)>::apply(createOutputIter,
                                                                                                 createOutputAccumIter,
                                                                                                 col);
                                    EpilogueOutputOp rescale(s_prime, m_prime);
                                    Epilogue         epilogue(
                                        shared_storage.epilogue_shared_storage(), thread_id(), warp_id(), lane_id());
                                    epilogue(rescale, dest_iter, accum_o, source_iter);
                                }));
                        }));
                    if (!kSingleValueIteration) {
                        __syncthreads();
                    }
                }
            }
            __syncthreads();  // we modify `m_prime` after
        }

        if (kKeepOutputInRF) {
            constexpr bool kIsFirst = true;
            constexpr bool kIsLast  = true;
            using DefaultEpilogue   = typename MM1::DefaultEpilogue;
            using DefaultOp         = typename MM1::DefaultConfig::EpilogueOutputOp;
            using ElementCompute    = typename DefaultOp::ElementCompute;
            using EpilogueOutputOp  = typename cutlass::epilogue::thread::MemoryEfficientAttentionNormalize<
                output_t,        // output
                output_accum_t,  // source
                DefaultOp::kCount,
                typename DefaultOp::ElementAccumulator,  // accum
                output_accum_t,                          // compute
                kIsFirst,
                kIsLast,
                cutlass::Array<ElementCompute, kQueriesPerBlock>>;
            using Epilogue = typename cutlass::epilogue::threadblock::EpiloguePipelined<
                typename DefaultEpilogue::Shape,
                typename MM1::Mma::Operator,
                DefaultEpilogue::kPartitionsK,
                typename MM1::OutputTileIterator,  // destination
                typename DefaultEpilogue::AccumulatorFragmentIterator,
                typename DefaultEpilogue::WarpTileIterator,
                typename DefaultEpilogue::SharedLoadIterator,
                EpilogueOutputOp,
                typename DefaultEpilogue::Padding,
                DefaultEpilogue::kFragmentsPerIteration,
                true,                                  // IterationsUnroll
                typename MM1::OutputTileIteratorAccum  // source tile
                >;
            auto             dest_iter = createOutputIter(0);
            EpilogueOutputOp rescale(s_prime, m_prime);
            Epilogue         epilogue(shared_storage.epilogue_shared_storage(), thread_id(), warp_id(), lane_id());
            epilogue(rescale, dest_iter, accum_o);
        }

        // 7. Calculate logsumexp
        // To make the backward easier, we pad logsumexp with `inf`
        // this avoids a few bound checks, and is not more expensive during fwd
        static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
        if (p.logsumexp_ptr && thread_id() < kQueriesPerBlock) {
            auto lse_dim = ceil_div((int32_t)p.num_queries, kAlignLSE) * kAlignLSE;
            if (thread_id() < p.num_queries) {
                p.logsumexp_ptr[thread_id()] =
                    accum_t(mi[thread_id()]) + cutlass::fast_log(accum_t(s_prime[thread_id()]));
            }
            else if (thread_id() < lse_dim) {
                p.logsumexp_ptr[thread_id()] = cutlass::platform::numeric_limits<accum_t>::infinity();
            }
        }
    }

    static CUTLASS_DEVICE int8_t lane_id()
    {
        return Base::lane_id();
    }
    static CUTLASS_DEVICE int8_t warp_id()
    {
        return Base::warp_id();
    }
    static CUTLASS_DEVICE int16_t thread_id()
    {
        return Base::thread_id();
    }
};

template<typename T, typename Attention>
void invokeFlashAttention_impl(int                                          batch_size,
                               int                                          head_num,
                               int                                          key_len,
                               int                                          seq_len,
                               int                                          size_per_head,
                               typename FlashAttentionOpImpl<T, 1>::Params& attention_params,
                               cudaStream_t                                 st)
{
    T*     out_ptr          = attention_params.attn_out;
    T*     query_ptr        = attention_params.query;
    T*     key_ptr          = attention_params.key;
    T*     value_ptr        = attention_params.val;
    T*     mask_ptr         = attention_params.mask;
    float* output_accum_ptr = attention_params.out_accum;
    auto*  cu_seqlens_q_ptr = attention_params.cu_seqlens_q;
    auto   layout_q         = attention_params.layout_q;
    auto   layout_k         = attention_params.layout_k;
    auto   layout_v         = attention_params.layout_v;
    auto   layout_o         = attention_params.layout_o;
    auto   group_size       = attention_params.group_size;

    using scalar_t =
        typename std::conditional_t<std::is_same<half, typename std::decay<T>::type>::value, cutlass::half_t, T>;

    const float qk_scale = static_cast<float>(1.f / sqrtf(size_per_head * 1.f));

    constexpr bool kNeedsOutputAccumulatorBuffer = Attention::kNeedsOutputAccumulatorBuffer;
    if (kNeedsOutputAccumulatorBuffer) {
        assert(output_accum_ptr != nullptr);
    }

    // fill param
    typename Attention::Params params{};
    {
        params.query_ptr      = (scalar_t*)(query_ptr);
        params.key_ptr        = (scalar_t*)(key_ptr);
        params.value_ptr      = (scalar_t*)(value_ptr);
        params.attn_bias_ptr  = (scalar_t*)(mask_ptr);
        params.seqstart_q_ptr = cu_seqlens_q_ptr;

        params.output_ptr       = (scalar_t*)(out_ptr);
        params.output_accum_ptr = kNeedsOutputAccumulatorBuffer ? output_accum_ptr : nullptr;
        params.logsumexp_ptr    = nullptr;

        params.scale = qk_scale;

        params.head_dim       = size_per_head;
        params.head_dim_value = size_per_head;
        params.num_queries    = seq_len;
        params.num_keys       = key_len;

        params.bias_strideH = 0;
        params.bias_strideM = key_len;
        params.bias_strideB = seq_len * params.bias_strideM;

        params.q_strideH           = layout_q.stride_head;
        params.q_strideM           = layout_q.stride_seq;
        params.q_strideB           = layout_q.stride_batch;
        params.q_use_seqlens       = layout_q.use_seqlens;
        params.q_batch_seqs_ptr    = (scalar_t**)(layout_q.batch_seqs);
        params.q_batch_seqs_offset = layout_q.batch_seqs_offset;

        params.k_strideH           = layout_k.stride_head;
        params.k_strideM           = layout_k.stride_seq;
        params.k_strideB           = layout_k.stride_batch;
        params.k_batch_seqs_ptr    = (scalar_t**)layout_k.batch_seqs;
        params.k_batch_seqs_offset = layout_k.batch_seqs_offset;

        params.v_strideH           = layout_v.stride_head;
        params.v_strideM           = layout_v.stride_seq;
        params.v_strideB           = layout_v.stride_batch;
        params.v_batch_seqs_ptr    = (scalar_t**)layout_v.batch_seqs;
        params.v_batch_seqs_offset = layout_v.batch_seqs_offset;

        params.o_strideM           = layout_o.stride_seq;
        params.o_use_seqlens       = layout_o.use_seqlens;
        params.o_batch_seqs_ptr    = (scalar_t**)layout_o.batch_seqs;
        params.o_batch_seqs_offset = layout_o.batch_seqs_offset;

        params.num_batches = batch_size;
        params.num_heads   = head_num;

        params.group_size = int32_t(group_size);
    }

    Attention::check_supported(params);

    // start kernel
    auto block_grid  = params.getBlocksGrid();
    auto thread_grid = params.getThreadsGrid();

    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
        cudaFuncSetAttribute(
            attention_kernel_batched_impl<Attention>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    attention_kernel_batched_impl<Attention><<<block_grid, thread_grid, smem_bytes, st>>>(params);
}

#define CUTLASS_ARCH(sm) cutlass::arch::Sm##sm

#define ATTENTION_KERNEL(scalar_t, sm, querys_per_block, keys_per_block, single_value)                                 \
    LlamaAttentionKernel<scalar_t, CUTLASS_ARCH(sm), querys_per_block, keys_per_block, single_value>

template<typename T, int kQueriesPerBlock, int kKeysPerBlock>
bool get_needs_accum_buffer()
{
    using scalar_t =
        typename std::conditional_t<std::is_same<half, typename std::decay<T>::type>::value, cutlass::half_t, T>;

#define GET_NEED_ACCUM_BUFFER(sm)                                                                                      \
    ATTENTION_KERNEL(scalar_t, sm, kQueriesPerBlock, kKeysPerBlock, false)::kNeedsOutputAccumulatorBuffer

    auto sm = getSMVersion();

    switch (sm) {
        case 75:
            return GET_NEED_ACCUM_BUFFER(75);
        default:
            if (sm >= 80) {
                return GET_NEED_ACCUM_BUFFER(80);
            }
            else {
                return GET_NEED_ACCUM_BUFFER(70);
            }
    }
#undef GET_NEED_ACCUM_BUFFER
}

template<typename T, int kQueriesPerBlock, int kKeysPerBlock>
void invoke_attention_impl(bool                                         single_val_iteration,
                           int                                          batch_size,
                           int                                          head_num,
                           int                                          key_len,
                           int                                          seq_len,
                           int                                          size_per_head,
                           typename FlashAttentionOpImpl<T, 1>::Params& params,
                           cudaStream_t                                 st)
{
    using scalar_t =
        typename std::conditional_t<std::is_same<half, typename std::decay<T>::type>::value, cutlass::half_t, T>;

#define INVOKE_ATTEN_IMPL(sm, single_value)                                                                            \
    {                                                                                                                  \
        using AttentionKernel = ATTENTION_KERNEL(scalar_t, sm, kQueriesPerBlock, kKeysPerBlock, single_value);         \
        invokeFlashAttention_impl<T, AttentionKernel>(                                                                 \
            batch_size, head_num, key_len, seq_len, size_per_head, params, st);                                        \
    }

#define INVOKE_ATTENN_IMPL_V2(sm)                                                                                      \
    {                                                                                                                  \
        if (single_val_iteration)                                                                                      \
            INVOKE_ATTEN_IMPL(sm, true)                                                                                \
        else                                                                                                           \
            INVOKE_ATTEN_IMPL(sm, false)                                                                               \
    }

    auto sm = getSMVersion();
    switch (sm) {
        case 75:
            INVOKE_ATTENN_IMPL_V2(75);
            break;
        default:
            if (sm >= 80) {
                INVOKE_ATTENN_IMPL_V2(80);
            }
            else {
                INVOKE_ATTENN_IMPL_V2(70);
            }
    }

#undef INVOKE_ATTENN_IMPL_V2
#undef INVOKE_ATTEN_IMPL
}

template<typename T>
class FlashAttentionOpImpl<T, 1> {

public:
    using AttentionLayout = BaseAttentionLayout<T>;
    using Params          = BaseAttentionParams<T>;

public:
    FlashAttentionOpImpl(int batch_size, int head_num, int key_len, int seq_len, int size_per_head);
    ~FlashAttentionOpImpl();

    int get_workspace_size() const;

    void operator()(Params& params, cudaStream_t st) const;

private:
    class impl;
    std::unique_ptr<impl> pimpl;
};

template<typename T>
class FlashAttentionOpImpl<T, 1>::impl {

private:
    static constexpr int kQueriesPerBlock = 32;
    static constexpr int kKeysPerBlock    = 128;
    using scalar_t =
        typename std::conditional_t<std::is_same<half, typename std::decay<T>::type>::value, cutlass::half_t, T>;
    using Params = typename FlashAttentionOpImpl<T, 1>::Params;

    int  batch_size_;
    int  head_num_;
    int  key_len_;
    int  seq_len_;
    int  size_per_head_;
    bool kSingleValueIteration;

public:
    impl(int batch_size, int head_num, int key_len, int seq_len, int size_per_head):
        batch_size_(batch_size),
        head_num_(head_num),
        key_len_(key_len),
        seq_len_(seq_len),
        size_per_head_(size_per_head)
    {
        kSingleValueIteration = (size_per_head <= kKeysPerBlock);
    }

    ~impl() {}

    int get_workspace_size() const
    {
        if (kSingleValueIteration) {
            return 0;
        }
        else {
            bool kNeedsOutputAccumulatorBuffer = get_needs_accum_buffer<T, kQueriesPerBlock, kKeysPerBlock>();
            if (kNeedsOutputAccumulatorBuffer) {
                return batch_size_ * head_num_ * seq_len_ * size_per_head_ * sizeof(float);
            }
            else {
                return 0;
            }
        }
    }

    void operator()(Params& params, cudaStream_t st) const
    {
        invoke_attention_impl<T, kQueriesPerBlock, kKeysPerBlock>(
            kSingleValueIteration, batch_size_, head_num_, key_len_, seq_len_, size_per_head_, params, st);
    }
};

template<typename T>
FlashAttentionOpImpl<T, 1>::FlashAttentionOpImpl(
    int batch_size, int head_num, int key_len, int seq_len, int size_per_head):
    pimpl{std::make_unique<FlashAttentionOpImpl<T, 1>::impl>(batch_size, head_num, key_len, seq_len, size_per_head)}
{
}

template<typename T>
FlashAttentionOpImpl<T, 1>::~FlashAttentionOpImpl()
{
}

template<typename T>
int FlashAttentionOpImpl<T, 1>::get_workspace_size() const
{
    return pimpl->get_workspace_size();
}

template<typename T>
void FlashAttentionOpImpl<T, 1>::operator()(Params& params, cudaStream_t st) const
{
    pimpl->operator()(params, st);
}

template class FlashAttentionOpImpl<float, 1>;
template class FlashAttentionOpImpl<half, 1>;

}  // namespace turbomind

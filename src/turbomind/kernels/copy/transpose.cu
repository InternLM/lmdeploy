#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/copy/copy.h"
#include <cute/algorithm/copy.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

namespace turbomind::core {

using namespace cute;

// ============================================================================
// CUDA kernel: TransposeCopyKernel (vectorized 3-phase smem-staged transpose)
// ============================================================================
namespace kernel {

extern __shared__ char smem_buf[];

template<int kTileDim, int kVec, typename SrcEngine, typename SrcLayout, typename DstEngine, typename DstLayout>
__global__ void __launch_bounds__(256)
    TransposeCopyKernel(cute::Tensor<SrcEngine, SrcLayout> src, cute::Tensor<DstEngine, DstLayout> dst)
{
    using T = typename SrcEngine::value_type;
    static_assert(std::is_same_v<T, typename DstEngine::value_type>,
                  "TransposeCopyKernel: src and dst value types must match");

    constexpr int kPad     = kVec;
    constexpr int kStride  = kTileDim + kPad;
    constexpr int kThrRows = 256 / kTileDim;
    using VecT             = uint_bit_t<kVec * sizeof_bits_v<T>>;

    T* smem_base = reinterpret_cast<T*>(smem_buf);

    // Smem1: row-major — contiguous dim 0 (for phase 1 vectorization)
    auto smem1 =
        make_tensor(make_smem_ptr(smem_base),
                    make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}), make_stride(Int<1>{}, Int<kStride>{})));

    // Smem2: col-major — contiguous dim 1 (for phase 2 vectorization)
    auto smem2 =
        make_tensor(make_smem_ptr(smem_base + kTileDim * kStride),
                    make_layout(make_shape(Int<kTileDim>{}, Int<kTileDim>{}), make_stride(Int<kStride>{}, Int<1>{})));

    // Decode blockIdx.z → multi-dim batch coord → per-block pointer offsets.
    // The `if constexpr (kRank > 2)` guard is REQUIRED: cute::crd2idx /
    // cute::idx2crd are implemented with unary fold expressions of the form
    // `(... + crd2idx_inner(...))` over the shape's tuple_seq. For rank == 2
    // the tuple_seq is empty, and a unary `+` fold over an empty pack is
    // ill-formed C++. Skipping the calls entirely is the simplest fix; the
    // 2D body below still runs, with src_off/dst_off both 0.
    constexpr int kRank   = rank_v<SrcLayout>;
    int64_t       src_off = 0;
    int64_t       dst_off = 0;
    if constexpr (kRank > 2) {
        auto batch_shape   = take<2, kRank>(shape(src));
        auto src_batch_str = take<2, kRank>(stride(src));
        auto dst_batch_str = take<2, kRank>(stride(dst));
        auto batch_coord   = idx2crd(int64_t(blockIdx.z), batch_shape);
        src_off            = crd2idx(batch_coord, batch_shape, src_batch_str);
        dst_off            = crd2idx(batch_coord, batch_shape, dst_batch_str);
    }

    // 2D view of the (I, J) plane at this batch's offset. The static Int<1>
    // strides at src dim 0 / dst dim 1 are preserved (they propagate through
    // shape<0>/stride<0>/shape<1>/stride<1> as compile-time values), keeping
    // the smem partitioning identical to the original 2D path. For kRank==2
    // src_off/dst_off are both 0, so this view equals the input tensors.
    auto src_2d =
        make_tensor(make_gmem_ptr(raw_pointer_cast(src.data()) + src_off),
                    make_layout(make_shape(shape<0>(src), shape<1>(src)), make_stride(stride<0>(src), stride<1>(src))));
    auto dst_2d =
        make_tensor(make_gmem_ptr(raw_pointer_cast(dst.data()) + dst_off),
                    make_layout(make_shape(shape<0>(dst), shape<1>(dst)), make_stride(stride<0>(dst), stride<1>(dst))));

    // Tile the 2D plane (existing logic, unchanged from here on)
    auto tiler     = make_shape(Int<kTileDim>{}, Int<kTileDim>{});
    auto src_tiled = tiled_divide(src_2d, tiler);
    auto dst_tiled = tiled_divide(dst_2d, tiler);

    // Bounds check on tile grid
    if (blockIdx.y >= size<1>(src_tiled) || blockIdx.x >= size<2>(src_tiled))
        return;

    auto src_tile = src_tiled(make_coord(_, _), blockIdx.y, blockIdx.x);
    auto dst_tile = dst_tiled(make_coord(_, _), blockIdx.y, blockIdx.x);

    // Phase 1: gmem(src) -> smem1, vectorize along dim 0
    auto tc1  = make_tiled_copy(Copy_Atom<UniversalCopy<VecT>, T>{},
                               make_layout(make_shape(Int<kThrRows>{}, Int<kTileDim>{})),
                               make_layout(make_shape(Int<kVec>{}, Int<1>{})));
    auto thr1 = tc1.get_slice(threadIdx.x);
    copy(tc1, thr1.partition_S(src_tile), thr1.partition_D(smem1));

    __syncthreads();

    // In-smem: smem1 -> smem2 (physical layout conversion, same logical data)
    auto tc_s  = make_tiled_copy(Copy_Atom<UniversalCopy<T>, T>{},
                                make_layout(make_shape(Int<16>{}, Int<16>{})),
                                make_layout(make_shape(Int<1>{}, Int<1>{})));
    auto thr_s = tc_s.get_slice(threadIdx.x);
    copy(tc_s, thr_s.partition_S(smem1), thr_s.partition_D(smem2));

    __syncthreads();

    // Phase 2: smem2 -> gmem(dst), vectorize along dim 1
    auto tc2 = make_tiled_copy(
        Copy_Atom<UniversalCopy<VecT>, T>{},
        make_layout(make_shape(Int<kTileDim>{}, Int<kThrRows>{}), make_stride(Int<kThrRows>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<kVec>{})));
    auto thr2 = tc2.get_slice(threadIdx.x);
    copy(tc2, thr2.partition_S(smem2), thr2.partition_D(dst_tile));
}

}  // namespace kernel

namespace detail {

// Build a CuTe Shape tuple from a runtime ssize_t array.
template<size_t... Is>
auto make_cute_shape_impl(const ssize_t* data, std::index_sequence<Is...>)
{
    return make_shape(static_cast<int32_t>(data[Is])...);
}

template<int kRank>
auto make_cute_shape(const ssize_t* data)
{
    return make_cute_shape_impl(data, std::make_index_sequence<kRank>{});
}

// Per-element selector: Int<1>{} at UnitPos, otherwise dynamic int64_t.
template<int UnitPos, size_t I>
auto stride_elem(const ssize_t* data)
{
    if constexpr (I == UnitPos) {
        return cute::Int<1>{};
    }
    else {
        return static_cast<int64_t>(data[I]);
    }
}

// Build a CuTe Stride tuple of length kRank with Int<1>{} at UnitPos and
// dynamic int64_t at all other positions.
template<int UnitPos, size_t... Is>
auto make_unit_stride_impl(const ssize_t* data, std::index_sequence<Is...>)
{
    return cute::make_stride(stride_elem<UnitPos, Is>(data)...);
}

template<int UnitPos, int kRank>
auto make_unit_stride(const ssize_t* data)
{
    return make_unit_stride_impl<UnitPos>(data, std::make_index_sequence<kRank>{});
}

}  // namespace detail

// ============================================================================
// TransposeCopy: 2D transpose via vectorized smem-staged TiledCopy
// ============================================================================
void TransposeCopy(
    const void* data_a, void* data_b, const Layout& a, const Layout& b, DataType dtype, cudaStream_t stream)
{
    const int rank = a.rank();
    int32_t   M    = static_cast<int32_t>(a.shape(0));
    int32_t   N    = static_cast<int32_t>(a.shape(1));

    auto launch = [&](auto t, auto kvec, auto ktiledim, auto rank_c) {
        using T                = decltype(t);
        constexpr int kVec     = decltype(kvec)::value;
        constexpr int kTileDim = decltype(ktiledim)::value;
        constexpr int kRank    = decltype(rank_c)::value;

        if (M % kTileDim || N % kTileDim) {
            TM_CHECK(0) << "TransposeCopy: shape (" << M << ", " << N << ") not divisible by tile " << kTileDim;
            return;
        }

        auto data_shape  = detail::make_cute_shape<kRank>(a.shape().data());
        auto src_strides = detail::make_unit_stride<0, kRank>(a.stride().data());
        auto dst_strides = detail::make_unit_stride<1, kRank>(b.stride().data());

        auto src_gmem =
            make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(data_a)), make_layout(data_shape, src_strides));
        auto dst_gmem = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(data_b)), make_layout(data_shape, dst_strides));

        int64_t total_batch = 1;
        for (int i = 2; i < kRank; ++i)
            total_batch *= a.shape(i);

        constexpr int smem_bytes = 2 * kTileDim * (kTileDim + kVec) * sizeof(T);
        dim3          grid(static_cast<uint32_t>(N / kTileDim),
                  static_cast<uint32_t>(M / kTileDim),
                  static_cast<uint32_t>(total_batch));

        kernel::TransposeCopyKernel<kTileDim, kVec><<<grid, 256, smem_bytes, stream>>>(src_gmem, dst_gmem);
    };

    auto dispatch_rank = [&](auto t, auto kvec, auto ktiledim) {
        switch (rank) {
            case 2:
                return launch(t, kvec, ktiledim, std::integral_constant<int, 2>{});
            case 3:
                return launch(t, kvec, ktiledim, std::integral_constant<int, 3>{});
            case 4:
                return launch(t, kvec, ktiledim, std::integral_constant<int, 4>{});
            default:
                TM_CHECK(0) << "TransposeCopy: rank " << rank << " not supported";
                return;
        }
    };

    switch (byte_size(dtype)) {
        case 1:
            return dispatch_rank(uint8_t{}, Int<16>{}, Int<64>{});
        case 2:
            return dispatch_rank(uint16_t{}, Int<8>{}, Int<64>{});
        case 4:
            return dispatch_rank(uint32_t{}, Int<4>{}, Int<32>{});
        case 8:
            return dispatch_rank(uint64_t{}, Int<2>{}, Int<32>{});
        default:
            TM_CHECK(0) << "TransposeCopy: unsupported element size " << byte_size(dtype);
            break;
    }
}

}  // namespace turbomind::core

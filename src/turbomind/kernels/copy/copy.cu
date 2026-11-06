#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include <array>
#include <numeric>
#include <utility>

namespace turbomind::core {

using namespace cute;

// CuTe's make_shape/make_stride require compile-time variadic template args,
// but our tensor shapes and strides are runtime values. These helpers bridge
// that gap via std::index_sequence expansion, producing CuTe tuple types
// from runtime shape/stride arrays.
namespace detail {

template<class Index, size_t... Is>
auto make_cute_shape_impl(const ssize_t* data, std::index_sequence<Is...>)
{
    return make_shape(static_cast<Index>(data[Is])...);
}

template<int kRank, class Index = int64_t>
auto make_cute_shape(const ssize_t* data)
{
    return make_cute_shape_impl<Index>(data, std::make_index_sequence<kRank>{});
}

template<size_t... Is>
auto make_cute_stride_impl(const ssize_t* data, std::index_sequence<Is...>)
{
    return make_stride(static_cast<int64_t>(data[Is])...);
}

template<int kRank>
auto make_cute_stride(const ssize_t* data)
{
    return make_cute_stride_impl(data, std::make_index_sequence<kRank>{});
}

// Construct vec_factors tuple: (kVec, 1, 1, ...) — used for element coord scaling.
// Dim 0 (innermost) scales by kVec; all other dims scale by 1.
template<int kVec, size_t I>
auto make_vec_factor()
{
    if constexpr (I == 0) {
        return Int<kVec>{};
    }
    else {
        return Int<1>{};
    }
}

template<int kVec, int kRank, size_t... Is>
auto make_vec_factors_impl(std::index_sequence<Is...>)
{
    return make_shape(make_vec_factor<kVec, Is>()...);
}

template<int kVec, int kRank>
auto make_vec_factors()
{
    return make_vec_factors_impl<kVec, kRank>(std::make_index_sequence<kRank>{});
}

// Compute thread partition: (T0, T1, ..., Tk-1) where T0*...*Tk-1 = 256.
// Use power-of-two factors for inner dimensions and give the outermost
// dimension all remaining threads, so every thread belongs to this tile.
template<int kRank>
auto compute_thr_partition(const ssize_t* shape, int kVec) -> std::array<ssize_t, kRank>
{
    std::array<ssize_t, kRank> partition{};
    partition.fill(1);

    ssize_t remaining = 256;
    for (int i = 0; i < kRank - 1; ++i) {
        const ssize_t extent = shape[i] / (i == 0 ? kVec : 1);
        ssize_t       threads = remaining;
        while (threads > 1 && threads > extent) {
            threads /= 2;
        }
        partition[i] = threads;
        remaining /= threads;
    }
    partition[kRank - 1] = remaining;
    return partition;
}

}  // namespace detail

// ============================================================================
// CUDA kernel: CopyKernelND (full-utilization manual-tiling copy)
// ============================================================================
namespace kernel {

template<int kVec,
         class DataShape,
         class SrcStride,
         class DstStride,
         class ThrPartition,
         class TileCounts,
         class VecFactors,
         typename T>
__global__ void __launch_bounds__(256) CopyKernelND(const T* __restrict__ src,
                                                    T* __restrict__ dst,
                                                    DataShape    data_shape,
                                                    SrcStride    src_strides,
                                                    DstStride    dst_strides,
                                                    ThrPartition thr_partition,
                                                    TileCounts   tile_counts,
                                                    VecFactors   vec_factors)
{
    using namespace cute;

    // 1. Decode threadIdx -> per-dim thread coordinate (colexicographic)
    auto thr_coord = idx2crd(threadIdx.x, thr_partition);

    // 2. Decode blockIdx -> per-dim tile coordinate
    auto tile_coord = idx2crd(int64_t(blockIdx.x), tile_counts);

    // 3. Compute element coordinate
    auto inner_coord =
        transform(tile_coord, thr_coord, thr_partition, [](auto tc, auto thr, auto tp) { return tc * tp + thr; });
    auto elem_coord = transform(inner_coord, vec_factors, [](auto ic, auto vf) { return ic * vf; });

    // 4. Bounds check: elem[i] + vec_factor[i] <= shape[i]
    bool valid = true;
    for_each(transform(elem_coord, vec_factors, data_shape, [](auto ec, auto vf, auto s) { return ec + vf <= s; }),
             [&](auto v) { valid = valid && static_cast<bool>(v); });
    if (!valid)
        return;

    // 5. Compute memory offsets via CuTe's crd2idx
    int64_t src_off = crd2idx(elem_coord, data_shape, src_strides);
    int64_t dst_off = crd2idx(elem_coord, data_shape, dst_strides);

    // 6. Per-thread vectorized copy via Copy_Atom
    auto src_frag =
        make_tensor(make_gmem_ptr(src + src_off), make_layout(make_shape(Int<kVec>{}), make_stride(Int<1>{})));
    auto dst_frag =
        make_tensor(make_gmem_ptr(dst + dst_off), make_layout(make_shape(Int<kVec>{}), make_stride(Int<1>{})));
    copy(Copy_Atom<UniversalCopy<uint_bit_t<kVec * sizeof_bits_v<T>>>, T>{}, src_frag, dst_frag);
}

}  // namespace kernel

// ============================================================================
// VectorizedCopy: alignment-gated vectorized ND copy via CopyKernelND
// ============================================================================
void VectorizedCopy(
    const void* data_a, void* data_b, const Layout& a, const Layout& b, int rank, DataType dtype, cudaStream_t stream)
{
    constexpr int kBlockThreads = 256;

    // --- Alignment detection ---
    int64_t alignment = 16;

    auto align = [&](auto v) { alignment = std::gcd(alignment, v); };

    align(byte_size(dtype, a.shape(0)));
    align(reinterpret_cast<uintptr_t>(data_a));
    align(reinterpret_cast<uintptr_t>(data_b));

    for (int i = 1; i < rank; ++i) {
        align(byte_size(dtype, a.stride(i)));
        align(byte_size(dtype, b.stride(i)));
    }

    // --- vec_size computation ---
    const int elem_size = byte_size(dtype);
    // A vector atom copies adjacent elements. Alignment alone cannot make a
    // broadcast or strided axis contiguous, so those layouts use one element.
    int vec_size = 1;
    if (a.stride(0) == 1 && b.stride(0) == 1) {
        vec_size = static_cast<int>(alignment / elem_size);
    }

    if (vec_size * elem_size > 16) {
        vec_size = 16 / elem_size;
    }

    while (vec_size > 1 && a.shape(0) % static_cast<int64_t>(vec_size) != 0) {
        vec_size /= 2;
    }

    // --- Dispatch on data type T and vec_size kVec ---
    auto dispatch_elem_size = [&](auto t) {
        using T                 = decltype(t);
        constexpr int kElemBits = sizeof_bits_v<T>;
        constexpr int kMaxVec   = 128 / kElemBits;

        auto dispatch_vec = [&](auto v) {
            auto dispatch_rank = [&](auto d) {
                // Keep template arguments local to this nested lambda. MSVC
                // otherwise captures constexpr values from the enclosing
                // lambda and rejects them as non-type template arguments.
                constexpr int kVec  = decltype(v)::value;
                constexpr int kRank = decltype(d)::value;

                auto data_shape  = detail::make_cute_shape<kRank>(a.shape().data());
                auto src_strides = detail::make_cute_stride<kRank>(a.stride().data());
                auto dst_strides = detail::make_cute_stride<kRank>(b.stride().data());

                auto partition_arr = detail::compute_thr_partition<kRank>(a.shape().data(), kVec);
                // Thread coordinates are bounded by the 256-thread block;
                // logical data shapes can exceed INT32_MAX after coalescing.
                auto thr_partition = detail::make_cute_shape<kRank, int32_t>(partition_arr.data());

                auto vec_factors = detail::make_vec_factors<kVec, kRank>();
                auto tile_sizes  = transform(thr_partition, vec_factors, [](auto tp, auto vf) { return tp * vf; });
                auto tile_counts = transform(data_shape, tile_sizes, [](auto s, auto ts) -> int64_t {
                    return (static_cast<int64_t>(s) + static_cast<int64_t>(ts) - 1) / static_cast<int64_t>(ts);
                });

                int64_t total_tiles = product(tile_counts);
                dim3    grid(static_cast<uint32_t>(total_tiles));

                kernel::CopyKernelND<kVec><<<grid, kBlockThreads, 0, stream>>>(reinterpret_cast<const T*>(data_a),
                                                                               reinterpret_cast<T*>(data_b),
                                                                               data_shape,
                                                                               src_strides,
                                                                               dst_strides,
                                                                               thr_partition,
                                                                               tile_counts,
                                                                               vec_factors);
            };

            switch (rank) {
                case 1:
                    dispatch_rank(constant<1>{});
                    break;
                case 2:
                    dispatch_rank(constant<2>{});
                    break;
                case 3:
                    dispatch_rank(constant<3>{});
                    break;
                case 4:
                    dispatch_rank(constant<4>{});
                    break;
                default:
                    TM_CHECK(0) << "VectorizedCopy: rank > 4 not implemented";
                    break;
            }
        };

        switch (vec_size) {
            case 16:
                if constexpr (16 <= kMaxVec)
                    dispatch_vec(constant<16>{});
                break;
            case 8:
                if constexpr (8 <= kMaxVec)
                    dispatch_vec(constant<8>{});
                break;
            case 4:
                if constexpr (4 <= kMaxVec)
                    dispatch_vec(constant<4>{});
                break;
            case 2:
                if constexpr (2 <= kMaxVec)
                    dispatch_vec(constant<2>{});
                break;
            default:
                dispatch_vec(constant<1>{});
                break;
        }
    };

    switch (byte_size(dtype)) {
        case 1:
            return dispatch_elem_size(uint8_t{});
        case 2:
            return dispatch_elem_size(uint16_t{});
        case 4:
            return dispatch_elem_size(uint32_t{});
        case 8:
            return dispatch_elem_size(uint64_t{});
        default:
            TM_CHECK(0) << "VectorizedCopy: unsupported element size " << byte_size(dtype);
            break;
    }
}

}  // namespace turbomind::core

#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace turbomind::core {

// Forward declarations — defined in copy.cu and transpose.cu
void VectorizedCopy(
    const void* data_a, void* data_b, const Layout& a, const Layout& b, int rank, DataType dtype, cudaStream_t stream);

void TransposeCopy(
    const void* data_a, void* data_b, const Layout& a, const Layout& b, DataType dtype, cudaStream_t stream);

// Both kernels decode one coordinate for source and destination. Remove
// singleton dimensions and merge contiguous dimensions jointly, in the
// innermost-first order used by the kernels. The first `begin` dimensions
// are preserved when coalescing only the batch axes of a transpose.
static std::pair<Layout, Layout> coalesce_copy_dims(const Layout& a, const Layout& b, int begin = 0)
{
    std::vector<ssize_t> shape, src_stride, dst_stride;
    for (int i = 0; i < a.rank(); ++i) {
        if (i >= begin && a.shape(i) == 1) {
            continue;
        }
        if (shape.size() > static_cast<size_t>(begin)
            && a.stride(i) == shape.back() * src_stride.back()
            && b.stride(i) == shape.back() * dst_stride.back()) {
            shape.back() *= a.shape(i);
        }
        else {
            shape.push_back(a.shape(i));
            src_stride.push_back(a.stride(i));
            dst_stride.push_back(b.stride(i));
        }
    }
    if (shape.empty()) {
        return {Layout{{1}, {1}}, Layout{{1}, {1}}};
    }
    return {Layout{shape, src_stride}, Layout{shape, dst_stride}};
}

// ============================================================================
// GenericCopy: layout normalization + dispatch
// ============================================================================
void GenericCopy(const Tensor& src, Tensor& dst, cudaStream_t stream)
{
    auto a = src.layout();
    auto b = dst.layout();

    TM_CHECK(src.dtype() == dst.dtype()) << "GenericCopy: src and dst must have the same dtype";
    TM_CHECK(a.shape() == b.shape()) << "GenericCopy: src and dst must have the same shape";
    TM_CHECK_GT(byte_size(src.dtype()), 0) << "GenericCopy: sub-byte elements are unsupported";
    if (a.size() == 0) {
        return;
    }

    // Put physical source axes first, keeping broadcast axes outside them.
    // Apply exactly the same permutation and coalescing to both layouts.
    std::vector<int> idxs(a.rank());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::stable_sort(idxs.begin(), idxs.end(), [&](int i, int j) {
        if ((a.stride(i) == 0) != (a.stride(j) == 0)) {
            return a.stride(i) != 0;
        }
        return a.stride(i) < a.stride(j);
    });

    a = a.permute(idxs);
    b = b.permute(idxs);

    std::tie(a, b) = coalesce_copy_dims(a, b);
    const int rank = a.rank();

    const DataType dtype = src.dtype();

    // --- Transpose detection (2D + batched) ---
    // After joint normalization, we dispatch to TransposeCopy when:
    //   - position 0 has src stride 1 (call it I),
    //   - some position J ∈ [1, rank-1] has dst stride 1,
    //   - both shape(0) and shape(J) are divisible by the per-dtype tile.
    // Then swap J → position 1 to get canonical (I=0, J=1, batch...) and
    // coalesce adjacent batch dims that are proportional in both a and b.
    const int kTileDim = byte_size(dtype) <= 2 ? 64 : 32;

    int J = -1;
    for (int i = 1; i < rank; ++i) {
        if (b.stride(i) == 1) {
            J = i;
            break;
        }
    }

    bool is_transpose = (J >= 1) && (a.stride(0) == 1) && (a.stride(J) > 1) && (b.stride(0) > 1)
                        && (a.shape(0) % kTileDim == 0) && (a.shape(J) % kTileDim == 0);

    // TransposeCopy uses 16-byte atoms. Every row and batch base must satisfy
    // that alignment; otherwise the generic path chooses a legal copy width.
    is_transpose = is_transpose && reinterpret_cast<uintptr_t>(src.raw_data()) % 16 == 0
                   && reinterpret_cast<uintptr_t>(dst.raw_data()) % 16 == 0;
    for (int i = 0; is_transpose && i < rank; ++i) {
        is_transpose = (i == 0 || byte_size(dtype, a.stride(i)) % 16 == 0)
                       && (i == J || byte_size(dtype, b.stride(i)) % 16 == 0);
    }

    if (is_transpose) {
        if (J != 1) {
            a = a.transpose(1, J);
            b = b.transpose(1, J);
        }
        std::tie(a, b) = coalesce_copy_dims(a, b, 2);

        // TransposeCopy maps all logical tile axes onto bounded linear launches.
        if (a.rank() <= 4) {
            TransposeCopy(src.raw_data(), dst.raw_data(), a, b, dtype, stream);
            return;
        }
    }

    // --- Vectorized / scalar copy ---
    VectorizedCopy(src.raw_data(), dst.raw_data(), a, b, a.rank(), dtype, stream);
}

}  // namespace turbomind::core

#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/logger.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace turbomind::core {

// Forward declarations — defined in copy.cu and transpose.cu
void VectorizedCopy(
    const void* data_a, void* data_b, const Layout& a, const Layout& b, int rank, DataType dtype, cudaStream_t stream);

void TransposeCopy(
    const void* data_a, void* data_b, const Layout& a, const Layout& b, DataType dtype, cudaStream_t stream);

// Merge adjacent batch dims (positions ≥ 2) of (a, b) when their strides are
// proportional in BOTH a and b. Single forward pass over positions 3..rank-1.
//
// Precondition: a and b have the same shape and rank, with positions 0 and 1
// being the (I, J) transpose pair (not coalesceable). Only positions 2.. are
// considered batch dims.
//
// Why the joint-proportionality requirement?
// A batch dim is shared between src and dst: the kernel decodes blockIdx.z
// once into a multi-dim batch coord and dots it with the batch strides on
// BOTH sides to compute the per-block src and dst pointer offsets. Merging
// two adjacent batch dims into one collapses that pair of coords into a
// single linear index whose decode (idx / inner_shape, idx % inner_shape)
// only reproduces the original (outer_idx, inner_idx) — and therefore the
// original outer_idx*outer_stride + inner_idx*inner_stride offset — when
// outer_stride == inner_shape * inner_stride. Since the same merged index
// is dotted into both layouts, that proportionality must hold in BOTH a
// and b; otherwise the merged single-dim decode would land at different
// positions in src vs dst and produce wrong results.
static std::pair<Layout, Layout> coalesce_batch_dims(const Layout& a, const Layout& b)
{
    const int rank = a.rank();
    if (rank < 4)
        return {a, b};  // need ≥ 2 batch dims to merge

    std::vector<ssize_t> ash(a.shape().begin(), a.shape().begin() + 3);
    std::vector<ssize_t> ast(a.stride().begin(), a.stride().begin() + 3);
    std::vector<ssize_t> bsh(b.shape().begin(), b.shape().begin() + 3);
    std::vector<ssize_t> bst(b.stride().begin(), b.stride().begin() + 3);

    for (int i = 3; i < rank; ++i) {
        const ssize_t ai_sh = a.shape(i), ai_st = a.stride(i);
        const ssize_t bi_sh = b.shape(i), bi_st = b.stride(i);

        // Merge with the previously accumulated batch dim if its stride equals
        // shape * stride of that dim, in BOTH a and b.
        if (ai_st == ash.back() * ast.back() && bi_st == bsh.back() * bst.back()) {
            ash.back() *= ai_sh;
            bsh.back() *= bi_sh;
            // strides at the back stay unchanged (they remain the inner stride)
        }
        else {
            ash.push_back(ai_sh);
            ast.push_back(ai_st);
            bsh.push_back(bi_sh);
            bst.push_back(bi_st);
        }
    }

    return {Layout{ash, ast}, Layout{bsh, bst}};
}

// ============================================================================
// GenericCopy: layout normalization + dispatch
// ============================================================================
void GenericCopy(const Tensor& src, Tensor& dst, cudaStream_t stream)
{
    auto a = src.layout();
    auto b = dst.layout();

    TM_CHECK_EQ(a.size(), b.size()) << "GenericCopy: src and dst must have the same number of elements";

    // Sort strides ascending so innermost (fastest-varying) dim is first
    std::vector<int> idxs(a.rank());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) { return a.stride()[i] < a.stride()[j]; });

    a = a.permute(idxs);
    b = b.permute(idxs);

    a = a.coalesce();
    b = b.coalesce();

    int rank = std::max(a.rank(), b.rank());

    if (a.rank() < rank) {
        a = a.view(b.shape());
    }
    else if (b.rank() < rank) {
        b = b.view(a.shape());
    }

    const DataType dtype = src.dtype();

    // --- Transpose detection (2D + batched) ---
    // After the src-stride-ascending sort above, position 0 holds the smallest
    // src stride (innermost). We dispatch to TransposeCopy when:
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

    if (is_transpose) {
        if (J != 1) {
            a = a.transpose(1, J);
            b = b.transpose(1, J);
        }
        std::tie(a, b) = coalesce_batch_dims(a, b);

        // Compute total batch (product of post-coalesce batch dims, positions ≥ 2).
        int64_t total_batch = 1;
        for (int i = 2; i < a.rank(); ++i)
            total_batch *= a.shape(i);

        // Dispatch only when the kernel can handle it:
        //   1. post-coalesce rank ≤ 4 (only 2/3/4 are instantiated in TransposeCopy host),
        //   2. total_batch ≤ gridDim.z hardware limit (65535 on all current archs).
        // Otherwise, fall through to VectorizedCopy.
        if (a.rank() <= 4 && total_batch <= 65535) {
            TransposeCopy(src.raw_data(), dst.raw_data(), a, b, dtype, stream);
            return;
        }
    }

    // --- Vectorized / scalar copy ---
    VectorizedCopy(src.raw_data(), dst.raw_data(), a, b, a.rank(), dtype, stream);
}

}  // namespace turbomind::core

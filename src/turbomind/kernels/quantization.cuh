
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"

namespace turbomind {

#if 0
template<int threads, class T, int N>
__device__ Array<T, 2> find_minmax(const Array<T, N>& a)
{
    static_assert((threads & (threads - 1)) == 0);
    static_assert(sizeof(Array<T, 2>) == sizeof(uint32_t));
    uint32_t data;
    auto&    minmax = reinterpret_cast<Array<T, 2>&>(data);
    minmax          = {a[0], a[0]};
    PRAGMA_UNROLL
    for (int i = 1; i < N; ++i) {
        minmax = hmin(minmax[0], a[i]);
        minmax = hmax(minmax[1], a[i]);
    }
    PRAGMA_UNROLL
    for (int mask = threads / 2; mask > 0; mask /= 2) {
        uint32_t tmp = __shfl_xor_sync(uint32_t(-1), data, mask);
        auto&    vec = reinterpret_cast<Array<T, 2>&>(tmp);
        minmax[0]    = hmin(minmax[0], vec[0]);
        minmax[1]    = hmax(minmax[1], vec[1]);
    }
    return minmax;
}
#endif

template<int threads, class T, int N>
__device__ T find_absmax(const Array<T, N>& a)
{
    static_assert((threads & (threads - 1)) == 0);
    static_assert(sizeof(Array<T, 2>) == sizeof(uint32_t));
    uint32_t data;
    auto&    val = *reinterpret_cast<T*>(&data);
    val          = __habs(a[0]);
    PRAGMA_UNROLL
    for (int i = 1; i < N; ++i) {
        val = __hmax(val, __habs(a[i]));
    }
    PRAGMA_UNROLL
    for (int mask = threads / 2; mask > 0; mask /= 2) {
        uint32_t tmp = __shfl_xor_sync(uint32_t(-1), data, mask);
        auto&    x   = *reinterpret_cast<T*>(&tmp);
        val          = __hmax(val, x);
    }
    return val;
}

}  // namespace turbomind

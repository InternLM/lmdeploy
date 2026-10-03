// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/verification/kernel_sm80.cuh"

#include <cub/block/block_reduce.cuh>

namespace turbomind::verification_attention {

template<class T, int HeadDim>
__global__ void ReduceSplits(T*           out_ptr,
                             const float* partial_o_ptr,
                             const float* partial_ml_ptr,
                             int          query_count,
                             int          query_offset,
                             int          query_head_count,
                             int          split_count)
{
    __shared__ float maxima[128];
    __shared__ float sums[128];
    __shared__ float weights[128];
    __shared__ float global_max;
    __shared__ float global_sum;

    using BlockReduce = cub::BlockReduce<float, 256>;
    union ReduceStorage {
        typename BlockReduce::TempStorage maximum;
        typename BlockReduce::TempStorage sum;
    };
    __shared__ ReduceStorage reduce_storage;

    auto partial_o = cute::make_tensor(
        cute::make_gmem_ptr(partial_o_ptr),
        cute::make_layout(
            cute::make_shape(query_count, split_count, query_head_count, cute::Int<HeadDim>{}),
            cute::make_stride(split_count * query_head_count * HeadDim,
                              query_head_count * HeadDim,
                              cute::Int<HeadDim>{},
                              cute::_1{})));
    auto partial_ml = cute::make_tensor(
        cute::make_gmem_ptr(partial_ml_ptr),
        cute::make_layout(
            cute::make_shape(query_count, split_count, query_head_count, cute::_2{}),
            cute::make_stride(split_count * query_head_count * 2,
                              query_head_count * 2,
                              cute::_2{},
                              cute::_1{})));
    auto out = cute::make_tensor(
        cute::make_gmem_ptr(out_ptr),
        cute::make_layout(
            cute::make_shape(query_count, query_head_count, cute::Int<HeadDim>{}),
            cute::make_stride(query_head_count * HeadDim, cute::Int<HeadDim>{}, cute::_1{})));

    using ThreadLayout = cute::Layout<cute::Shape<cute::_256>>;
    auto thread_coordinates = cute::make_identity_tensor(cute::Shape<cute::_256>{});
    auto owned_coordinate = cute::local_partition(thread_coordinates, ThreadLayout{}, threadIdx.x);
    static_assert(cute::size(owned_coordinate) == 1);

    const int local_query = blockIdx.x;
    const int head       = blockIdx.y;
    const int coordinate = cute::get<0>(owned_coordinate(cute::_0{}));
    const int split      = coordinate;

    float thread_max = -CUDART_INF_F;
    if (split < split_count) {
        maxima[split] = partial_ml(local_query, split, head, cute::_0{});
        sums[split]    = partial_ml(local_query, split, head, cute::_1{});
        thread_max    = maxima[split];
    }
    thread_max = BlockReduce(reduce_storage.maximum).Reduce(thread_max, cub::Max{});
    if (threadIdx.x == 0) {
        global_max = thread_max;
    }
    __syncthreads();

    float thread_sum = 0.f;
    if (split < split_count) {
        const float weight = maxima[split] == -CUDART_INF_F
                                 ? 0.f
                                 : exp2f(maxima[split] - global_max);
        weights[split] = weight;
        thread_sum = weight * sums[split];
    }
    thread_sum = BlockReduce(reduce_storage.sum).Sum(thread_sum);
    if (threadIdx.x == 0) {
        global_sum = thread_sum;
    }
    __syncthreads();

    const int d = coordinate;
    if (d < HeadDim) {
        float numerator = 0.f;
        for (int split_index = 0; split_index < split_count; ++split_index) {
            numerator += weights[split_index] * partial_o(local_query, split_index, head, d);
        }
        out(local_query + query_offset, head, d) =
            global_sum == 0.f ? T(0) : static_cast<T>(numerator / global_sum);
    }
}

template<class T>
void DispatchReduce(const Arguments& arguments)
{
    dim3 grid(arguments.query_count, arguments.query_head_count);
    if (arguments.head_dim == 128) {
        ReduceSplits<T, 128><<<grid, 256, 0, arguments.stream>>>(
            static_cast<T*>(arguments.out),
            arguments.partial_o,
            arguments.partial_ml,
            arguments.query_count,
            arguments.query_offset,
            arguments.query_head_count,
            arguments.split_count);
    }
    else {
        ReduceSplits<T, 256><<<grid, 256, 0, arguments.stream>>>(
            static_cast<T*>(arguments.out),
            arguments.partial_o,
            arguments.partial_ml,
            arguments.query_count,
            arguments.query_offset,
            arguments.query_head_count,
            arguments.split_count);
    }
}

void Reduce(const Arguments& arguments)
{
    if (arguments.data_type == DataType::kHalf) {
        DispatchReduce<cutlass::half_t>(arguments);
    }
    else {
        DispatchReduce<cutlass::bfloat16_t>(arguments);
    }
}

}  // namespace turbomind::verification_attention

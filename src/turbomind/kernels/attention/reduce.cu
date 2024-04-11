// Copyright (c) OpenMMLab. All rights reserved.

#include "reduce.h"
#include "reduce_template.h"

namespace turbomind {

template<class T>
void dispatchReduce(T*           out,
                    float*       partial_M,
                    float*       partial_L,
                    float*       partial_O,
                    int*         signals,
                    const int*   split_cnt,
                    int          max_split_cnt,
                    int          dyn_split_cnt,
                    int          query_num,
                    int          head_num,
                    int          head_dim,
                    float        exp_scale,
                    cudaStream_t stream)
{
}

}  // namespace turbomind

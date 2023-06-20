// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/fastertransformer/utils/nccl_utils.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <mutex>

namespace fastertransformer {

struct NcclGuard {
    static constexpr int kMaxGroupCount = 32;

    static std::mutex& globalNcclMutex()
    {
        static std::mutex inst;
        return inst;
    }

    struct GroupState {
        std::mutex              mutex;
        std::condition_variable cv;
        int                     ref_count;
    };

    static GroupState& groupState(int group_id)
    {
        static std::array<GroupState, kMaxGroupCount> array{};
        FT_CHECK(group_id < kMaxGroupCount);
        return array[group_id];
    }

    NcclGuard(NcclParam tensor_para, cudaStream_t stream, bool barrier = false):
        tensor_para_(tensor_para), stream_(stream), barrier_(barrier)
    {
        if (is_active()) {
            auto& group = groupState(tensor_para_.group_id_);
            if (tensor_para_.rank_ == 0) {
                /// TODO: use std::optional after switching to C++17
                global_nccl_lock_ = std::make_unique<std::lock_guard<std::mutex>>(globalNcclMutex());
                {
                    std::lock_guard<std::mutex> lock(group.mutex);
                    group.ref_count = tensor_para_.world_size_;
                }
                group.cv.notify_all();
            }
            else {
                std::unique_lock<std::mutex> lock(group.mutex);
                group.cv.wait(lock, [&] { return group.ref_count > 0; });
            }
        }
    }

    ~NcclGuard()
    {
        if (is_active()) {
            ftNcclStreamSynchronize(tensor_para_, NcclParam{}, stream_);

            auto& group = groupState(tensor_para_.group_id_);

            int value = -1;
            {
                std::lock_guard<std::mutex> lock(group.mutex);
                value = --group.ref_count;
            }
            if (value == 0) {
                group.cv.notify_all();
            }
            else if (barrier_ || tensor_para_.rank_ == 0) {
                std::unique_lock<std::mutex> lock(group.mutex);
                group.cv.wait(lock, [&] { return group.ref_count == 0; });
            }

            // rank 0 unlocks global NCCL mutex automatically
        }
    }

    bool is_active()
    {
        return barrier_ || (ftNcclGroupCount() > 1 && tensor_para_.world_size_ > 1);
    }

    NcclParam                                    tensor_para_;
    cudaStream_t                                 stream_;
    bool                                         barrier_;
    std::unique_ptr<std::lock_guard<std::mutex>> global_nccl_lock_;
};

}  // namespace fastertransformer
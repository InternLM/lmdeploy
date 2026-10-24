/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>
#include <string_view>

namespace ft_nvtx {
static std::string scope;
std::string        getScope();
void               addScope(std::string name);
void               setScope(std::string name);
void               resetScope();
static int         domain = 0;
void               setDeviceDomain(int deviceId);
int                getDeviceDomain();
void               resetDeviceDomain();
bool               isEnableNvtx();

static bool has_read_nvtx_env = false;
static bool is_enable_ft_nvtx = false;
void        ftNvtxRangePush(std::string_view name);
void        ftNvtxRangePop();
}  // namespace ft_nvtx

namespace turbomind {

struct NvtxScope {
    explicit NvtxScope(std::string_view name): active_{ft_nvtx::isEnableNvtx()}
    {
        if (active_) {
            ft_nvtx::ftNvtxRangePush(name);
        }
    }

    NvtxScope(const NvtxScope&)            = delete;
    NvtxScope& operator=(const NvtxScope&) = delete;

    ~NvtxScope()
    {
        if (active_) {
            ft_nvtx::ftNvtxRangePop();
        }
    }

private:
    bool active_;
};

}  // namespace turbomind

#define PUSH_RANGE(name)                                                                                               \
    {                                                                                                                  \
        if (ft_nvtx::isEnableNvtx()) {                                                                                 \
            ft_nvtx::ftNvtxRangePush(name);                                                                            \
        }                                                                                                              \
    }

#define POP_RANGE                                                                                                      \
    {                                                                                                                  \
        if (ft_nvtx::isEnableNvtx()) {                                                                                 \
            ft_nvtx::ftNvtxRangePop();                                                                                 \
        }                                                                                                              \
    }

/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

namespace turbomind {

enum IA3_config
{
    KEY_ADAPTER   = 1 << 0,
    VALUE_ADAPTER = 1 << 1,
    MLP_ADAPTER   = 1 << 2,
};

static constexpr IA3_config IA3_NONE                    = static_cast<IA3_config>(0);
static constexpr size_t     IA3_ADAPTER_MAX_NUM_ENCODER = 3;
static constexpr size_t     IA3_ADAPTER_MAX_NUM_DECODER = 5;

static inline IA3_config operator&(IA3_config x, IA3_config y)
{
    return static_cast<IA3_config>(static_cast<int>(x) & static_cast<int>(y));
}

static inline IA3_config operator|(IA3_config x, IA3_config y)
{
    return static_cast<IA3_config>(static_cast<int>(x) | static_cast<int>(y));
}

static inline IA3_config& operator|=(IA3_config& x, IA3_config y)
{
    return x = static_cast<IA3_config>(static_cast<int>(x) | static_cast<int>(y));
}

}  // namespace turbomind

/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "stdio.h"
#include "stdlib.h"

// be consistent with FasterTransformer
int8_t float_to_int8_rn_host(float x)
{
    int8_t  res;
    int32_t tmp;
    if (x >= 0) {
        tmp = int(x + 0.5);
        tmp = tmp > 127 ? 127 : tmp;
        res = int8_t(tmp);
    }
    else {
        tmp = int(x - 0.5);
        tmp = tmp < -127 ? -127 : tmp;
        res = int8_t(tmp);
    }
    return res;
}

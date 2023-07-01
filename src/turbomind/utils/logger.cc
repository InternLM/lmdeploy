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

#include "src/turbomind/utils/logger.h"
#include <cuda_runtime.h>

namespace turbomind {

Logger::Logger()
{
    char* is_first_rank_only_char = std::getenv("FT_LOG_FIRST_RANK_ONLY");
    bool  is_first_rank_only =
        (is_first_rank_only_char != nullptr && std::string(is_first_rank_only_char) == "ON") ? true : false;

    int device_id;
    cudaGetDevice(&device_id);

    char* level_name = std::getenv("FT_LOG_LEVEL");
    if (level_name != nullptr) {
        std::map<std::string, Level> name_to_level = {
            {"TRACE", TRACE},
            {"DEBUG", DEBUG},
            {"INFO", INFO},
            {"WARNING", WARNING},
            {"ERROR", ERROR},
        };
        auto level = name_to_level.find(level_name);
        // If FT_LOG_FIRST_RANK_ONLY=ON, set LOG LEVEL of other device to ERROR
        if (is_first_rank_only && device_id != 0) {
            level = name_to_level.find("ERROR");
        }
        if (level != name_to_level.end()) {
            setLevel(level->second);
        }
        else {
            fprintf(stderr,
                    "[FT][WARNING] Invalid logger level FT_LOG_LEVEL=%s. "
                    "Ignore the environment variable and use a default "
                    "logging level.\n",
                    level_name);
            level_name = nullptr;
        }
    }
}

}  // namespace turbomind

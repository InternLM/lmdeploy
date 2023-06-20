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

#include "word_list.h"
#include "memory_utils.h"

#include "assert.h"

namespace fastertransformer {

int read_word_list(const std::string& filename, std::vector<int>& file_data)
{
    std::ifstream word_list_file(filename, std::ios::in);

    std::string line_buf;
    int         line_count   = 0;
    size_t      id_counts[2] = {0, 0};
    while (std::getline(word_list_file, line_buf)) {

        std::stringstream line_stream(line_buf);
        std::string       vals;
        while (std::getline(line_stream, vals, ',')) {
            file_data.push_back(std::stoi(vals));
            id_counts[line_count]++;
        }
        line_count++;

        if (line_count > 1) {
            break;
        }
    }
    assert(id_counts[0] == id_counts[1]);

    return 0;
}

}  // namespace fastertransformer

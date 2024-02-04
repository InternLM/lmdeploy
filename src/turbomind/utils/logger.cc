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
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/hourly_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>

namespace turbomind {

Logger::Logger()
{
    char* log_path = std::getenv("TM_LOG_PATH");
    if (log_path != nullptr) {
        SpdLogger::get_instance().set_log_path(std::string(log_path));
    }
    else {
#ifndef _MSC_VER
        SpdLogger::get_instance().set_log_path("/var/log/lmdeploy.log");
#else
        SpdLogger::get_instance().set_log_path("C:\Users\lmdeploy.log");
#endif
    }
    SpdLogger::get_instance().init();
    char* is_first_rank_only_char = std::getenv("TM_LOG_FIRST_RANK_ONLY");
    bool  is_first_rank_only =
        (is_first_rank_only_char != nullptr && std::string(is_first_rank_only_char) == "ON") ? true : false;

    int device_id;
    cudaGetDevice(&device_id);

    char* level_name = std::getenv("TM_LOG_LEVEL");
    if (level_name != nullptr) {
        std::map<std::string, Level> name_to_level = {
            {"TRACE", TRACE},
            {"DEBUG", DEBUG},
            {"INFO", INFO},
            {"WARNING", WARNING},
            {"ERROR", ERROR},
        };
        auto level = name_to_level.find(level_name);
        // If TM_LOG_FIRST_RANK_ONLY=ON, set LOG LEVEL of other device to ERROR
        if (is_first_rank_only && device_id != 0) {
            level = name_to_level.find("ERROR");
        }
        if (level != name_to_level.end()) {
            setLevel(level->second);
        }
        else {
            fprintf(stderr,
                    "[TM][WARNING] Invalid logger level TM_LOG_LEVEL=%s. "
                    "Ignore the environment variable and use a default "
                    "logging level.\n",
                    level_name);
            level_name = nullptr;
        }
    }
}

void SpdLogger::init()
{
    if (inited_) {
        return;
    }

    spdlog::init_thread_pool(8192, 1);

    // rotate 500 MB
    auto basic_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(path_, 500 * 1024 * 1024, 0);
    // async
    auto logger = std::make_shared<spdlog::async_logger>(
        "logger", basic_sink, spdlog::thread_pool(), spdlog::async_overflow_policy::overrun_oldest);
    logger->set_level(spdlog::level::trace);
    // ms, thread_id
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%t] %v");

    logger_ = logger;

    spdlog::register_logger(logger_);

    // real-time refresh
    logger_->flush_on(spdlog::level::trace);

    inited_ = true;
}

}  // namespace turbomind

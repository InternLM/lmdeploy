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

#include <cstdlib>
#include <map>
#include <string>

#include "src/turbomind/utils/string_utils.h"

namespace fastertransformer {

class Logger {

public:
    enum Level {
        TRACE   = 0,
        DEBUG   = 10,
        INFO    = 20,
        WARNING = 30,
        ERROR   = 40
    };

    static Logger& getLogger()
    {
        thread_local Logger instance;
        return instance;
    }
    Logger(Logger const&)         = delete;
    void operator=(Logger const&) = delete;

    template<typename... Args>
    void log(const Level level, const std::string format, const Args&... args)
    {
        if (level_ <= level) {
            std::string fmt = getPrefix(level) + format + "\n";
            // FILE*       out    = level_ < WARNING ? stdout : stderr;
            std::string logstr = fmtstr(fmt, args...);
            fprintf(stderr, "%s", logstr.c_str());
        }
    }

    template<typename... Args>
    void log(const Level level, const int rank, const std::string format, const Args&... args)
    {
        if (level_ <= level) {
            std::string fmt = getPrefix(level, rank) + format + "\n";
            // FILE*       out    = level_ < WARNING ? stdout : stderr;
            std::string logstr = fmtstr(fmt, args...);
            fprintf(stderr, "%s", logstr.c_str());
        }
    }

    void setLevel(const Level level)
    {
        level_ = level;
        log(INFO, "Set logger level by %s", getLevelName(level).c_str());
    }

    int getLevel() const
    {
        return level_;
    }

private:
    const std::string                              PREFIX      = "[FT]";
    const std::map<const Level, const std::string> level_name_ = {
        {TRACE, "TRACE"}, {DEBUG, "DEBUG"}, {INFO, "INFO"}, {WARNING, "WARNING"}, {ERROR, "ERROR"}};

#ifndef NDEBUG
    const Level DEFAULT_LOG_LEVEL = DEBUG;
#else
    const Level DEFAULT_LOG_LEVEL = INFO;
#endif
    Level level_ = DEFAULT_LOG_LEVEL;

    Logger();

    inline const std::string getLevelName(const Level level)
    {
        return level_name_.at(level);
    }

    inline const std::string getPrefix(const Level level)
    {
        return PREFIX + "[" + getLevelName(level) + "] ";
    }

    inline const std::string getPrefix(const Level level, const int rank)
    {
        return PREFIX + "[" + getLevelName(level) + "][" + std::to_string(rank) + "] ";
    }
};

#define FT_LOG(level, ...)                                                                                             \
    do {                                                                                                               \
        if (fastertransformer::Logger::getLogger().getLevel() <= level) {                                              \
            fastertransformer::Logger::getLogger().log(level, __VA_ARGS__);                                            \
        }                                                                                                              \
    } while (0)

#define FT_LOG_TRACE(...) FT_LOG(fastertransformer::Logger::TRACE, __VA_ARGS__)
#define FT_LOG_DEBUG(...) FT_LOG(fastertransformer::Logger::DEBUG, __VA_ARGS__)
#define FT_LOG_INFO(...) FT_LOG(fastertransformer::Logger::INFO, __VA_ARGS__)
#define FT_LOG_WARNING(...) FT_LOG(fastertransformer::Logger::WARNING, __VA_ARGS__)
#define FT_LOG_ERROR(...) FT_LOG(fastertransformer::Logger::ERROR, __VA_ARGS__)
}  // namespace fastertransformer

// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <fmt/format.h>

#include <string>

#include "src/turbomind/core/check.h"

namespace turbomind::core {

class Logger {
public:
    enum class Level {
        kTrace   = 0,
        kDebug   = 10,
        kInfo    = 20,
        kWarning = 30,
        kError   = 40,
    };

    // Returns the thread-local Logger instance.
    static Logger& Instance();

    Logger(const Logger&)            = delete;
    Logger& operator=(const Logger&) = delete;

    template<typename... Args>
    void Log(Level level, fmt::format_string<Args...> fmt_str, Args&&... args)
    {
        if (level_ <= level) {
            Enqueue(level, fmt::format(fmt_str, std::forward<Args>(args)...));
        }
    }

    void set_level(Level level);

    Level get_level() const
    {
        return level_;
    }

    // Blocks until all previously enqueued log records have been written.
    // In sync mode this is a no-op (output is already written by the caller).
    static void Flush();

    bool is_async() const
    {
        return async_;
    }

private:
    Logger();

    void Enqueue(Level level, std::string message);

    static std::string LevelName(Level level);
    static std::string Prefix(Level level);

#ifndef NDEBUG
    Level level_ = Level::kDebug;
#else
    Level level_ = Level::kInfo;
#endif
    bool async_ = true;
};

}  // namespace turbomind::core

// ---------------------------------------------------------------------------
// Convenience macros — distinct from the old TM_LOG_* to avoid collisions.
// ---------------------------------------------------------------------------
#define TM2_LOG(level, ...)                                                                                            \
    do {                                                                                                               \
        if (turbomind::core::Logger::Instance().get_level() <= (level)) {                                              \
            turbomind::core::Logger::Instance().Log((level), __VA_ARGS__);                                             \
        }                                                                                                              \
    } while (0)

#define TM2_LOG_TRACE(...)   TM2_LOG(turbomind::core::Logger::Level::kTrace, __VA_ARGS__)
#define TM2_LOG_DEBUG(...)   TM2_LOG(turbomind::core::Logger::Level::kDebug, __VA_ARGS__)
#define TM2_LOG_INFO(...)    TM2_LOG(turbomind::core::Logger::Level::kInfo, __VA_ARGS__)
#define TM2_LOG_WARNING(...) TM2_LOG(turbomind::core::Logger::Level::kWarning, __VA_ARGS__)
#define TM2_LOG_ERROR(...)   TM2_LOG(turbomind::core::Logger::Level::kError, __VA_ARGS__)

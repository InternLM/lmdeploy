// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <fmt/format.h>

#include <cstdlib>
#include <string>

#include "src/turbomind/core/check.h"

namespace turbomind::core {

struct SourceLocation {
    const char* file;
    int         line;
};

class Logger {
public:
    enum class Level
    {
        kTrace   = 0,
        kDebug   = 10,
        kInfo    = 20,
        kWarning = 30,
        kError   = 40,
        kFatal   = 50,
    };

    // Returns the thread-local Logger instance.
    static Logger& Instance();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    template<typename... Args>
    void Log(Level level, fmt::format_string<Args...> fmt_str, Args&&... args)
    {
        if (level_ <= level) {
            Enqueue(level, nullptr, 0, fmt::format(fmt_str, std::forward<Args>(args)...));
        }
    }

    template<typename... Args>
    void Log(Level level, SourceLocation loc, fmt::format_string<Args...> fmt_str, Args&&... args)
    {
        if (level_ <= level) {
            Enqueue(level, loc.file, loc.line, fmt::format(fmt_str, std::forward<Args>(args)...));
        }
    }

    template<typename... Args>
    [[noreturn]] void LogFatal(SourceLocation loc, fmt::format_string<Args...> fmt_str, Args&&... args)
    {
        Enqueue(Level::kFatal, loc.file, loc.line, fmt::format(fmt_str, std::forward<Args>(args)...));
        std::abort();
    }

    void set_level(Level level);

    Level get_level() const
    {
        return level_;
    }

    bool is_async() const
    {
        return async_;
    }

private:
    Logger();

    void Enqueue(Level level, std::string message);
    void Enqueue(Level level, const char* file, int line, std::string message);

    static std::string LevelName(Level level);
    static std::string Prefix(Level level, const char* file, int line);

#ifndef NDEBUG
    Level level_ = Level::kDebug;
#else
    Level level_ = Level::kInfo;
#endif
    bool async_ = true;
};

}  // namespace turbomind::core

// ---------------------------------------------------------------------------
// Convenience macros
// ---------------------------------------------------------------------------
#define TM_LOG(level, ...)                                                                                             \
    do {                                                                                                               \
        if (turbomind::core::Logger::Instance().get_level() <= (level)) {                                              \
            turbomind::core::Logger::Instance().Log(                                                                   \
                (level), turbomind::core::SourceLocation{__FILE__, __LINE__}, __VA_ARGS__);                            \
        }                                                                                                              \
    } while (0)

#define TM_LOG_TRACE(...) TM_LOG(turbomind::core::Logger::Level::kTrace, __VA_ARGS__)
#define TM_LOG_DEBUG(...) TM_LOG(turbomind::core::Logger::Level::kDebug, __VA_ARGS__)
#define TM_LOG_INFO(...) TM_LOG(turbomind::core::Logger::Level::kInfo, __VA_ARGS__)
#define TM_LOG_WARN(...) TM_LOG(turbomind::core::Logger::Level::kWarning, __VA_ARGS__)
#define TM_LOG_ERROR(...) TM_LOG(turbomind::core::Logger::Level::kError, __VA_ARGS__)
#define TM_LOG_FATAL(...)                                                                                               \
    do {                                                                                                                 \
        turbomind::core::Logger::Instance().LogFatal(                                                                   \
            turbomind::core::SourceLocation{__FILE__, __LINE__}, __VA_ARGS__);                                          \
    } while (0)

#define TM_LOG_WARNING(...) TM_LOG_WARN(__VA_ARGS__)

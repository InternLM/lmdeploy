// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/logger.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <string_view>
#include <thread>

#include <blockingconcurrentqueue.h>
#include <fmt/color.h>

namespace turbomind::core {

// ---------------------------------------------------------------------------
// Color palette per log level
// ---------------------------------------------------------------------------
static fmt::text_style StyleFor(Logger::Level level)
{
    switch (level) {
        case Logger::Level::kTrace:
            return fmt::fg(fmt::color::gray);
        case Logger::Level::kDebug:
            return fmt::fg(fmt::color::cyan);
        case Logger::Level::kInfo:
            return fmt::fg(fmt::color::green);
        case Logger::Level::kWarning:
            return fmt::fg(fmt::color::yellow);
        case Logger::Level::kError:
            return fmt::fg(fmt::color::red) | fmt::emphasis::bold;
        default:
            return {};
    }
}

// ---------------------------------------------------------------------------
// AsyncLogWorker internals — entirely private to this translation unit
// ---------------------------------------------------------------------------

enum class RecordKind { kNormal, kFlush, kStop };

struct LogRecord {
    RecordKind          kind          = RecordKind::kNormal;
    Logger::Level       level         = Logger::Level::kInfo;
    std::string         message;
    std::promise<void>* flush_promise = nullptr;
};

class AsyncLogWorker {
public:
    static AsyncLogWorker& Instance();

    AsyncLogWorker(const AsyncLogWorker&)            = delete;
    AsyncLogWorker& operator=(const AsyncLogWorker&) = delete;

    void Enqueue(LogRecord record);
    void Flush();

    ~AsyncLogWorker();

private:
    AsyncLogWorker();

    void Run();

    moodycamel::BlockingConcurrentQueue<LogRecord> queue_;
    std::thread                                    thread_;
};

// ---------------------------------------------------------------------------
// Logger — thread-local frontend
// ---------------------------------------------------------------------------

Logger& Logger::Instance()
{
    thread_local Logger inst;
    return inst;
}

Logger::Logger()
{
    const char* async_env = std::getenv("TM_LOG_ASYNC");
    if (async_env != nullptr && std::string_view{async_env} == "0") {
        async_ = false;
    }

    const char* level_env = std::getenv("TM_LOG_LEVEL");
    if (level_env == nullptr) {
        return;
    }
    else {
        using Entry = std::pair<std::string_view, Level>;
        static constexpr std::array<Entry, 5> kNameToLevel = {{
            {"TRACE", Level::kTrace},
            {"DEBUG", Level::kDebug},
            {"INFO", Level::kInfo},
            {"WARNING", Level::kWarning},
            {"ERROR", Level::kError},
        }};

        const std::string_view name{level_env};
        auto                   it =
            std::find_if(kNameToLevel.begin(), kNameToLevel.end(), [&](const Entry& e) { return e.first == name; });
        if (it != kNameToLevel.end()) {
            level_ = it->second;
        }
        else {
            fmt::print(stderr,
                       StyleFor(Level::kWarning),
                       "[TM][WARNING] Invalid TM_LOG_LEVEL='{}'. Using default level.\n",
                       level_env);
        }
    }
}

void Logger::set_level(Level level)
{
    level_ = level;
}

void Logger::Flush()
{
    if (TM_LIKELY(Instance().async_)) {
        AsyncLogWorker::Instance().Flush();
    }
    else {
        std::fflush(stderr);
    }
}

std::string Logger::LevelName(Level level)
{
    switch (level) {
        case Level::kTrace:
            return "TRACE";
        case Level::kDebug:
            return "DEBUG";
        case Level::kInfo:
            return "INFO";
        case Level::kWarning:
            return "WARNING";
        case Level::kError:
            return "ERROR";
        default:
            return "UNKNOWN";
    }
}

std::string Logger::Prefix(Level level)
{
    return fmt::format("[TM][{}] ", LevelName(level));
}

void Logger::Enqueue(Level level, std::string message)
{
    std::string line = Prefix(level);
    line.reserve(line.size() + message.size() + 1);
    line.append(std::move(message));
    line += '\n';
    if (TM_LIKELY(async_)) {
        LogRecord record;
        record.level   = level;
        record.message = std::move(line);
        AsyncLogWorker::Instance().Enqueue(std::move(record));
    }
    else {
        fmt::print(stderr, StyleFor(level), "{}", line);
    }
}

// ---------------------------------------------------------------------------
// AsyncLogWorker — background I/O thread
// ---------------------------------------------------------------------------

AsyncLogWorker& AsyncLogWorker::Instance()
{
    static AsyncLogWorker worker;
    return worker;
}

AsyncLogWorker::AsyncLogWorker() : thread_(&AsyncLogWorker::Run, this) {}

AsyncLogWorker::~AsyncLogWorker()
{
    LogRecord stop;
    stop.kind = RecordKind::kStop;
    queue_.enqueue(std::move(stop));
    thread_.join();
}

void AsyncLogWorker::Enqueue(LogRecord record)
{
    queue_.enqueue(std::move(record));
}

void AsyncLogWorker::Flush()
{
    std::promise<void> promise;
    auto               future = promise.get_future();

    LogRecord sentinel;
    sentinel.kind          = RecordKind::kFlush;
    sentinel.flush_promise = &promise;
    queue_.enqueue(std::move(sentinel));

    future.get();
}

void AsyncLogWorker::Run()
{
    LogRecord record;
    while (true) {
        queue_.wait_dequeue(record);

        if (record.kind == RecordKind::kStop) {
            break;
        }
        else if (record.kind == RecordKind::kFlush) {
            record.flush_promise->set_value();
        }
        else {
            fmt::print(stderr, StyleFor(record.level), "{}", record.message);
        }
    }
}

}  // namespace turbomind::core

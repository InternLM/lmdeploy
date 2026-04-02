// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/logger.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string_view>
#include <thread>
#ifndef _WIN32
#include <unistd.h>
#endif

#include <blockingconcurrentqueue.h>
#include <fmt/color.h>

namespace turbomind::core {

// ---------------------------------------------------------------------------
// Timestamp: MMDD.HH:MM:SS.uuuuuu (dot connects date and time).
// Same approach as glog: system_clock::now(), to_time_t, localtime_r; microseconds
// as time since start of current second (no modulo).
// ---------------------------------------------------------------------------
static std::string Timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(now - std::chrono::system_clock::from_time_t(t)).count();
    std::tm tm_buf;
#ifdef _WIN32
    if (::localtime_s(&tm_buf, &t) != 0) {
        return "0000.00:00:00.000000";
    }
    std::tm* tm = &tm_buf;
#else
    std::tm* tm = ::localtime_r(&t, &tm_buf);
    if (tm == nullptr) {
        return "0000.00:00:00.000000";
    }
#endif
    return fmt::format("{:02}{:02}.{:02}:{:02}:{:02}.{:06}",
                       tm->tm_mon + 1,
                       tm->tm_mday,
                       tm->tm_hour,
                       tm->tm_min,
                       tm->tm_sec,
                       static_cast<int>(us));
}

// ---------------------------------------------------------------------------
// Basename of __FILE__ (substring after last '/')
// ---------------------------------------------------------------------------
static const char* Basename(const char* file)
{
    const char* last_sep = std::strrchr(file, '/');
#ifdef _WIN32
    const char* last_bs = std::strrchr(file, '\\');
    if (last_bs && (!last_sep || last_bs > last_sep)) {
        last_sep = last_bs;
    }
#endif
    return last_sep ? last_sep + 1 : file;
}

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
            return {};
        case Logger::Level::kWarning:
            return fmt::fg(fmt::color::yellow);
        case Logger::Level::kError:
            return fmt::fg(fmt::color::red) | fmt::emphasis::bold;
        case Logger::Level::kFatal:
            return fmt::fg(fmt::color::red) | fmt::emphasis::bold;
        default:
            return {};
    }
}

// ---------------------------------------------------------------------------
// Color auto-detection: TM_LOG_COLOR env var → isatty fallback
// ---------------------------------------------------------------------------
static bool UseColor()
{
    static const bool kUseColor = [] {
        const char* env = std::getenv("TM_LOG_COLOR");
        if (env != nullptr) {
            return std::string_view{env} != "0";
        }
#ifndef _WIN32
        return ::isatty(STDERR_FILENO) != 0;
#else
        return true;
#endif
    }();
    return kUseColor;
}

static void PrintStyled(Logger::Level level, std::string_view msg)
{
    if (UseColor()) {
        fmt::print(stderr, StyleFor(level), "{}", msg);
    }
    else {
        fmt::print(stderr, "{}", msg);
    }
}

// ---------------------------------------------------------------------------
// AsyncLogWorker internals — entirely private to this translation unit
// ---------------------------------------------------------------------------

enum class RecordKind
{
    kNormal,
    kStop
};

struct LogRecord {
    RecordKind    kind  = RecordKind::kNormal;
    Logger::Level level = Logger::Level::kInfo;
    std::string   message;
};

class AsyncLogWorker {
public:
    static AsyncLogWorker& Instance();

    AsyncLogWorker(const AsyncLogWorker&) = delete;
    AsyncLogWorker& operator=(const AsyncLogWorker&) = delete;

    void Enqueue(LogRecord record);
    void Stop();
    void OnSignal();

    ~AsyncLogWorker();

private:
    AsyncLogWorker();

    void Run();

    moodycamel::BlockingConcurrentQueue<LogRecord> queue_;
    std::thread                                    thread_;
    std::atomic_flag                               stopped_ = ATOMIC_FLAG_INIT;

    std::atomic<bool> signal_shutdown_{false};
    std::atomic<bool> signal_drain_done_{false};
    std::atomic<bool> worker_ready_{false};
    std::thread::id   worker_thread_id_{};
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
        using Entry                                        = std::pair<std::string_view, Level>;
        static constexpr std::array<Entry, 7> kNameToLevel = {{
            {"TRACE", Level::kTrace},
            {"DEBUG", Level::kDebug},
            {"INFO", Level::kInfo},
            {"WARN", Level::kWarning},
            {"WARNING", Level::kWarning},
            {"ERROR", Level::kError},
            {"FATAL", Level::kFatal},
        }};

        const std::string name_upper = [&] {
            std::string s{level_env};
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
            return s;
        }();
        auto it = std::find_if(
            kNameToLevel.begin(), kNameToLevel.end(), [&](const Entry& e) { return e.first == name_upper; });
        if (it != kNameToLevel.end()) {
            level_ = it->second;
        }
        else {
            PrintStyled(Level::kWarning,
                        fmt::format("[TM][WARN] Invalid TM_LOG_LEVEL='{}'. Using default level.\n", level_env));
        }
    }
}

void Logger::set_level(Level level)
{
    level_ = level;
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
            return "WARN";
        case Level::kError:
            return "ERROR";
        case Level::kFatal:
            return "FATAL";
        default:
            return "UNKNOWN";
    }
}

std::string Logger::Prefix(Level level, const char* file, int line)
{
    std::string s = fmt::format("[TM][{}][{}]", LevelName(level), Timestamp());
    if (file != nullptr) {
        s += fmt::format("[{}:{}]", Basename(file), line);
    }
    s += " ";
    return s;
}

void Logger::Enqueue(Level level, const char* file, int line, std::string message)
{
    std::string line_str = Prefix(level, file, line);
    line_str.reserve(line_str.size() + message.size() + 1);
    line_str.append(std::move(message));
    line_str += '\n';
    if (TM_LIKELY(async_ && level != Level::kFatal)) {
        LogRecord record;
        record.level   = level;
        record.message = std::move(line_str);
        AsyncLogWorker::Instance().Enqueue(std::move(record));
    }
    else {
        PrintStyled(level, line_str);
    }
}

void Logger::Enqueue(Level level, std::string message)
{
    Enqueue(level, nullptr, 0, std::move(message));
}

// ---------------------------------------------------------------------------
// AsyncLogWorker — background I/O thread
// ---------------------------------------------------------------------------

AsyncLogWorker& AsyncLogWorker::Instance()
{
    static AsyncLogWorker worker;
    return worker;
}

static void OnFatalSignal(int signum)
{
    AsyncLogWorker::Instance().OnSignal();
    ::signal(signum, SIG_DFL);
    ::raise(signum);
}

AsyncLogWorker::AsyncLogWorker()
{
    thread_ = std::thread(&AsyncLogWorker::Run, this);
    while (!worker_ready_.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    const char* signals_env = std::getenv("TM_LOG_SIGNALS");
    if (signals_env == nullptr || std::string_view{signals_env} != "0") {
        for (int sig : {SIGSEGV, SIGABRT, SIGFPE, SIGILL, SIGBUS}) {
            ::signal(sig, OnFatalSignal);
        }
    }
}

void AsyncLogWorker::OnSignal()
{
    stopped_.test_and_set();

    if (std::this_thread::get_id() == worker_thread_id_) {
        LogRecord record;
        while (queue_.try_dequeue(record)) {
            if (record.kind != RecordKind::kStop) {
                PrintStyled(record.level, record.message);
            }
        }
    }
    else {
        signal_shutdown_.store(true, std::memory_order_release);
        for (int i = 0; i < 2000; ++i) {
            if (signal_drain_done_.load(std::memory_order_acquire)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void AsyncLogWorker::Stop()
{
    if (stopped_.test_and_set()) {
        return;  // already stopping or stopped
    }
    LogRecord stop;
    stop.kind = RecordKind::kStop;
    queue_.enqueue(std::move(stop));
    thread_.join();
}

AsyncLogWorker::~AsyncLogWorker()
{
    Stop();
}

void AsyncLogWorker::Enqueue(LogRecord record)
{
    queue_.enqueue(std::move(record));
}

void AsyncLogWorker::Run()
{
    worker_thread_id_ = std::this_thread::get_id();
    worker_ready_.store(true, std::memory_order_release);

    LogRecord record;
    while (true) {
        bool got = queue_.wait_dequeue_timed(record, std::chrono::milliseconds(100));

        if (got) {
            if (record.kind == RecordKind::kStop) {
                while (queue_.try_dequeue(record)) {
                    if (record.kind != RecordKind::kStop) {
                        PrintStyled(record.level, record.message);
                    }
                }
                signal_drain_done_.store(true, std::memory_order_release);
                return;
            }
            PrintStyled(record.level, record.message);
        }

        if (signal_shutdown_.load(std::memory_order_acquire)) {
            while (queue_.try_dequeue(record)) {
                if (record.kind != RecordKind::kStop) {
                    PrintStyled(record.level, record.message);
                }
            }
            signal_drain_done_.store(true, std::memory_order_release);
            return;
        }
    }
}

}  // namespace turbomind::core

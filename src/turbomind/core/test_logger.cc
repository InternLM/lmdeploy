// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/logger.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <functional>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using turbomind::core::Logger;

// Tests run in sync mode (TM_LOG_ASYNC=0) so log output is written inline and
// CaptureStderr sees complete output when fn() returns.
static bool SetSyncModeForTests()
{
    return ::setenv("TM_LOG_ASYNC", "0", 1) == 0;
}
static bool kSyncMode = SetSyncModeForTests();

// ---------------------------------------------------------------------------
// Stderr capture helper
// ---------------------------------------------------------------------------
// Redirects stderr to a pipe, runs `fn`, restores stderr, returns what was written.
// A background reader drains the pipe so it never fills (avoids blocking when
// multiple threads write in sync mode). Requires sync mode (TM_LOG_ASYNC=0).
static std::string CaptureStderr(std::function<void()> fn)
{
    int saved = ::dup(STDERR_FILENO);
    REQUIRE(saved >= 0);

    std::array<int, 2> pipefd{};
    REQUIRE(::pipe(pipefd.data()) == 0);
    REQUIRE(::dup2(pipefd[1], STDERR_FILENO) >= 0);
    ::close(pipefd[1]);

    std::string output;
    std::thread reader([&output, read_fd = pipefd[0]]() {
        char    buf[4096];
        ssize_t n;
        while ((n = ::read(read_fd, buf, sizeof(buf))) > 0) {
            output.append(buf, static_cast<size_t>(n));
        }
        ::close(read_fd);
    });

    fn();

    REQUIRE(::dup2(saved, STDERR_FILENO) >= 0);
    ::close(saved);
    reader.join();
    return output;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("Logger: SetLevel / GetLevel round-trip", "[logger]")
{
    auto& log = Logger::Instance();

    log.set_level(Logger::Level::kTrace);
    REQUIRE(log.get_level() == Logger::Level::kTrace);

    log.set_level(Logger::Level::kWarning);
    REQUIRE(log.get_level() == Logger::Level::kWarning);

    log.set_level(Logger::Level::kError);
    REQUIRE(log.get_level() == Logger::Level::kError);

    // Restore default
    log.set_level(Logger::Level::kDebug);
}

TEST_CASE("Logger: prefix format", "[logger]")
{
    auto& log = Logger::Instance();
    log.set_level(Logger::Level::kTrace);

    auto output = CaptureStderr([&] { log.Log(Logger::Level::kInfo, "hello"); });

    // New format: [TM][MMDD HH:MM:SS.uuuuuu][LEVEL] message (no file/line for direct Log)
    REQUIRE(output.find("[TM]") != std::string::npos);
    REQUIRE(output.find("[INFO]") != std::string::npos);
    REQUIRE(output.find("hello") != std::string::npos);
}

TEST_CASE("Logger: format arguments", "[logger]")
{
    auto& log = Logger::Instance();
    log.set_level(Logger::Level::kTrace);

    auto output = CaptureStderr([&] {
        log.Log(Logger::Level::kDebug, "int={} float={:.2f} str={}", 42, 3.14f, std::string("world"));
    });

    REQUIRE(output.find("int=42") != std::string::npos);
    REQUIRE(output.find("float=3.14") != std::string::npos);
    REQUIRE(output.find("str=world") != std::string::npos);
}

TEST_CASE("Logger: level filtering", "[logger]")
{
    auto& log = Logger::Instance();
    log.set_level(Logger::Level::kWarning);

    auto output = CaptureStderr([&] {
        log.Log(Logger::Level::kDebug, "should be suppressed");
        log.Log(Logger::Level::kInfo, "also suppressed");
        log.Log(Logger::Level::kWarning, "should appear");
        log.Log(Logger::Level::kError, "also appears");
    });

    REQUIRE(output.find("should be suppressed") == std::string::npos);
    REQUIRE(output.find("also suppressed") == std::string::npos);
    REQUIRE(output.find("should appear") != std::string::npos);
    REQUIRE(output.find("also appears") != std::string::npos);

    // Restore
    log.set_level(Logger::Level::kDebug);
}

TEST_CASE("Logger: TM_LOG_LEVEL env var", "[logger]")
{
    // This must use a fresh thread to get a new thread_local Logger instance
    // that picks up the env var in its constructor.
    ::setenv("TM_LOG_LEVEL", "WARNING", /*overwrite=*/1);

    std::string output;
    {
        std::thread t([&] {
            output = CaptureStderr([&] {
                auto& log = Logger::Instance();
                log.Log(Logger::Level::kDebug, "env-suppressed");
                log.Log(Logger::Level::kWarning, "env-visible");
            });
        });
        t.join();
    }

    ::unsetenv("TM_LOG_LEVEL");

    REQUIRE(output.find("env-suppressed") == std::string::npos);
    REQUIRE(output.find("env-visible") != std::string::npos);
}

TEST_CASE("Logger: macros emit correct prefix", "[logger]")
{
    Logger::Instance().set_level(Logger::Level::kTrace);

    auto output = CaptureStderr([&] {
        TM_LOG_TRACE("trace-msg");
        TM_LOG_DEBUG("debug-msg");
        TM_LOG_INFO("info-msg");
        TM_LOG_WARN("warn-msg");
        TM_LOG_ERROR("error-msg");
    });

    // Format: [TM][MMDD HH:MM:SS.uuuuuu][LEVEL][basename:line] message
    REQUIRE(output.find("[TM]") != std::string::npos);
    REQUIRE(output.find("[TRACE]") != std::string::npos);
    REQUIRE(output.find("[DEBUG]") != std::string::npos);
    REQUIRE(output.find("[INFO]") != std::string::npos);
    REQUIRE(output.find("[WARN]") != std::string::npos);
    REQUIRE(output.find("[ERROR]") != std::string::npos);
    // Macros pass __FILE__ and __LINE__: expect basename and line in output
    REQUIRE(output.find("test_logger.cc:") != std::string::npos);
    // Glog-style timestamp: MMDD HH:MM:SS.uuuuuu (contains space, colon, dot for time)
    REQUIRE(output.find(".") != std::string::npos);

    Logger::Instance().set_level(Logger::Level::kDebug);
}

TEST_CASE("Logger: TM_LOG_ASYNC=0 selects sync mode", "[logger]")
{
    // Sync mode: output is written by the calling thread.
    ::setenv("TM_LOG_ASYNC", "0", /*overwrite=*/1);

    std::string output;
    {
        std::thread t([&] {
            // Capture stderr inside the new thread.
            int                saved = ::dup(STDERR_FILENO);
            std::array<int, 2> pipefd{};
            REQUIRE(::pipe(pipefd.data()) == 0);
            REQUIRE(::dup2(pipefd[1], STDERR_FILENO) >= 0);
            ::close(pipefd[1]);

            auto& log = Logger::Instance();
            REQUIRE_FALSE(log.is_async());
            log.Log(Logger::Level::kInfo, "sync-message");
            // Sync mode writes inline.

            REQUIRE(::dup2(saved, STDERR_FILENO) >= 0);
            ::close(saved);

            char    buf[4096];
            ssize_t n = ::read(pipefd[0], buf, sizeof(buf));
            if (n > 0) {
                output.assign(buf, static_cast<size_t>(n));
            }
            ::close(pipefd[0]);
        });
        t.join();
    }

    ::unsetenv("TM_LOG_ASYNC");

    REQUIRE(output.find("sync-message") != std::string::npos);
    REQUIRE(output.find("[INFO]") != std::string::npos);
}

TEST_CASE("Logger: async ordering under concurrent producers", "[logger]")
{
    ::setenv("TM_LOG_ASYNC", "0", 1);  // ensure worker threads use sync mode so capture sees all output
    Logger::Instance().set_level(Logger::Level::kTrace);

    constexpr int kThreads   = 4;
    constexpr int kPerThread = 250;
    constexpr int kTotal     = kThreads * kPerThread;

    auto output = CaptureStderr([&] {
        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([t] {
                for (int i = 0; i < kPerThread; ++i) {
                    TM_LOG_INFO("thread={} i={}", t, i);
                }
            });
        }
        for (auto& th : threads) {
            th.join();
        }
    });

    // Count lines — each message is exactly one line.
    int lines = 0;
    for (char c : output) {
        if (c == '\n') {
            ++lines;
        }
    }

    REQUIRE(lines == kTotal);
}

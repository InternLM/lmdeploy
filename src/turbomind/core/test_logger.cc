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

// ---------------------------------------------------------------------------
// Stderr capture helper
// ---------------------------------------------------------------------------
// Redirects stderr to a pipe, runs `fn`, flushes the async worker, then
// restores stderr and returns everything that was written.
static std::string CaptureStderr(std::function<void()> fn)
{
    // Save real stderr fd.
    int saved = ::dup(STDERR_FILENO);
    REQUIRE(saved >= 0);

    // Create pipe: read end = pipefd[0], write end = pipefd[1].
    std::array<int, 2> pipefd{};
    REQUIRE(::pipe(pipefd.data()) == 0);

    // Point stderr at the write end.
    REQUIRE(::dup2(pipefd[1], STDERR_FILENO) >= 0);
    ::close(pipefd[1]);

    fn();

    // Drain the async worker before we read.
    Logger::Flush();

    // Restore real stderr.
    REQUIRE(::dup2(saved, STDERR_FILENO) >= 0);
    ::close(saved);

    // Read everything from the pipe.
    std::string output;
    char        buf[4096];
    ssize_t     n;
    while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0) {
        output.append(buf, static_cast<size_t>(n));
    }
    ::close(pipefd[0]);

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

    REQUIRE(output.find("[TM][INFO]") != std::string::npos);
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
        TM2_LOG_TRACE("trace-msg");
        TM2_LOG_DEBUG("debug-msg");
        TM2_LOG_INFO("info-msg");
        TM2_LOG_WARNING("warn-msg");
        TM2_LOG_ERROR("error-msg");
    });

    REQUIRE(output.find("[TM][TRACE]") != std::string::npos);
    REQUIRE(output.find("[TM][DEBUG]") != std::string::npos);
    REQUIRE(output.find("[TM][INFO]") != std::string::npos);
    REQUIRE(output.find("[TM][WARNING]") != std::string::npos);
    REQUIRE(output.find("[TM][ERROR]") != std::string::npos);

    Logger::Instance().set_level(Logger::Level::kDebug);
}

TEST_CASE("Logger: TM_LOG_ASYNC=0 selects sync mode", "[logger]")
{
    // Sync mode: output is written by the calling thread, no Flush() needed.
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
            // No Flush() call — sync mode writes inline.

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
    REQUIRE(output.find("[TM][INFO]") != std::string::npos);
}

TEST_CASE("Logger: async ordering under concurrent producers", "[logger]")
{
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
                    TM2_LOG_INFO("thread={} i={}", t, i);
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

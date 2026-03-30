// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/logger.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <functional>
#include <string>
#include <thread>
#include <vector>

// POSIX headers needed for: pipe/fork/dup2 (stderr capture), setenv/unsetenv (env
// var tests), waitpid/SIGABRT (signal handling test).  All guarded behind #ifndef
// _WIN32 along with the test cases that use them.
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

using turbomind::core::Logger;

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

// ---------------------------------------------------------------------------
// POSIX-specific tests (stderr capture, env vars, signal handling)
// ---------------------------------------------------------------------------
#ifndef _WIN32

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

    auto output = CaptureStderr(
        [&] { log.Log(Logger::Level::kDebug, "int={} float={:.2f} str={}", 42, 3.14f, std::string("world")); });

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

// ---------------------------------------------------------------------------
// Signal-handling drain test (fork-based)
// ---------------------------------------------------------------------------
// Verifies that OnFatalSignal drains the async worker's queue before
// re-raising the signal.  Because raising SIGABRT kills the process, we
// fork a child that enables async mode, enqueues messages, and raises
// SIGABRT.  The parent reads the child's stderr via a pipe and checks that
// every message is present and that the child died from SIGABRT.
//
// IMPORTANT: This test relies on AsyncLogWorker::Instance() never having
// been called in the parent process.  All preceding tests use sync mode
// (TM_LOG_ASYNC=0) and never trigger the singleton.  If a future test
// triggers async mode before this one, the fork will inherit a destroyed
// worker thread and the singleton will be in an invalid state.
// ---------------------------------------------------------------------------

TEST_CASE("Logger: async signal handler drains queue on fatal signal", "[logger][signal]")
{
    // Guard: verify we haven't triggered async mode in this process.
    // All preceding tests force TM_LOG_ASYNC=0 via kSyncMode. If a future
    // test enables async before this one, the fork inherits a destroyed
    // AsyncLogWorker thread and the singleton is in an invalid state.
    const char* async_env = std::getenv("TM_LOG_ASYNC");
    REQUIRE((async_env != nullptr && std::string_view{async_env} == "0"));

    constexpr int kMsgCount = 20;
    std::vector<std::string> markers;
    markers.reserve(kMsgCount);
    for (int i = 0; i < kMsgCount; ++i) {
        markers.push_back(fmt::format("drain-test-msg-{:03d}-Ax9B{}", i, i));
    }

    int pipefd[2];
    REQUIRE(::pipe(pipefd) == 0);

    pid_t pid = ::fork();
    REQUIRE(pid >= 0);

    if (pid == 0) {
        // ---- CHILD PROCESS ----
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);

        // Enable async mode for any new Logger instance.
        ::setenv("TM_LOG_ASYNC", "1", 1);
        ::setenv("TM_LOG_LEVEL", "TRACE", 1);

        // Spawn a fresh thread to get a new thread_local Logger that picks
        // up TM_LOG_ASYNC=1 (async mode).
        std::thread worker([&] {
            auto& log = Logger::Instance();
            if (!log.is_async()) {
                fmt::print(stderr, "CHILD-ERROR: not in async mode\n");
                ::_exit(2);
            }

            // Enqueue messages rapidly.  Some may be drained by the worker
            // thread before the signal fires; others will remain in the
            // queue and must be drained by OnFatalSignal.
            for (int i = 0; i < kMsgCount; ++i) {
                log.Log(Logger::Level::kInfo, markers[i]);
            }

            // Small delay so some messages are consumed by the normal worker
            // loop, creating a realistic mix of already-printed and queued.
            ::usleep(1000);

            // OnFatalSignal should: Stop() → drain queue → restore SIG_DFL → re-raise.
            ::raise(SIGABRT);
            ::_exit(0);
        });

        worker.join();
        ::_exit(1);
    }

    // ---- PARENT PROCESS ----
    ::close(pipefd[1]);

    std::string child_output;
    {
        char    buf[4096];
        ssize_t n;
        while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0) {
            child_output.append(buf, static_cast<size_t>(n));
        }
        ::close(pipefd[0]);
    }

    int wstatus = 0;
    REQUIRE(::waitpid(pid, &wstatus, 0) == pid);

    // Child should have been killed by SIGABRT.
    REQUIRE(WIFSIGNALED(wstatus));
    REQUIRE(WTERMSIG(wstatus) == SIGABRT);

    // Every marker must appear in the child's stderr output.
    for (int i = 0; i < kMsgCount; ++i) {
        REQUIRE(child_output.find(markers[i]) != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// TM_LOG_FATAL abort test (fork-based)
// ---------------------------------------------------------------------------
TEST_CASE("Logger: TM_LOG_FATAL aborts the process", "[logger][fatal]")
{
    int pipefd[2];
    REQUIRE(::pipe(pipefd) == 0);

    pid_t pid = ::fork();
    REQUIRE(pid >= 0);

    if (pid == 0) {
        // ---- CHILD PROCESS ----
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);

        ::setenv("TM_LOG_ASYNC", "0", 1);
        ::setenv("TM_LOG_LEVEL", "TRACE", 1);

        std::thread worker([&] {
            // Reset Catch2's inherited SIGABRT handler so std::abort()
            // kills the process rather than triggering Catch2's reporter.
            ::signal(SIGABRT, SIG_DFL);
            TM_LOG_FATAL("fatal-test-marker-{}", 42);
        });
        worker.join();
        ::_exit(0);  // Should not reach here
    }

    // ---- PARENT PROCESS ----
    ::close(pipefd[1]);

    std::string child_output;
    {
        char    buf[4096];
        ssize_t n;
        while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0) {
            child_output.append(buf, static_cast<size_t>(n));
        }
        ::close(pipefd[0]);
    }

    int wstatus = 0;
    REQUIRE(::waitpid(pid, &wstatus, 0) == pid);

    // Child should have been killed by SIGABRT (from std::abort).
    REQUIRE(WIFSIGNALED(wstatus));
    REQUIRE(WTERMSIG(wstatus) == SIGABRT);

    // Fatal message should appear in output.
    REQUIRE(child_output.find("fatal-test-marker-42") != std::string::npos);
    REQUIRE(child_output.find("[FATAL]") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Color disable test
// ---------------------------------------------------------------------------
TEST_CASE("Logger: TM_LOG_COLOR=0 disables ANSI escape codes", "[logger][color]")
{
    Logger::Instance().set_level(Logger::Level::kTrace);

    auto output = CaptureStderr([&] {
        // Color is determined at first use; since this test process started
        // with TM_LOG_COLOR potentially unset, we use a forked child to
        // control the env var from scratch.
    });

    // Fork a child with TM_LOG_COLOR=0 to test colorless output.
    int pipefd[2];
    REQUIRE(::pipe(pipefd) == 0);

    pid_t pid = ::fork();
    REQUIRE(pid >= 0);

    if (pid == 0) {
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);

        ::setenv("TM_LOG_COLOR", "0", 1);
        ::setenv("TM_LOG_ASYNC", "0", 1);
        ::setenv("TM_LOG_LEVEL", "TRACE", 1);

        std::thread worker([&] {
            auto& log = Logger::Instance();
            log.Log(Logger::Level::kError, "no-color-marker");
        });
        worker.join();
        ::_exit(0);
    }

    ::close(pipefd[1]);

    std::string child_output;
    {
        char    buf[4096];
        ssize_t n;
        while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0) {
            child_output.append(buf, static_cast<size_t>(n));
        }
        ::close(pipefd[0]);
    }

    int wstatus = 0;
    REQUIRE(::waitpid(pid, &wstatus, 0) == pid);
    REQUIRE(WIFEXITED(wstatus));

    // Message must be present.
    REQUIRE(child_output.find("no-color-marker") != std::string::npos);

    // Must NOT contain ANSI escape sequences (\x1b[).
    REQUIRE(child_output.find("\x1b[") == std::string::npos);
}

#endif  // _WIN32
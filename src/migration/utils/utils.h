#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

#if defined(__x86_64__)
#include <immintrin.h>
#define PAUSE() _mm_pause()
#else
#define PAUSE()
#endif

class RWSpinlock {
    union RWTicket {
        constexpr RWTicket(): whole(0) {}
        uint64_t whole;
        uint32_t readWrite;
        struct {
            uint16_t write;
            uint16_t read;
            uint16_t users;
        };
    } ticket;

private:
    static void asm_volatile_memory()
    {
        asm volatile("" ::: "memory");
    }

    template<class T>
    static T load_acquire(T* addr)
    {
        T t = *addr;
        asm_volatile_memory();
        return t;
    }

    template<class T>
    static void store_release(T* addr, T v)
    {
        asm_volatile_memory();
        *addr = v;
    }

public:
    RWSpinlock() {}

    RWSpinlock(RWSpinlock const&)            = delete;
    RWSpinlock& operator=(RWSpinlock const&) = delete;

    void lock()
    {
        writeLockNice();
    }

    bool tryLock()
    {
        RWTicket t;
        uint64_t old = t.whole = load_acquire(&ticket.whole);
        if (t.users != t.write)
            return false;
        ++t.users;
        return __sync_bool_compare_and_swap(&ticket.whole, old, t.whole);
    }

    void writeLockAggressive()
    {
        uint32_t count = 0;
        uint16_t val   = __sync_fetch_and_add(&ticket.users, 1);
        while (val != load_acquire(&ticket.write)) {
            PAUSE();
            if (++count > 1000)
                std::this_thread::yield();
        }
    }

    void writeLockNice()
    {
        uint32_t count = 0;
        while (!tryLock()) {
            PAUSE();
            if (++count > 1000)
                std::this_thread::yield();
        }
    }

    void unlockAndLockShared()
    {
        uint16_t val = __sync_fetch_and_add(&ticket.read, 1);
        (void)val;
    }

    void unlock()
    {
        RWTicket t;
        t.whole = load_acquire(&ticket.whole);
        ++t.read;
        ++t.write;
        store_release(&ticket.readWrite, t.readWrite);
    }

    void lockShared()
    {
        uint_fast32_t count = 0;
        while (!tryLockShared()) {
            PAUSE();
            if (++count > 1000)
                std::this_thread::yield();
        }
    }

    bool tryLockShared()
    {
        RWTicket t, old;
        old.whole = t.whole = load_acquire(&ticket.whole);
        old.users           = old.read;
        ++t.read;
        ++t.users;
        return __sync_bool_compare_and_swap(&ticket.whole, old.whole, t.whole);
    }

    void unlockShared()
    {
        __sync_fetch_and_add(&ticket.write, 1);
    }

public:
    struct WriteGuard {
        WriteGuard(RWSpinlock& lock): lock(lock)
        {
            lock.lock();
        }

        WriteGuard(const WriteGuard&) = delete;

        WriteGuard& operator=(const WriteGuard&) = delete;

        ~WriteGuard()
        {
            lock.unlock();
        }

        RWSpinlock& lock;
    };

    struct ReadGuard {
        ReadGuard(RWSpinlock& lock): lock(lock)
        {
            lock.lockShared();
        }

        ReadGuard(const ReadGuard&) = delete;

        ReadGuard& operator=(const ReadGuard&) = delete;

        ~ReadGuard()
        {
            lock.unlockShared();
        }

        RWSpinlock& lock;
    };

private:
    const static int64_t kExclusiveLock = INT64_MIN / 2;

    std::atomic<int64_t> lock_;
    uint64_t             padding_[15];
};
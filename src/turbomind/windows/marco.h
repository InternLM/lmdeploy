#pragma once

#if !defined(__PRETTY_FUNCTION__) && !defined(__GNUC__)

#define __PRETTY_FUNCTION__ __FUNCSIG__

namespace turbomind {
typedef struct timeval {
    long tv_sec;
    long tv_usec;
} TIMEVAL, *PTIMEVAL, *LPTIMEVAL;

}  // namespace turbomind

#endif

typedef unsigned int uint;

#pragma once

#if !defined(__PRETTY_FUNCTION__) && !defined(__GNUC__)
#define __PRETTY_FUNCTION__ __FUNCSIG__
typedef struct timeval {
    long tv_sec;
    long tv_usec;
} TIMEVAL, *PTIMEVAL, *LPTIMEVAL;
#endif

typedef unsigned int uint;

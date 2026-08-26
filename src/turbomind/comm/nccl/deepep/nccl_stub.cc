// Copyright (c) OpenMMLab. All rights reserved.

#include <nccl.h>
#include <nccl_device/core.h>

#include <cstdio>
#include <cstdlib>

#define TM_NCCL_STUB_EXPORT __attribute__((visibility("default")))
#define TM_NCCL_STUB_FALLBACK TM_NCCL_STUB_EXPORT __attribute__((weak))

namespace {

[[noreturn]] void ReportMissingNcclSymbol(const char* symbol)
{
    std::fprintf(stderr,
                 "TurboMind fatal error: NCCL compatibility stub invoked for `%s`.\n"
                 "TurboMind was built with NCCL %d.%d.%d (NCCL_VERSION_CODE=%d), and DeepEP requires NCCL >= "
                 "2.30.4. The runtime NCCL library does not provide this symbol, or the NCCL libraries were loaded "
                 "in an unsupported order. Aborting.\n",
                 symbol,
                 NCCL_MAJOR,
                 NCCL_MINOR,
                 NCCL_PATCH,
                 NCCL_VERSION_CODE);
    std::fflush(stderr);
    std::abort();
}

}  // namespace

extern "C" {

TM_NCCL_STUB_EXPORT void turbomind_nccl_stub_anchor() {}

TM_NCCL_STUB_FALLBACK ncclResult_t ncclCommQueryProperties(ncclComm_t, ncclCommProperties_t*)
{
    ReportMissingNcclSymbol("ncclCommQueryProperties");
}

TM_NCCL_STUB_FALLBACK ncclResult_t ncclCommWindowRegister(ncclComm_t, void*, size_t, ncclWindow_t*, int)
{
    ReportMissingNcclSymbol("ncclCommWindowRegister");
}

TM_NCCL_STUB_FALLBACK ncclResult_t ncclCommWindowDeregister(ncclComm_t, ncclWindow_t)
{
    ReportMissingNcclSymbol("ncclCommWindowDeregister");
}

TM_NCCL_STUB_FALLBACK ncclResult_t ncclDevCommCreate(ncclComm_t, const ncclDevCommRequirements_t*, ncclDevComm_t*)
{
    ReportMissingNcclSymbol("ncclDevCommCreate");
}

TM_NCCL_STUB_FALLBACK ncclResult_t ncclDevCommDestroy(ncclComm_t, const ncclDevComm_t*)
{
    ReportMissingNcclSymbol("ncclDevCommDestroy");
}

TM_NCCL_STUB_FALLBACK ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t, size_t, int, void**)
{
    ReportMissingNcclSymbol("ncclGetLsaDevicePointer");
}

TM_NCCL_STUB_FALLBACK ncclTeam_t ncclTeamLsa(ncclComm_t)
{
    ReportMissingNcclSymbol("ncclTeamLsa");
}

}  // extern "C"

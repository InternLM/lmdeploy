// Copyright (c) OpenMMLab. All rights reserved.
//
// Host-side architecture dispatcher for the verification-attention path.
// Selects the SM90 (WGMMA / cp.async) or SM75 (mma.sync, sync copies)
// implementation by device arch at runtime. Contains NO device code.
//
// The arch libraries are optional (guarded by TM_BUILD_VERIFICATION_ATTENTION_*),
// so configurations without one simply fall through to the other / unsupported.
#include "src/turbomind/kernels/attention/verification/attention.h"

#ifndef TM_BUILD_VERIFICATION_ATTENTION_SM90
#define TM_BUILD_VERIFICATION_ATTENTION_SM90 0
#endif
#ifndef TM_BUILD_VERIFICATION_ATTENTION_SM75
#define TM_BUILD_VERIFICATION_ATTENTION_SM75 0
#endif

namespace turbomind::verification_attention {

#if TM_BUILD_VERIFICATION_ATTENTION_SM90
namespace sm90 {
bool supports(const Capability&);
int  choose_split_count(int, int, int, int, int, int, int);
void run(const Arguments&);
}  // namespace sm90
#endif

#if TM_BUILD_VERIFICATION_ATTENTION_SM75
namespace sm75 {
bool supports(const Capability&);
int  choose_split_count(int, int, int, int, int, int, int);
void run(const Arguments&);
}  // namespace sm75
#endif

bool supports(const Capability& c)
{
#if TM_BUILD_VERIFICATION_ATTENTION_SM90
    if (c.arch >= 90) {
        return sm90::supports(c);
    }
#endif
#if TM_BUILD_VERIFICATION_ATTENTION_SM75
    if (c.arch >= 75 && c.arch < 80) {
        return sm75::supports(c);
    }
#endif
    return false;
}

int choose_split_count(int query_count,
                       int base_cta_count,
                       int max_key_length,
                       int key_tile,
                       int partial_capacity,
                       int requested_max_splits,
                       int sm_count)
{
    // Arch-neutral arithmetic.
#if TM_BUILD_VERIFICATION_ATTENTION_SM90
    return sm90::choose_split_count(
        query_count, base_cta_count, max_key_length, key_tile, partial_capacity, requested_max_splits, sm_count);
#elif TM_BUILD_VERIFICATION_ATTENTION_SM75
    return sm75::choose_split_count(
        query_count, base_cta_count, max_key_length, key_tile, partial_capacity, requested_max_splits, sm_count);
#else
    return 1;
#endif
}

void run(const Arguments& arguments)
{
#if TM_BUILD_VERIFICATION_ATTENTION_SM90
    if (arguments.arch >= 90) {
        sm90::run(arguments);
        return;
    }
#endif
#if TM_BUILD_VERIFICATION_ATTENTION_SM75
    sm75::run(arguments);
#endif
}

}  // namespace turbomind::verification_attention

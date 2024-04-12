// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

// Operand A
// shape: [M, K]
// outer stride     : K
// outer tile stride: CTA_M * K
// inner tile stride: CTA_K

// Operand B
// shape: [N, K]
// outer stride     : K
// outer tile stride: CTA_N * K
// inner tile stride: CTA_K

// Operand B
// shape: [N/(P_N*OP_N), K/(P_K*OP_K), P_K*P_N*OP_N*OP_K]
// outer stride     : K*P_N*OP_N
// outer tile stride: CTA_N/(P_N*OP_N)*K*P_N*OP_N -> CTA_N*K
// inner tile stride: CTA_K*P_N*OP_N

}  // namespace turbomind
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class MoeGateCase:
    tokens: int
    experts: int
    top_k: int

    @property
    def name(self) -> str:
        return f't{self.tokens}_e{self.experts}_k{self.top_k}'


SMOKE_CASES: tuple[MoeGateCase, ...] = (
    MoeGateCase(tokens=16, experts=8, top_k=2),
    MoeGateCase(tokens=32, experts=64, top_k=4),
    MoeGateCase(tokens=16, experts=160, top_k=6),
    MoeGateCase(tokens=7, experts=8, top_k=2),
    MoeGateCase(tokens=16, experts=256, top_k=8),
    MoeGateCase(tokens=16, experts=2560, top_k=8),
)

BENCH_TOKEN_COUNTS: tuple[int, ...] = (1, 4, 16, 64, 256, 1024, 2048, 4096, 8192, 16384)
BENCH_EXPERT_TOPK: tuple[tuple[int, int], ...] = (
    (8, 2),
    (64, 4),
    (160, 6),
    (256, 8),
    (512, 8),
    (2560, 8),
)


def bench_cases() -> tuple[MoeGateCase, ...]:
    return tuple(
        MoeGateCase(tokens=tokens, experts=experts, top_k=top_k)
        for tokens in BENCH_TOKEN_COUNTS
        for experts, top_k in BENCH_EXPERT_TOPK
    )

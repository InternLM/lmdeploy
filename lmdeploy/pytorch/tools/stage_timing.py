"""Stage timing utilities for LMDeploy.

Used to accumulate per-stage latency in long-running services and periodically
print a small table.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from typing import Dict, Optional


def _get_env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name, None)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def stage_timing_enabled() -> bool:
    return _get_env_bool("LMDEPLOY_STAGE_TIMING", default=False)


def stage_timing_sync_enabled() -> bool:
    """Whether to synchronize around timed regions for accuracy."""
    return _get_env_bool("LMDEPLOY_STAGE_TIMING_SYNC", default=True)


_LOCK = threading.Lock()
_STATS: Dict[str, Dict[str, float]] = {}
_FORWARD_COUNT: int = 0
_ATEXIT_REGISTERED = False


def _get_rank() -> int:
    try:
        import torch.distributed as dist  # type: ignore

        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        return 0


def _print_table(snapshot: Dict[str, Dict[str, float]], *, report_every: int, reset_each: bool) -> None:
    rank = _get_rank()

    def _stats(stage: str) -> tuple[float, int, float]:
        d = snapshot.get(stage, {"total_time_s": 0.0, "count": 0.0})
        total = float(d.get("total_time_s", 0.0))
        cnt_i = int(float(d.get("count", 0.0)))
        avg_ms = (total / cnt_i * 1000.0) if cnt_i > 0 else 0.0
        return total, cnt_i, avg_ms

    vit_total, vit_cnt, vit_avg = _stats("vit")
    pre_total, pre_cnt, pre_avg = _stats("lm_prefill")
    dec_total, dec_cnt, dec_avg = _stats("lm_decode")

    # Single-line log for easy grep/parse.
    window = "reset" if reset_each else "cumulative"
    print(
        "[LMDEPLOY_STAGE_TIMING]"
        f" rank={rank}"
        f" report_every={report_every}"
        f" window={window}"
        f" | vit_total_s={vit_total:.6f} vit_count={vit_cnt} vit_avg_ms={vit_avg:.3f}"
        f" | lm_prefill_total_s={pre_total:.6f} lm_prefill_count={pre_cnt} lm_prefill_avg_ms={pre_avg:.3f}"
        f" | lm_decode_total_s={dec_total:.6f} lm_decode_count={dec_cnt} lm_decode_avg_ms={dec_avg:.3f}"
        ,
        flush=True,
    )


def _register_atexit() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    _ATEXIT_REGISTERED = True

    def _flush() -> None:
        if not stage_timing_enabled():
            return
        report_every = _get_env_int("LMDEPLOY_STAGE_TIMING_MAX_COUNT", 1000)
        reset_each = _get_env_bool("LMDEPLOY_STAGE_TIMING_RESET", default=False)
        print_all_ranks = _get_env_bool("LMDEPLOY_STAGE_TIMING_PRINT_ALL_RANKS", default=False)
        rank = _get_rank()
        if (not print_all_ranks) and rank != 0:
            return
        with _LOCK:
            if not _STATS:
                return
            snapshot = {k: dict(v) for k, v in _STATS.items()}
        _print_table(snapshot, report_every=report_every, reset_each=reset_each)

    atexit.register(_flush)


def record_stage(stage: str, elapsed_s: float) -> None:
    """Accumulate time and count for a stage."""
    if not stage_timing_enabled():
        return
    with _LOCK:
        st = _STATS.get(stage)
        if st is None:
            st = {"total_time_s": 0.0, "count": 0.0}
            _STATS[stage] = st
        st["total_time_s"] += float(elapsed_s)
        st["count"] += 1.0


def bump_forward_and_maybe_report(*, enabled: bool) -> None:
    """Increment forward counter and report timing table every N forwards."""
    if not enabled:
        return
    _register_atexit()

    report_every = _get_env_int("LMDEPLOY_STAGE_TIMING_MAX_COUNT", 1000)
    if report_every <= 0:
        report_every = 1000
    # Default to cumulative stats across the whole process.
    # Set LMDEPLOY_STAGE_TIMING_RESET=1 to reset after each report (windowed mode).
    reset_each = _get_env_bool("LMDEPLOY_STAGE_TIMING_RESET", default=False)
    print_all_ranks = _get_env_bool("LMDEPLOY_STAGE_TIMING_PRINT_ALL_RANKS", default=False)
    rank = _get_rank()

    with _LOCK:
        global _FORWARD_COUNT
        _FORWARD_COUNT += 1
        if (_FORWARD_COUNT % report_every) != 0:
            return
        snapshot = {k: dict(v) for k, v in _STATS.items()}
        if reset_each:
            _STATS.clear()

    if (not print_all_ranks) and rank != 0:
        return
    _print_table(snapshot, report_every=report_every, reset_each=reset_each)


class NPUTimer:
    """Wall-time timer for NPU regions. Synchronizes for accuracy if enabled."""

    def __init__(self, device=None, *, sync: Optional[bool] = None, enabled: bool = True):
        self.enabled = enabled and stage_timing_enabled()
        self.sync = stage_timing_sync_enabled() if sync is None else bool(sync)
        self._t0 = 0.0
        self.elapsed_s = 0.0
        self._dev_index = None
        self._torch_npu = None

        if not self.enabled:
            return
        try:
            import torch_npu  # type: ignore

            self._torch_npu = torch_npu
            # infer device index best-effort
            try:
                if device is not None and getattr(device, "type", None) == "npu":
                    self._dev_index = device.index
            except Exception:
                self._dev_index = None
            if self._dev_index is None:
                self._dev_index = torch_npu.npu.current_device()
        except Exception:
            self.enabled = False
            self._torch_npu = None

    def __enter__(self):
        if self.enabled and self.sync and self._torch_npu is not None:
            self._torch_npu.npu.synchronize(self._dev_index)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.sync and self._torch_npu is not None:
            self._torch_npu.npu.synchronize(self._dev_index)
        self.elapsed_s = time.perf_counter() - self._t0
        return False



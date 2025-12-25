import time
import pickle
from pathlib import Path
from contextlib import contextmanager
from typing import List

import torch
import torch.distributed as dist
import torch_npu


MEMORY_SNAPSHOT_MAX_ENTRIES = 30000000


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_npu_devices() -> List[int]:
    torch_npu.npu._lazy_init()
    return list(range(torch_npu.npu.device_count()))


class NPUMemoryProfiler:
    """
    Minimal / safe profiler:
      - always dumps ONE full snapshot (directly compatible with memory_viz)
      - dumps per-device memory_summary for quick inspection
      - no per-device snapshot splitting (no ambiguity, no misleading labels)
    """

    def __init__(
        self,
        profile_dir: Path,
        *,
        enabled: str = "all",
        context: str = "all",
        stacks: str = "all",
        max_entries: int = MEMORY_SNAPSHOT_MAX_ENTRIES,
        dump_summary: bool = True,
    ):
        self.profile_dir = Path(profile_dir)
        self.dump_summary = dump_summary

        torch_npu.npu._lazy_init()

        torch_npu.npu.memory._record_memory_history(
            enabled=enabled,
            context=context,
            stacks=stacks,
            max_entries=max_entries,
        )

    def _dump_summary(self, output_dir: Path, rank: int, tag: str) -> None:
        for device in _get_npu_devices():
            summary = torch_npu.npu.memory_summary(device=device)
            out = output_dir / f"rank{rank}_device{device}_memory_summary_{tag}.txt"
            with open(out, "w") as f:
                f.write(summary)

    def step(self, *, exit_ctx: bool = False, tag: str = "step") -> Path:
        rank = _get_rank()

        if exit_ctx:
            output_dir = self.profile_dir.with_name(self.profile_dir.name + "_exit")
        else:
            output_dir = self.profile_dir
        output_dir.mkdir(exist_ok=True, parents=True)

        begin = time.monotonic()
        out_file = output_dir / f"rank{rank}_npu_memory_snapshot_full_{tag}.pickle"
        torch_npu.npu.memory._dump_snapshot(out_file)

        print(
            f"[NPUMemoryProfiler] rank={rank} dumped full snapshot "
            f"in {time.monotonic() - begin:.2f}s -> {out_file}"
        )
        return out_file


@contextmanager
def profiling_npu_memory(profile_dir: Path, **profiler_kwargs):
    """
    Normal exit  -> dump to profile_dir (tag='exit')
    OOM exit     -> dump to profile_dir_exit (tag='oom')
    """
    profiler = NPUMemoryProfiler(profile_dir, **profiler_kwargs)
    try:
        yield profiler
    except torch.OutOfMemoryError:
        profiler.step(exit_ctx=True, tag="oom")
        raise
    else:
        profiler.step(exit_ctx=False, tag="exit")
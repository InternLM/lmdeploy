#!/usr/bin/env python3
"""Resumable multi-GPU SM90 kernel-usage scan.

The full benchmark run in one process accumulates GPU memory across the large/MoE
model cases and eventually OOMs on the testbed. This driver packs concrete
``(case, TP, EP)`` fixtures into cost-balanced tasks, runs every task in a fresh
Python process, and distributes the tasks across one worker per GPU.

Each successful task has its own atomic log and completion record. An interrupted
scan can be continued with ``--resume`` when its configuration and loaded
``_turbomind`` binary still match. Completed task logs are merged into ``bench.log``
in deterministic order, so downstream aggregation remains unchanged:

    python -m tests.turbomind.linear.kernel_usage_report tmp/sm90_scan/bench.log

Use --type to select the TypeSpec to scan, e.g. bf16_bf16_bf16 (SM90 BF16 kernels),
bf16_e4m3b128_bf16 (FP8 weight-as-A) or e4m3k128_e4m3b128_bf16 (FP8 v3 act-as-A).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lmdeploy.turbomind import _tm

from .cases import LinearCase, expand_suite

_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_VERSION = 2


@dataclass(frozen=True)
class ScanTask:
    index: int
    fixtures: tuple[LinearCase, ...]
    predicted_work: int

    @property
    def task_id(self) -> str:
        return f'task-{self.index:04d}'

    @property
    def case_names(self) -> tuple[str, ...]:
        return tuple(fixture.name for fixture in self.fixtures)

    @property
    def tp(self) -> int:
        return self.fixtures[0].tp

    @property
    def ep(self) -> int:
        return self.fixtures[0].ep

    @property
    def weight(self) -> int:
        return len(self.fixtures)


@dataclass(frozen=True)
class ScanConfig:
    type_name: str
    outdir: Path
    python: str
    batches: tuple[int, ...]
    env: dict[str, str]
    manifest_signature: str


@dataclass
class ScanProgress:
    total_tasks: int
    total_fixtures: int
    initial_fixtures: int
    completed_tasks: int
    completed_fixtures: int
    started_at: float


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(',') if x.strip())


def parse_name_list(value: str) -> tuple[str, ...] | None:
    names = tuple(x.strip() for x in value.split(',') if x.strip())
    return names or None


def collect_fixtures(runs) -> list[LinearCase]:
    fixtures: dict[tuple[str, int, int], LinearCase] = {}
    for run in runs:
        case = run.case
        fixtures[(case.name, case.tp, case.ep)] = case
    return [fixtures[key] for key in sorted(fixtures)]


def estimate_fixture_work(fixture: LinearCase, batches: tuple[int, ...]) -> int:
    routed = fixture.experts_per_token if fixture.expert_num else 1
    bounded_tokens = sum(min(batch * routed, 4096) for batch in batches)
    weight_matrices = max(fixture.expert_num, 1)
    return fixture.input_dim * fixture.output_dim * (bounded_tokens + weight_matrices)


def balance_fixtures(
    fixtures: list[LinearCase],
    max_fixtures: int,
    batches: tuple[int, ...],
) -> list[ScanTask]:
    if max_fixtures <= 0:
        raise ValueError('chunk_must_be_positive')
    groups: dict[tuple[int, int], list[LinearCase]] = {}
    for fixture in fixtures:
        groups.setdefault((fixture.tp, fixture.ep), []).append(fixture)

    packed: list[tuple[tuple[LinearCase, ...], int]] = []
    for _, group in sorted(groups.items()):
        task_count = math.ceil(len(group) / max_fixtures)
        bins: list[list[LinearCase]] = [[] for _ in range(task_count)]
        weights = [0] * task_count
        ordered = sorted(group, key=lambda item: (-estimate_fixture_work(item, batches), item.name))
        for fixture in ordered:
            candidates = [i for i, task in enumerate(bins) if len(task) < max_fixtures]
            index = min(candidates, key=lambda i: (weights[i], len(bins[i]), i))
            bins[index].append(fixture)
            weights[index] += estimate_fixture_work(fixture, batches)
        packed.extend((tuple(task), weight) for task, weight in zip(bins, weights) if task)

    packed.sort(
        key=lambda item: (
            -item[1],
            item[0][0].tp,
            item[0][0].ep,
            tuple(fixture.name for fixture in item[0]),
        )
    )
    return [ScanTask(index, task, weight) for index, (task, weight) in enumerate(packed)]


def resolve_python(command: str) -> str:
    resolved = shutil.which(command)
    return str(Path(resolved or command).resolve())


def extension_fingerprint() -> dict[str, str | int]:
    path = Path(_tm.__file__).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def task_manifest(task: ScanTask) -> dict[str, Any]:
    return {
        'id': task.task_id,
        'weight': task.weight,
        'predicted_work': task.predicted_work,
        'tp': task.tp,
        'ep': task.ep,
        'cases': list(task.case_names),
        'fixtures': [asdict(fixture) for fixture in task.fixtures],
    }


def make_manifest(
    *,
    type_name: str,
    batches: tuple[int, ...],
    tps: tuple[int, ...],
    eps: tuple[int, ...],
    tune_params: str,
    python: str,
    tasks: list[ScanTask],
) -> dict[str, Any]:
    identity = {
        'version': _MANIFEST_VERSION,
        'type': type_name,
        'batches': list(batches),
        'tp': list(tps),
        'ep': list(eps),
        'tune_params': tune_params,
        'python': resolve_python(python),
        'extension': extension_fingerprint(),
        'tasks': [task_manifest(task) for task in tasks],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
    return {'signature': hashlib.sha256(encoded).hexdigest(), **identity}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def prepare_outdir(outdir: Path, manifest: dict[str, Any], resume: bool) -> None:
    manifest_path = outdir / 'scan.json'
    if resume:
        if not manifest_path.is_file():
            raise RuntimeError(f'resume_manifest_not_found_{manifest_path}')
        saved = json.loads(manifest_path.read_text())
        if saved != manifest:
            raise RuntimeError(f'resume_manifest_mismatch_{manifest_path}')
    else:
        if outdir.exists() and any(outdir.iterdir()):
            raise RuntimeError(f'output_directory_not_empty_{outdir}')
        outdir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest_path, manifest)
    (outdir / 'tasks').mkdir(exist_ok=True)


def record_path(outdir: Path, fixture: LinearCase) -> Path:
    return outdir / f'records.{fixture.name}__tp{fixture.tp}__ep{fixture.ep}'


def task_paths(outdir: Path, task: ScanTask) -> tuple[Path, Path, Path, Path]:
    root = outdir / 'tasks' / task.task_id
    return (
        root.with_suffix('.log'),
        root.with_suffix('.log.tmp'),
        root.with_suffix('.failed.log'),
        root.with_suffix('.done.json'),
    )


def task_is_complete(outdir: Path, task: ScanTask, signature: str) -> bool:
    log, _, _, done = task_paths(outdir, task)
    if not log.is_file() or log.stat().st_size == 0 or not done.is_file():
        return False
    try:
        status = json.loads(done.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(status, dict):
        return False
    if status.get('signature') != signature or status.get('task') != task.task_id:
        return False
    return all(
        (path := record_path(outdir, fixture)).is_file() and path.stat().st_size > 0
        for fixture in task.fixtures
    )


def build_command(task: ScanTask, config: ScanConfig) -> list[str]:
    return [
        config.python,
        '-m',
        'tests.turbomind.linear.bench_linear',
        '--suite',
        'full',
        '--case',
        ','.join(task.case_names),
        '--type',
        config.type_name,
        '--batch',
        ','.join(map(str, config.batches)),
        '--tp',
        str(task.tp),
        '--ep',
        str(task.ep),
        '--exact-parallel',
        '--tune',
        '--iters',
        '0',
        '--no-validate',
        '--quiet',
        '--export',
        str(config.outdir / 'records'),
    ]


def run_task(
    task: ScanTask,
    gpu: str,
    config: ScanConfig,
    active: dict[str, subprocess.Popen[bytes]],
    active_lock: threading.Lock,
) -> tuple[bool, float, str | None]:
    log, tmp_log, failed_log, done = task_paths(config.outdir, task)
    tmp_log.unlink(missing_ok=True)
    done.unlink(missing_ok=True)
    command = build_command(task, config)
    env = config.env.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    started = time.monotonic()

    try:
        with tmp_log.open('wb') as output:
            output.write(f"# TASK {task.task_id}: {','.join(task.case_names)}\n".encode())
            output.flush()
            proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env, cwd=str(_ROOT))
            with active_lock:
                active[task.task_id] = proc
            try:
                returncode = proc.wait()
            finally:
                with active_lock:
                    active.pop(task.task_id, None)
    except Exception as error:
        if tmp_log.exists():
            os.replace(tmp_log, failed_log)
        duration = time.monotonic() - started
        return False, duration, f'{type(error).__name__}: {error}'

    duration = time.monotonic() - started
    missing = []
    for fixture in task.fixtures:
        path = record_path(config.outdir, fixture)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
    if returncode != 0 or missing:
        os.replace(tmp_log, failed_log)
        reason = f'exit={returncode}' if returncode != 0 else f'missing_records={len(missing)}'
        return False, duration, reason

    os.replace(tmp_log, log)
    failed_log.unlink(missing_ok=True)
    atomic_write_json(
        done,
        {
            'duration_seconds': duration,
            'gpu': gpu,
            'signature': config.manifest_signature,
            'task': task.task_id,
        },
    )
    return True, duration, None


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f'{hours:d}:{minute:02d}:{second:02d}'
    return f'{minute:d}:{second:02d}'


def progress_line(progress: ScanProgress) -> str:
    elapsed = time.monotonic() - progress.started_at
    new_fixtures = progress.completed_fixtures - progress.initial_fixtures
    remaining = progress.total_fixtures - progress.completed_fixtures
    eta = remaining * elapsed / new_fixtures if new_fixtures else None
    percent = 100 * progress.completed_fixtures / progress.total_fixtures
    eta_text = '?' if eta is None else format_duration(eta)
    return (
        f'{progress.completed_tasks}/{progress.total_tasks} tasks, '
        f'{progress.completed_fixtures}/{progress.total_fixtures} fixtures '
        f'({percent:.1f}%), elapsed {format_duration(elapsed)}, ETA {eta_text}'
    )


def worker(
    gpu: str,
    tasks: queue.Queue[ScanTask],
    config: ScanConfig,
    failures: list[tuple[ScanTask, str]],
    progress: ScanProgress,
    lock: threading.Lock,
    stop: threading.Event,
    active: dict[str, subprocess.Popen[bytes]],
    active_lock: threading.Lock,
) -> None:
    while not stop.is_set():
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        names = ', '.join(task.case_names)
        with lock:
            print(
                f'[gpu{gpu}] start {task.task_id} ({task.weight} fixtures, tp={task.tp}, ep={task.ep}): {names}',
                flush=True,
            )
        ok, duration, reason = run_task(task, gpu, config, active, active_lock)
        with lock:
            if ok:
                progress.completed_tasks += 1
                progress.completed_fixtures += task.weight
                print(
                    f'[gpu{gpu}] done {task.task_id} in {format_duration(duration)} | {progress_line(progress)}',
                    flush=True,
                )
            else:
                failures.append((task, reason or 'unknown_failure'))
                print(f'[gpu{gpu}] failed {task.task_id} ({reason}) after {format_duration(duration)}', flush=True)
        tasks.task_done()


def terminate_active(active: dict[str, subprocess.Popen[bytes]], lock: threading.Lock) -> None:
    with lock:
        processes = list(active.values())
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + 10
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()


def merge_logs(outdir: Path, tasks: list[ScanTask], signature: str) -> tuple[Path, int]:
    merged = outdir / 'bench.log'
    tmp = merged.with_suffix('.log.tmp')
    count = 0
    with tmp.open('wb') as output:
        for task in tasks:
            if not task_is_complete(outdir, task, signature):
                continue
            log, _, _, _ = task_paths(outdir, task)
            with log.open('rb') as source:
                shutil.copyfileobj(source, output)
            count += 1
    os.replace(tmp, merged)
    return merged, count


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir', type=Path, default=Path('tmp/sm90_scan'))
    p.add_argument('--type', type=str, default='bf16_bf16_bf16', help='TypeSpec name to scan')
    p.add_argument('--case', type=str, default='', help='optional comma-separated LinearCase names')
    p.add_argument('--chunk', type=int, default=1, help='maximum same-TP/EP fixtures per subprocess')
    p.add_argument('--gpus', type=str, default='0', help='comma-separated CUDA_VISIBLE_DEVICES values')
    p.add_argument('--batch', type=str, default='', help='comma-separated batch sizes; default: full-suite batches')
    p.add_argument('--tp', type=str, default='1,2,4,8', help='comma-separated TP sizes')
    p.add_argument('--ep', type=str, default='1,2,4,8', help='comma-separated EP sizes')
    p.add_argument('--python', type=str, default=sys.executable)
    p.add_argument('--resume', action='store_true', help='resume an exactly matching scan manifest')
    p.add_argument('--dry-run', action='store_true', help='print the balanced task plan without creating output')
    p.add_argument(
        '--tune-params',
        type=str,
        default='swizzle=[0,1,2,3],clusters=0,min_iter=3,max_iter=10,max_time=10',
        help='TM_GEMM_TUNE value forwarded to every benchmark subprocess',
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    case_names = parse_name_list(args.case)
    batches = parse_int_list(args.batch) if args.batch.strip() else None
    tps = parse_int_list(args.tp)
    eps = parse_int_list(args.ep)
    gpus = [gpu.strip() for gpu in args.gpus.split(',') if gpu.strip()]
    if args.chunk <= 0:
        raise ValueError('chunk_must_be_positive')
    if not gpus:
        raise ValueError('gpus_must_be_non_empty')
    if len(gpus) != len(set(gpus)):
        raise ValueError('gpus_must_be_unique')

    runs = expand_suite('full', case_names, batches, (args.type,), tps=tps, eps=eps)
    mismatched = sorted({run.case.type_name for run in runs if run.case.type_name != args.type})
    if mismatched:
        raise ValueError(f'case_type_mismatch_expected_{args.type}_got_{mismatched}')
    first_fixture = (runs[0].case.name, runs[0].case.tp, runs[0].case.ep)
    resolved_batches = tuple(
        run.batch_size
        for run in runs
        if (run.case.name, run.case.tp, run.case.ep) == first_fixture
    )
    fixtures = collect_fixtures(runs)
    tasks = balance_fixtures(fixtures, args.chunk, resolved_batches)
    case_count = len({fixture.name for fixture in fixtures})
    total_fixtures = sum(task.weight for task in tasks)
    print(
        f'Total {args.type}: {case_count} case names, {total_fixtures} fixtures, '
        f'{len(runs)} runs, {len(tasks)} tasks',
        file=sys.stderr,
    )

    if args.dry_run:
        for task in tasks:
            print(
                f'{task.task_id}: fixtures={task.weight} tp={task.tp} ep={task.ep} '
                f'work={task.predicted_work} names={",".join(task.case_names)}'
            )
        return 0

    manifest = make_manifest(
        type_name=args.type,
        batches=resolved_batches,
        tps=tps,
        eps=eps,
        tune_params=args.tune_params,
        python=args.python,
        tasks=tasks,
    )
    prepare_outdir(args.outdir, manifest, args.resume)
    signature = manifest['signature']
    assert isinstance(signature, str)

    completed = [task for task in tasks if task_is_complete(args.outdir, task, signature)]
    pending = [task for task in tasks if task not in completed]
    completed_fixtures = sum(task.weight for task in completed)
    if completed:
        print(f'Resuming with {len(completed)}/{len(tasks)} tasks already complete.', file=sys.stderr)

    base_env = os.environ.copy()
    base_env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    base_env['TM_GEMM_TUNE'] = args.tune_params
    config = ScanConfig(
        type_name=args.type,
        outdir=args.outdir,
        python=args.python,
        batches=resolved_batches,
        env=base_env,
        manifest_signature=signature,
    )

    task_queue: queue.Queue[ScanTask] = queue.Queue()
    for task in pending:
        task_queue.put(task)
    failures: list[tuple[ScanTask, str]] = []
    output_lock = threading.Lock()
    active_lock = threading.Lock()
    active: dict[str, subprocess.Popen[bytes]] = {}
    stop = threading.Event()
    progress = ScanProgress(
        total_tasks=len(tasks),
        total_fixtures=total_fixtures,
        initial_fixtures=completed_fixtures,
        completed_tasks=len(completed),
        completed_fixtures=completed_fixtures,
        started_at=time.monotonic(),
    )
    threads = [
        threading.Thread(
            target=worker,
            args=(gpu, task_queue, config, failures, progress, output_lock, stop, active, active_lock),
            name=f'sm90-scan-gpu{gpu}',
        )
        for gpu in gpus
    ]

    interrupted = False
    for thread in threads:
        thread.start()
    try:
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.2)
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
        print('\nInterrupted; terminating active scan tasks.', file=sys.stderr)
        terminate_active(active, active_lock)
    finally:
        for thread in threads:
            thread.join()

    merged, merged_tasks = merge_logs(args.outdir, tasks, signature)
    if failures:
        print(f'\n{len(failures)} task(s) failed:', file=sys.stderr)
        for task, reason in failures:
            print(f'  {task.task_id}: {reason}: {", ".join(task.case_names)}', file=sys.stderr)

    complete = merged_tasks == len(tasks)
    state = 'Done' if complete else 'Partial'
    print(f'\n{state}. Log: {merged} ({merged_tasks}/{len(tasks)} tasks)', file=sys.stderr)
    print(
        f'Aggregate usage: python -m tests.turbomind.linear.kernel_usage_report'
        f' {merged} --type {args.type}',
        file=sys.stderr,
    )
    if interrupted:
        return 130
    return 0 if complete and not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())

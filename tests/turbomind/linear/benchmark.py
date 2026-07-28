from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch

from tests.turbomind.linear.cases import VALID_SUITES, expand_suite
from tests.turbomind.linear.fixture import LinearFixture


@dataclass(frozen=True)
class BenchmarkRequest:
    validate_outputs: bool
    print_diffs: bool
    l2_flush: bool
    warmup: int
    iters: int
    tune: bool
    import_path: str | None
    export_path: str | None


class L2CacheFlusher:
    def __init__(self, device: torch.device):
        props = torch.cuda.get_device_properties(device)
        # allocate ~2x L2 to flush; mirror linear_attn.benchmark.L2CacheFlusher if present
        nbytes = int(getattr(props, 'L2_cache_size', 4 * 1024 * 1024) * 2)
        self._buf = torch.empty(nbytes, dtype=torch.uint8, device=device)

    def __call__(self) -> None:
        self._buf.fill_(0)


def resolve_tune_paths(args) -> tuple[bool, str | None, str | None]:
    tune = bool(args.tune) or bool(os.environ.get('TM_GEMM_TUNE'))
    import_path = args.import_path or os.environ.get('TM_GEMM_IMPORT')
    export_path = args.export_path or os.environ.get('TM_GEMM_EXPORT')
    if import_path:
        tune = False  # match testbed_v3: import disables tuning
    return tune, import_path, export_path


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='bench_linear')
    p.add_argument('--suite', choices=VALID_SUITES, default='smoke')
    p.add_argument('--case', type=str, default='', help='comma-separated LinearCase names')
    p.add_argument('--type', dest='type_names', type=str, default='', help='comma-separated TypeSpec names')
    p.add_argument('--shape', dest='shape_names', type=str, default='', help='comma-separated ShapeSpec names')
    p.add_argument('--batch', type=str, default='', help='comma-separated batch sizes')
    p.add_argument('--tp', type=str, default='1', help='comma-separated TP sizes')
    p.add_argument('--ep', type=str, default='1', help='comma-separated EP sizes')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--tune', action='store_true')
    p.add_argument('--import', dest='import_path', default=None)
    p.add_argument('--export', dest='export_path', default=None)
    p.add_argument('--print-diffs', action='store_true')
    p.add_argument('--no-validate', action='store_true')
    p.add_argument('--no-l2-flush', action='store_true')
    return p


def parse_int_list(s: str) -> tuple[int, ...] | None:
    if not s.strip():
        return None
    return tuple(int(x) for x in s.split(',') if x.strip())


def parse_name_list(s: str) -> tuple[str, ...] | None:
    if not s.strip():
        return None
    return tuple(x for x in s.split(',') if x.strip())


def flop_count(m: int, n: int, k: int, expert_tokens: int | None = None) -> int:
    # dense: 2*m*n*k; MoE packed: 2 * token_count * n * k
    if expert_tokens is None:
        return 2 * m * n * k
    return 2 * expert_tokens * n * k


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    case_names = parse_name_list(args.case)
    type_names = parse_name_list(args.type_names)
    shape_names = parse_name_list(args.shape_names)
    batches = parse_int_list(args.batch)
    tps = parse_int_list(args.tp)
    eps = parse_int_list(args.ep)
    runs = expand_suite(
        args.suite,
        case_names,
        batches,
        type_names,
        shape_names,
        tps=tps,
        eps=eps,
    )
    tune, import_path, export_path = resolve_tune_paths(args)
    device = torch.device('cuda')
    flusher = None if args.no_l2_flush else L2CacheFlusher(device)

    # Group runs by concrete local shape so weights build once per batch sweep.
    by_case: dict[tuple[str, int, int], list] = {}
    for run in runs:
        key = (run.case.name, run.case.tp, run.case.ep)
        by_case.setdefault(key, []).append(run)

    for case_runs in by_case.values():
        fx = LinearFixture(case_runs[0].case, device=device)
        try:
            assert fx.linear is not None
            if import_path:
                fx.linear.import_records(import_path)
            for run in case_runs:
                fx.prepare_batch(run.batch_size)
                row = {
                    'case': run.case.name,
                    'type': run.case.type_name,
                    'shape': run.case.shape_name,
                    'm': run.batch_size,
                    'n': run.case.output_dim,
                    'k': run.case.input_dim,
                    'data_type': run.case.data_type,
                    'weight_type': run.case.weight_type,
                    'input_type': run.case.input_type,
                    'expert_num': run.case.expert_num,
                    'experts_per_token': run.case.experts_per_token,
                    'tp': run.case.tp,
                    'ep': run.case.ep,
                    'max_tp': run.case.max_tp,
                    'max_ep': run.case.max_ep,
                    'fuse_silu': run.case.fuse_silu,
                }
                if not args.no_validate or args.print_diffs:
                    fx.run_reference()
                    fx.run_linear()
                    metrics = fx.compare()
                    if args.print_diffs or not args.no_validate:
                        row.update({f'{a}.{b}': c for a, d in metrics.items() for b, c in d.items()})
                    if not args.no_validate:
                        fx.check_tolerances(metrics)
                if tune or args.iters > 0:
                    assert fx.linear is not None
                    with fx.on_tm_stream() as stream:
                        # Tune before warmup/timed so measure overhead is not in TFLOPS.
                        # One forward is enough: GEMM measure policy runs its own internal iters.
                        if tune:
                            fx.linear.set_measure(True)
                            fx.run_linear_forward()
                            fx.sync_tm()
                            fx.linear.set_measure(False)
                            fx.release_forward_result()
                        if args.iters > 0:
                            for _ in range(args.warmup):
                                fx.run_linear_forward()
                                fx.release_forward_result()
                            fx.sync_tm()
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            elapsed_ms = 0.0
                            # L2 flush is on the TM stream, before the timing window.
                            for _ in range(args.iters):
                                if flusher is not None:
                                    flusher()
                                start.record(stream)
                                fx.run_linear_forward()
                                end.record(stream)
                                fx.release_forward_result()
                                end.synchronize()
                                elapsed_ms += start.elapsed_time(end)
                            ms = elapsed_ms / args.iters
                            row['latency_ms'] = ms
                            tokens = None
                            if run.case.expert_num:
                                tokens = run.batch_size * run.case.experts_per_token
                            flops = flop_count(run.batch_size, run.case.output_dim, run.case.input_dim, tokens)
                            row['tflops'] = (flops / (ms * 1e-3)) / 1e12
                if args.iters == 0:
                    row['latency_ms'] = 0.0
                    row['tflops'] = 0.0
                print(row)
            assert fx.linear is not None
            if export_path:
                # Per-case export so a full-suite scan does not overwrite previous records;
                # the dispatch cache is per-LinearFixture (per-Linear / per-Gemm).
                suffix = f'{case_runs[0].case.name}__tp{case_runs[0].case.tp}__ep{case_runs[0].case.ep}'
                fx.linear.export_records(f'{export_path}.{suffix}')
        finally:
            fx.close()
            # Release PyTorch cache between case groups; the MoE cases can be large and
            # the full-suite scan is run on a shared GPU.
            torch.cuda.empty_cache()
    return 0

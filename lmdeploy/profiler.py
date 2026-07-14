# Copyright (c) OpenMMLab. All rights reserved.
import csv
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class Session:
    """Represents a single profiling session."""

    UNKNOWN = 0
    SUCCESS = 1
    FAIL = 2

    def __init__(self, input_len: int, req_output_len: int) -> None:
        if input_len < 0:
            raise ValueError(f'input_len must be non-negative, got {input_len}')
        if req_output_len < 0:
            raise ValueError(f'req_output_len must be non-negative, got {req_output_len}')

        self.ts: List[float] = []
        self.ns: List[int] = []
        self.input_len = input_len
        self.req_output_len = req_output_len
        self.status = Session.UNKNOWN

    def tick(self, n_token: int) -> None:
        """Record a timing tick with the current token count."""
        if n_token < 0:
            raise ValueError(f'n_token must be non-negative, got {n_token}')
        self.ts.append(time.perf_counter())
        self.ns.append(n_token)

    def finish(self, status: int) -> None:
        """Mark the session as finished with the given status."""
        if status not in (Session.UNKNOWN, Session.SUCCESS, Session.FAIL):
            raise ValueError(f'Invalid session status: {status}')
        self.status = status


class Profiler:
    """Performance profiler for LLM inference."""

    def __init__(self, stream_output: bool, percentages: List[int]) -> None:
        if not all(0 <= p <= 100 for p in percentages):
            raise ValueError('All percentages must be in range [0, 100]')

        self.sessions: List[Session] = []
        self.stream_output = stream_output
        self.percentages = percentages
        self.t_start: float = 0.0
        self.elapsed_time: float = 0.0

    def new_session(self, input_len: int, req_output_len: int) -> Session:
        """Create a new profiling session."""
        sess = Session(input_len, req_output_len)
        self.sessions.append(sess)
        return sess

    def start(self) -> None:
        """Start the profiler timer."""
        self.t_start = time.perf_counter()

    def finish(self) -> None:
        """Finish the profiler timer."""
        self.elapsed_time = time.perf_counter() - self.t_start

    def compute_metrics(self) -> None:
        """Compute performance metrics from collected sessions."""
        ttfts: List[float] = []
        tpots: List[float] = []
        e2es: List[float] = []
        itls: List[float] = []
        tpts: List[int] = []
        total_output = 0
        total_input = 0
        success = 0

        for sess in self.sessions:
            if sess.status != Session.SUCCESS:
                continue
            if len(sess.ns) < 2 or len(sess.ts) < 2:
                logger.warning(f'Session has insufficient data points: {len(sess.ts)} ticks')
                continue
            ns = sess.ns
            ts = sess.ts
            if ns[-1] < sess.req_output_len:
                continue
            success += 1
            total_output += ns[-1]
            total_input += sess.input_len
            e2es.append(ts[-1] - ts[0])
            ttfts.append(ts[1] - ts[0])
            if ns[-1] > ns[1]:
                tpots.append((ts[-1] - ts[1]) / (ns[-1] - ns[1]))
            else:
                tpots.append((ts[-1] - ts[0]) / (ns[-1] - ns[0]))
            t_dif = np.subtract(ts[1:], ts[:-1])
            n_dif = np.subtract(ns[1:], ns[:-1])
            itls.extend(t_dif[1:])
            tpts.extend(n_dif)

        self.output_throughput = total_output / self.elapsed_time if self.elapsed_time > 0 else 0.0
        self.input_throughput = total_input / self.elapsed_time if self.elapsed_time > 0 else 0.0

        qs = self.percentages

        e2es = e2es or [float('inf')]
        tpots = tpots or [float('inf')]
        ttfts = ttfts or [float('inf')]
        itls = itls or [float('inf')]
        tpts = tpts or [0]

        self.tpot_mean = float(np.mean(tpots))
        self.tpot_stat = tuple(np.percentile(tpots, qs))
        self.e2e_mean = float(np.mean(e2es))
        self.e2e_stat = tuple(np.percentile(e2es, qs))

        if self.stream_output:
            self.ttft_mean = float(np.mean(ttfts))
            self.ttft_stat = tuple(np.percentile(ttfts, qs))
            self.itls_mean = float(np.mean(itls))
            self.itls_stat = tuple(np.percentile(itls, qs))
            self.tpts_mean = float(np.mean(tpts))
            self.tpts_stat = tuple(np.percentile(tpts, qs).astype(int))

        self.rps = success / self.elapsed_time if self.elapsed_time > 0 else 0.0
        self.total_output = total_output
        self.total_input = total_input
        self.success = success

    def summarize(self, title: str, hyperparams: Optional[List[Tuple[str, Any]]] = None,
                  header: int = 40, digits: int = 10) -> None:
        """Print a summary of the profiling results.

        Args:
            title: Title for the summary
            hyperparams: List of (key, value) tuples for additional parameters
            header: Width of the header column
            digits: Number of digits for numeric formatting
        """
        width = header + digits * (1 + len(self.percentages))

        def tab_row(name: str, *items: Any) -> None:
            def fmt(x: Any) -> str:
                return '{:>{d}.3f}'.format(x, d=digits) if isinstance(x, float) else '{:>{d}}'.format(x, d=digits)
            print('{:<{p}}{}'.format(name, ''.join([fmt(x) for x in items]), p=header))

        print('\n{s:{c}^{n}}'.format(s=f' {title} ', n=width, c='='))
        tab_row('Benchmark duration', self.elapsed_time)
        tab_row('Total requests', len(self.sessions))
        tab_row('Successful requests', self.success)
        if hyperparams:
            for k, v in hyperparams:
                tab_row(k, v)
        tab_row('Total input tokens', self.total_input)
        tab_row('Total generated tokens', self.total_output)
        tab_row('Input throughput (tok/s)', self.input_throughput)
        tab_row('Output throughput (tok/s)', self.output_throughput)
        tab_row('Request throughput (req/s)', self.rps)
        print('-' * width)
        tab_row('', 'mean', *(f'P{q}' for q in self.percentages))
        tab_row('End-to-end Latency', self.e2e_mean, *self.e2e_stat)
        if self.stream_output:
            tab_row('Time to First Token (TTFT)', self.ttft_mean, *self.ttft_stat)
        tab_row('Time per Output Token (TPOT)', self.tpot_mean, *self.tpot_stat)
        if self.stream_output:
            tab_row('Inter-token Latency (ITL)', self.itls_mean, *self.itls_stat)
            tab_row('Tokens per Tick', self.tpts_mean, *self.tpts_stat)
        print('=' * width)

    def save_csv(self, csv_file: str, hyperparams: List[Tuple[str, Any]]) -> None:
        """Export legacy metrics to CSV.

        Args:
            csv_file: Path to the CSV file
            hyperparams: List of (key, value) tuples for additional parameters
        """
        try:
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                keys, vals = zip(*hyperparams)
                if not file_exists:
                    writer.writerow([
                        *keys,
                        'completed',
                        'total_input_tokens',
                        'total_output_tokens',
                        'duration',
                        'request_throughput',
                        'input_throughput',
                        'output_throughput',
                        'mean_e2e_latency_ms',
                        'mean_ttft_ms',
                        'mean_tpot_ms',
                        'mean_itl_ms',
                    ])
                ttft_ms = f'{getattr(self, "ttft_mean", 0) * 1000:.3f}' if self.stream_output else '-'
                itl_ms = f'{getattr(self, "itls_mean", 0) * 1000:.3f}' if self.stream_output else '-'
                writer.writerow([
                    *vals,
                    self.success,
                    self.total_input,
                    self.total_output,
                    self.elapsed_time,
                    f'{self.rps:.3f}',
                    f'{self.input_throughput:.3f}',
                    f'{self.output_throughput:.3f}',
                    f'{self.e2e_mean * 1000:.3f}',
                    ttft_ms,
                    f'{self.tpot_mean * 1000:.3f}',
                    itl_ms,
                ])
        except IOError as e:
            logger.error(f'Failed to save CSV file {csv_file}: {e}')
            raise

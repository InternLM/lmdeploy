# Copyright (c) OpenMMLab. All rights reserved.
import csv
import os
import time
from typing import List

import numpy as np


class Session:

    UNKNOWN = 0
    SUCCESS = 1
    FAIL = 2

    def __init__(self, input_len, req_output_len):
        self.ts = []
        self.ns = []
        self.input_len = input_len
        self.req_output_len = req_output_len
        self.status = Session.UNKNOWN

    def tick(self, n_token):
        self.ts.append(time.perf_counter())
        self.ns.append(n_token)

    def finish(self, status):
        self.status = status


class Profiler:

    def __init__(self, stream_output: bool, percentages: List[int]):
        self.sessions: List[Session] = []
        self.stream_output = stream_output
        self.percentages = percentages

    def new_session(self, *args, **kwargs):
        sess = Session(*args, **kwargs)
        self.sessions.append(sess)
        return sess

    def start(self):
        self.t_start = time.perf_counter()

    def finish(self):
        self.elapsed_time = time.perf_counter() - self.t_start

    def compute_metrics(self):
        self.ttfts: List[float] = []
        self.tpots: List[float] = []
        self.e2es: List[float] = []
        self.itls: List[float] = []
        self.tpts: List[int] = []
        self.total_output = 0
        self.total_input = 0
        self.success = 0

        for sess in self.sessions:
            if sess.status != Session.SUCCESS:
                continue
            ns = sess.ns
            ts = sess.ts
            if ns[-1] < sess.req_output_len:
                continue
            self.success += 1
            self.total_output += ns[-1]
            self.total_input += sess.input_len
            self.e2es.append(ts[-1] - ts[0])
            self.ttfts.append(ts[1] - ts[0])
            if ns[-1] > ns[1]:
                self.tpots.append((ts[-1] - ts[1]) / (ns[-1] - ns[1]))
            else:  # no-stream-output
                self.tpots.append((ts[-1] - ts[0]) / (ns[-1] - ns[0]))
            t_dif = np.subtract(ts[1:], ts[:-1])
            n_dif = np.subtract(ns[1:], ns[:-1])
            self.itls.extend(t_dif[1:])
            self.tpts.extend(n_dif)

        self.output_throughput = self.total_output / self.elapsed_time
        self.input_throughput = self.total_input / self.elapsed_time

        qs = self.percentages

        self.e2es = self.e2es or [float('inf')]
        self.tpots = self.tpots or [float('inf')]
        self.ttfts = self.ttfts or [float('inf')]
        self.itls = self.itls or [float('inf')]
        self.tpts = self.tpts or [0]

        self.tpot_mean = np.mean(self.tpots)
        self.tpot_stat = tuple(np.percentile(self.tpots, qs))
        self.e2e_mean = np.mean(self.e2es)
        self.e2e_stat = tuple(np.percentile(self.e2es, qs))

        if self.stream_output:
            self.ttft_mean = np.mean(self.ttfts)
            self.ttft_stat = tuple(np.percentile(self.ttfts, qs))
            self.itls_mean = np.mean(self.itls)
            self.itls_stat = tuple(np.percentile(self.itls, qs))
            self.tpts_mean = np.mean(self.tpts)
            self.tpts_stat = tuple(np.percentile(self.tpts, qs).astype(int))

        self.rps = self.success / self.elapsed_time

    def summarize(self, title: str, hyperparams: List = None, header=40, digits=10):

        width = header + digits * (1 + len(self.percentages))

        def tab_row(name, *items):

            def fmt(x):
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

    def save_csv(self, csv_file: str, hyperparams):
        """Export legacy metrics to CSV."""
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as csvfile:
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
            writer.writerow([
                *vals,
                self.success,
                self.total_input,
                self.total_output,
                self.elapsed_time,
                f'{self.rps:.3f}',
                f'{(self.input_throughput):.3f}',
                f'{self.output_throughput:.3f}',
                f'{self.e2e_mean*1000:.3f}',
                f'{self.ttft_mean*1000:.3f}' if self.stream_output else '-',
                f'{self.tpot_mean*1000:.3f}',
                f'{self.itls_mean*1000:.3f}' if self.stream_output else '-',
            ])

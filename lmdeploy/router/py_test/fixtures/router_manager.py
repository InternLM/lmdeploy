# Copyright (c) OpenMMLab. All rights reserved.
import subprocess
import time
from dataclasses import dataclass

import requests

from .ports import find_free_ports


@dataclass
class ProcHandle:
    process: subprocess.Popen
    url: str


class RouterStartupError(RuntimeError):
    """Raised when the router fails before its HTTP server starts."""


class RouterManager:
    """Helper to spawn a router process and interact with admin endpoints."""

    def __init__(self):
        self._children: list[subprocess.Popen] = []

    def start_router(
        self,
        worker_urls: list[str] | None = None,
        policy: str = 'round_robin',
        port: int | None = None,
        extra: dict | None = None,
        # PD options
        lmdeploy_pd_disaggregation: bool = False,
        prefill_urls: list[str] | None = None,
        decode_urls: list[str] | None = None,
        prefill_policy: str | None = None,
        decode_policy: str | None = None,
    ) -> ProcHandle:
        worker_urls = worker_urls or []
        retry_count = 0 if port is not None else 2
        while True:
            try:
                return self._start_router_process(
                    worker_urls=worker_urls,
                    policy=policy,
                    port=port,
                    extra=extra,
                    lmdeploy_pd_disaggregation=lmdeploy_pd_disaggregation,
                    prefill_urls=prefill_urls,
                    decode_urls=decode_urls,
                    prefill_policy=prefill_policy,
                    decode_policy=decode_policy,
                )
            except RouterStartupError:
                if retry_count == 0:
                    raise
                retry_count -= 1
                port = None

    def _start_router_process(
        self,
        worker_urls: list[str],
        policy: str,
        port: int | None,
        extra: dict | None,
        lmdeploy_pd_disaggregation: bool,
        prefill_urls: list[str] | None,
        decode_urls: list[str] | None,
        prefill_policy: str | None,
        decode_policy: str | None,
    ) -> ProcHandle:
        router_port, prom_port = (
            [port, *find_free_ports(1)] if port is not None else find_free_ports(2)
        )
        cmd = [
            'python3',
            '-m',
            'lmdeploy_router.launch_router',
            '--host',
            '127.0.0.1',
            '--port',
            str(router_port),
            '--policy',
            policy,
        ]
        # Avoid Prometheus port collisions by assigning unique free ports.
        cmd.extend(
            ['--prometheus-port', str(prom_port), '--prometheus-host', '127.0.0.1']
        )
        if worker_urls:
            cmd.extend(['--worker-urls', *worker_urls])

        # PD routing configuration
        if lmdeploy_pd_disaggregation:
            cmd.append('--lmdeploy-pd-disaggregation')
            if prefill_urls:
                for url in prefill_urls:
                    cmd.extend(['--prefill', url])
            if decode_urls:
                for url in decode_urls:
                    cmd.extend(['--decode', url])
            if prefill_policy:
                cmd.extend(['--prefill-policy', prefill_policy])
            if decode_policy:
                cmd.extend(['--decode-policy', decode_policy])

        # Map supported extras to CLI flags (subset for integration)
        if extra:
            flag_map = {
                'max_payload_size': '--max-payload-size',
                'api_key': '--api-key',
                # Health/monitoring
                'worker_startup_check_interval': '--worker-startup-check-interval',
                # Cache-aware tuning
                'cache_threshold': '--cache-threshold',
                'balance_abs_threshold': '--balance-abs-threshold',
                'balance_rel_threshold': '--balance-rel-threshold',
                # Retry
                'retry_max_retries': '--retry-max-retries',
                'retry_initial_backoff_ms': '--retry-initial-backoff-ms',
                'retry_max_backoff_ms': '--retry-max-backoff-ms',
                'retry_backoff_multiplier': '--retry-backoff-multiplier',
                'retry_jitter_factor': '--retry-jitter-factor',
                'disable_retries': '--disable-retries',
                # Circuit breaker
                'cb_failure_threshold': '--cb-failure-threshold',
                'cb_success_threshold': '--cb-success-threshold',
                'cb_timeout_duration_secs': '--cb-timeout-duration-secs',
                'cb_window_duration_secs': '--cb-window-duration-secs',
                'disable_circuit_breaker': '--disable-circuit-breaker',
                # Rate limiting
                'max_concurrent_requests': '--max-concurrent-requests',
                'queue_size': '--queue-size',
                'queue_timeout_secs': '--queue-timeout-secs',
                'rate_limit_tokens_per_second': '--rate-limit-tokens-per-second',
            }
            for k, v in extra.items():
                if v is None:
                    continue
                flag = flag_map.get(k)
                if not flag:
                    continue
                if isinstance(v, bool):
                    if v:
                        cmd.append(flag)
                else:
                    cmd.extend([flag, str(v)])

        proc = subprocess.Popen(cmd)
        self._children.append(proc)
        url = f'http://127.0.0.1:{router_port}'
        try:
            self._wait_health(url)
        except TimeoutError:
            if self._router_failed_before_serving(proc):
                raise RouterStartupError(f'Router at {url} failed during startup') from None
            raise
        return ProcHandle(process=proc, url=url)

    @staticmethod
    def _router_failed_before_serving(process: subprocess.Popen) -> bool:
        if process.poll() is None:
            return False
        stdout, _ = process.communicate(timeout=2)
        return 'Router ready | workers:' not in stdout

    def _wait_health(self, base_url: str, timeout: float = 30.0):
        start = time.time()
        with requests.Session() as s:
            while time.time() - start < timeout:
                try:
                    r = s.get(f'{base_url}/health', timeout=2)
                    if r.status_code == 200:
                        return
                except requests.RequestException:
                    pass
                time.sleep(0.2)
        raise TimeoutError(f'Router at {base_url} did not become healthy')

    def add_worker(self, base_url: str, worker_url: str) -> None:
        r = requests.post(f'{base_url}/add_worker', params={'url': worker_url})
        assert r.status_code == 200, f'add_worker failed: {r.status_code} {r.text}'

    def remove_worker(self, base_url: str, worker_url: str) -> None:
        r = requests.post(f'{base_url}/remove_worker', params={'url': worker_url})
        assert r.status_code == 200, f'remove_worker failed: {r.status_code} {r.text}'

    def add_lmdeploy_node(self, base_url: str, worker_url: str, role: int) -> None:
        response = requests.post(
            f'{base_url}/nodes/add', json={'url': worker_url, 'status': {'role': role}}
        )
        assert response.status_code == 200, response.text

    def remove_lmdeploy_node(self, base_url: str, worker_url: str, role: int) -> None:
        response = requests.post(
            f'{base_url}/nodes/remove',
            json={'url': worker_url, 'status': {'role': role}},
        )
        assert response.status_code == 200, response.text

    def lmdeploy_nodes(self, base_url: str) -> dict:
        response = requests.get(f'{base_url}/nodes/status')
        assert response.status_code == 200, response.text
        return response.json()

    def list_workers(self, base_url: str) -> list[str]:
        r = requests.get(f'{base_url}/list_workers')
        assert r.status_code == 200, f'list_workers failed: {r.status_code} {r.text}'
        data = r.json()
        return data.get('urls', [])

    def stop_all(self):
        for p in self._children:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        self._children.clear()

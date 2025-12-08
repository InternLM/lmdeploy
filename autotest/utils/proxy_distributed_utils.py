import os
import random
import socket
import subprocess
import time
from typing import Any, Dict, Tuple

import requests
from utils.ray_distributed_utils import verify_service_functionality

time_time = time.time

DEFAULT_PROXY_PORT = 8000
WORKER_WAIT_INTERVAL = 15  # seconds


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False


def check_nodes_status(host: str, proxy_port: int, model_name: str, expected_instances: int, check_count: int,
                       current_time: float, last_progress_print: float,
                       progress_print_interval: int) -> Tuple[bool, int]:
    try:
        nodes_url = f'http://{host}:{proxy_port}/nodes/status'
        resp = requests.get(nodes_url, timeout=10)

        if resp.status_code != 200:
            if current_time - last_progress_print >= progress_print_interval:
                print(f'🔧 Check {check_count}: Failed to get node status, status code: {resp.status_code}')
            return False, 0

        nodes_data = resp.json()
        ready_instances = 0
        total_instances = len(nodes_data)

        for node_info in nodes_data.values():
            models = node_info.get('models', [])
            if model_name in models:
                ready_instances += 1

        should_print = current_time - last_progress_print >= progress_print_interval

        if should_print:
            basename = os.path.basename(model_name)
            print(f'📊 Check {check_count}: Model registration progress: '
                  f'{ready_instances}/{expected_instances} instances ready '
                  f'(Total reported: {total_instances})')
            for node_url, node_info in nodes_data.items():
                models = node_info.get('models', [])
                if model_name in models:
                    print(f'   ✅ Instance {node_url} registered model {basename}')
                else:
                    print(f'   ⏳ Instance {node_url} has not registered target model')

        if ready_instances >= expected_instances:
            if should_print:
                print(f'🎯 All {expected_instances} API server instances have registered the target model')
            return True, ready_instances
        else:
            if should_print:
                print(f'⏳ Waiting for more instances to register... ({ready_instances}/{expected_instances})')
            return False, ready_instances

    except Exception as e:
        if current_time - last_progress_print >= progress_print_interval:
            print(f'🔧 Check {check_count}: Exception getting node status - {e}')
        return False, 0


def wait_for_model_service_ready(host: str,
                                 proxy_port: int,
                                 model_name: str,
                                 timeout_seconds: int = 2000,
                                 expected_instances: int = None) -> bool:
    if expected_instances:
        print(f'⏳ Waiting for model service to be fully ready (Model: {model_name}), '
              f'expected instances: {expected_instances}, timeout: {timeout_seconds}s')
    else:
        print(f'⏳ Waiting for model service to be fully ready (Model: {model_name}), '
              f'timeout: {timeout_seconds}s')

    start_time = time_time()
    check_count = 0
    last_progress_print = 0
    progress_print_interval = 30

    initial_delay = random.uniform(1, 5)
    time.sleep(initial_delay)

    while time_time() - start_time < timeout_seconds:
        check_count += 1
        current_time = time_time()

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5)
                if sock.connect_ex((host, proxy_port)) != 0:
                    if current_time - last_progress_print >= progress_print_interval:
                        print(f'🔌 Check {check_count}: proxy port not ready')
                        last_progress_print = current_time
                    time.sleep(10)
                    continue

            if expected_instances:
                instances_ready, ready_count = check_nodes_status(host, proxy_port, model_name, expected_instances,
                                                                  check_count, current_time, last_progress_print,
                                                                  progress_print_interval)
                if not instances_ready:
                    if ready_count is not None and current_time - last_progress_print >= progress_print_interval:
                        last_progress_print = current_time
                    time.sleep(10)
                    continue

            service_ready = verify_service_functionality(host, proxy_port, model_name, check_count)
            if service_ready:
                if expected_instances:
                    print(f'✅ All {expected_instances} API server instances are ready and service is functional!')
                else:
                    print('✅ Model service is fully ready!')
                return True

        except requests.exceptions.RequestException as e:
            if current_time - last_progress_print >= progress_print_interval:
                print(f'🔧 Check {check_count}: Request exception - {e}')
                last_progress_print = current_time
        except Exception as e:
            if current_time - last_progress_print >= progress_print_interval:
                print(f'🔧 Check {check_count}: Unknown exception - {e}')
                last_progress_print = current_time

        sleep_time = 10 + random.uniform(-2, 2)
        time.sleep(sleep_time)

    print(f'❌ Model service startup timed out ({timeout_seconds} seconds)')
    return False


def proxy_worker_node_wait(manager, timeout_minutes: int = 120):
    """Worker node waits by periodically checking if the master's proxy service
    is still alive. If the proxy becomes unreachable for several consecutive
    checks, assume master has finished.

    Args:
        manager: ProxyDistributedManager instance
        timeout_minutes: Maximum time to wait before giving up (default: 120 minutes)
    """
    print(f'⏸️ Worker node {manager.node_rank} entering monitoring mode...')

    max_checks = (timeout_minutes * 60) // WORKER_WAIT_INTERVAL
    consecutive_failures = 0
    max_consecutive_failures = 3

    for i in range(max_checks):
        if not is_port_open(manager.master_addr, manager.proxy_port, timeout=2.0):
            consecutive_failures += 1
            print(f'⚠️ Proxy connection to master failed ({consecutive_failures}/{max_consecutive_failures})')
            if consecutive_failures >= max_consecutive_failures:
                print('📡 Master proxy service stopped, worker node exiting')
                break
        else:
            consecutive_failures = 0

        if i % 4 == 0:
            elapsed = (i * WORKER_WAIT_INTERVAL) // 60
            print(f'⏳ Worker node {manager.node_rank} monitoring... Running for {elapsed} minutes')

        time.sleep(WORKER_WAIT_INTERVAL)
    else:
        print(f'⏰ Worker node {manager.node_rank} monitoring timed out ({timeout_minutes} minutes)')

    print(f'✅ Worker node {manager.node_rank} completed waiting')


class ProxyDistributedManager:

    def __init__(self):
        self.master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
        self.node_rank = int(os.getenv('NODE_RANK', '0'))
        self.proxy_port = int(os.getenv('PROXY_PORT', str(DEFAULT_PROXY_PORT)))

        self.is_master = (self.node_rank == 0)
        self.proxy_process = None

    def start(self):
        if not self.is_master:
            return

        cmd = [
            'lmdeploy', 'serve', 'proxy', '--server-name', self.master_addr, '--server-port',
            str(self.proxy_port), '--routing-strategy', 'min_expected_latency', '--serving-strategy', 'Hybrid'
        ]
        print(f"[Proxy] Starting: {' '.join(cmd)}")
        self.proxy_process = subprocess.Popen(cmd)

        time.sleep(5)

    def cleanup(self):
        if self.proxy_process and self.proxy_process.poll() is None:
            print('[Proxy] Terminating proxy process...')
            self.proxy_process.terminate()
            try:
                self.proxy_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proxy_process.kill()


class ApiServerPerTest:

    def __init__(self, proxy_manager: ProxyDistributedManager, model_path: str, model_param: Dict[str, Any]):
        self.proxy_manager = proxy_manager
        self.model_path = model_path
        self.model_param = model_param or {}

        self.master_addr = proxy_manager.master_addr
        self.proxy_port = proxy_manager.proxy_port
        self.node_rank = int(os.getenv('NODE_RANK', '0'))
        self.node_count = int(os.getenv('NODE_COUNT', '1'))
        self.proc_per_node = int(os.getenv('PROC_PER_NODE', '1'))

        self.backend = self.model_param.get('backend', 'turbomind')
        self.communicator = self.model_param.get('communicator', 'nccl')
        self.quant_policy = self.model_param.get('quant_policy', 0)
        self.tp = int(self.model_param.get('tp', 1))
        parallel_config = self.model_param.get('parallel_config', {})
        self.ep = int(parallel_config.get('ep', 1))
        self.dp = int(parallel_config.get('dp', 1))
        self.extra = self.model_param.get('extra', '')

        self.expected_instances = self.node_count * self.proc_per_node
        self.is_master = (self.node_rank == 0)
        self.api_process = None

    def start(self):
        proxy_url = f'http://{self.master_addr}:{self.proxy_port}'
        cmd = [
            'lmdeploy',
            'serve',
            'api_server',
            self.model_path,
            '--backend',
            str(self.backend),
            '--proxy-url',
            proxy_url,
        ]
        if self.node_count > 1:
            cmd += ['--nnodes', str(self.node_count), '--node-rank', str(self.node_rank)]
        if self.quant_policy != 0:
            cmd += ['--quant-policy', str(self.quant_policy)]

        if self.backend == 'turbomind':
            cmd += ['--communicator', str(self.communicator)]

        if self.ep != 1:
            cmd += ['--ep', str(self.ep)]
        if self.dp != 1:
            cmd += ['--dp', str(self.dp)]
        if self.tp != 1:
            cmd += ['--tp', str(self.tp)]
        if self.extra.strip() != '':
            extra_args = self.extra.strip().split()
            cmd.extend(extra_args)

        print(f"[API Server] Starting: {' '.join(cmd)}")
        self.api_process = subprocess.Popen(cmd)

    def wait_until_ready(self):
        if not self.is_master:
            return
        success = wait_for_model_service_ready(host=self.master_addr,
                                               proxy_port=self.proxy_port,
                                               model_name=self.model_path,
                                               timeout_seconds=2000,
                                               expected_instances=self.expected_instances)
        if not success:
            raise RuntimeError(f'API Server failed to register model: {self.model_path}')

    def cleanup(self):
        if self.api_process and self.api_process.poll() is None:
            print(f'[API Server] Terminating for model: {self.model_path}')
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.api_process.kill()

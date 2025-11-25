import os
import random
import socket
import subprocess
import time
from time import time as time_time
from typing import Any, Dict, Tuple

import requests

# Default constants
LM_DEPLOY_PROXY_PORT = 8000
HEALTH_CHECK_TIMEOUT = 30
CONNECTION_CHECK_TIMEOUT = 5
WORKER_WAIT_INTERVAL = 30


def wait_for_model_service_ready(host: str,
                                 proxy_port: int,
                                 model_name: str,
                                 timeout_seconds: int = 1500,
                                 expected_nodes: int = None) -> bool:
    """Wait for LM Deploy Proxy + backend workers to be fully ready, ensuring
    all nodes are registered.

    Check all nodes' readiness status through /nodes/status API.
    """
    if expected_nodes:
        print(f'⏳ Waiting for model service to be fully ready (Model: {model_name}), '
              f'expected nodes: {expected_nodes}, timeout: {timeout_seconds}s')
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

            if expected_nodes:
                nodes_ready, ready_nodes = check_nodes_status(host, proxy_port, model_name, expected_nodes, check_count,
                                                              current_time, last_progress_print,
                                                              progress_print_interval)
                if not nodes_ready:
                    if ready_nodes is not None and current_time - last_progress_print >= progress_print_interval:
                        last_progress_print = current_time
                    continue

            service_ready = verify_service_functionality(host, proxy_port, model_name, check_count)
            if service_ready:
                if expected_nodes:
                    print(f'✅ All {expected_nodes} nodes are ready and service is functional!')
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


def check_nodes_status(host: str, proxy_port: int, model_name: str, expected_nodes: int, check_count: int,
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
        ready_nodes = 0
        total_nodes = len(nodes_data)

        for node_url, node_info in nodes_data.items():
            models = node_info.get('models', [])
            if model_name in models:
                ready_nodes += 1

        should_print = current_time - last_progress_print >= progress_print_interval

        if should_print:
            print(f'📊 Check {check_count}: Node readiness progress: {ready_nodes}/{expected_nodes} '
                  f'(Total nodes: {total_nodes})')
            for node_url, node_info in nodes_data.items():
                models = node_info.get('models', [])
                basename = os.path.basename(model_name)
                if model_name in models:
                    print(f'   ✅ Node {node_url} registered model {basename}')
                else:
                    print(f'   ⏳ Node {node_url} has not registered target model')

        if ready_nodes >= expected_nodes:
            if should_print:
                print(f'🎯 All {expected_nodes} nodes have registered the target model')
            return True, ready_nodes
        else:
            if should_print:
                print(f'⏳ Waiting for more nodes to register... ({ready_nodes}/{expected_nodes})')
            return False, ready_nodes

    except Exception as e:
        if current_time - last_progress_print >= progress_print_interval:
            print(f'🔧 Check {check_count}: Exception getting node status - {e}')
        return False, 0


def verify_service_functionality(host: str, proxy_port: int, model_name: str, check_count: int) -> bool:
    try:
        test_data = {
            'model': model_name,
            'messages': [{
                'role': 'user',
                'content': 'hi'
            }],
            'max_tokens': 5,
            'stream': False
        }

        resp = requests.post(f'http://{host}:{proxy_port}/v1/chat/completions', json=test_data, timeout=15)

        if resp.status_code == 200:
            print(f'✅ Check {check_count}: Service functionality OK (received valid response)')
            return True
        elif resp.status_code == 400:
            print(f'✅ Check {check_count}: Service framework activated (received 400)')
            return True
        else:
            print(f'🔧 Check {check_count}: Service functionality test failed, status code: {resp.status_code}')
            return False

    except requests.exceptions.RequestException as e:
        print(f'🔧 Check {check_count}: Service functionality test exception - {e}')
        return False


class ProxyDistributedManager:

    def __init__(self, health_check: bool = True, proxy_port: int = None, log_dir: str = '.'):
        self.health_check = health_check
        self.proxy_port = proxy_port or LM_DEPLOY_PROXY_PORT
        self.log_dir = log_dir
        self._cleaned = False

        self._lmdeploy_proxy_process = None
        self._local_lmdeploy_process = None

        self.node_rank = int(os.getenv('NODE_RANK', '0'))
        self.is_master = (self.node_rank == 0)

        os.makedirs(self.log_dir, exist_ok=True)

        role = 'master' if self.is_master else 'worker'
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_filename = f'lmdeploy_{role}_rank{self.node_rank}_{timestamp}.log'
        self._lmdeploy_log_path = os.path.join(self.log_dir, log_filename)

        self._setup_from_env()
        self._setup_distributed_cluster()

        print(f'📝 Node {self.node_rank} LMDeploy log path: {self._lmdeploy_log_path}')

    def _setup_from_env(self):
        self.node_count = int(os.getenv('NODE_COUNT', '1'))
        self.master_addr = os.getenv('MASTER_ADDR', 'localhost')
        self.proc_per_node = int(os.getenv('PROC_PER_NODE', '1'))
        self.job_id = os.getenv('JOB_ID', 'unknown')
        self.total_gpus = self.node_count * self.proc_per_node

        print(f'🎯 Node {self.node_rank} distributed environment info:')
        print(f'  - Nodes: {self.node_count} nodes × {self.proc_per_node} GPUs = {self.total_gpus} GPUs')
        print(f"  - Current: Rank {self.node_rank} ({'Master node' if self.is_master else 'Worker node'})")
        print(f'  - Master address: {self.master_addr}')
        print(f'  - Proxy port: {self.proxy_port}')
        print(f'  - Job ID: {self.job_id}')

    def _setup_distributed_cluster(self):
        if self.is_master:
            self._start_lmdeploy_proxy()
        if self.health_check:
            self._basic_health_check()

    def _start_lmdeploy_proxy(self):
        print(f'🚀 Master node starting lmdeploy proxy (port: {self.proxy_port})...')
        env = os.environ.copy()
        self._lmdeploy_proxy_process = subprocess.Popen([
            'lmdeploy',
            'serve',
            'proxy',
            '--server-name',
            self.master_addr,
            '--server-port',
            str(self.proxy_port),
            '--routing-strategy',
            'min_expected_latency',
            '--serving-strategy',
            'Hybrid',
        ],
                                                        env=env)
        time.sleep(10)

        if self._check_service_health(self.proxy_port):
            print('✅ lmdeploy proxy started successfully')
        else:
            print('⚠️ lmdeploy proxy may have issues starting')

    def start_lmdeploy_api_server_async(self,
                                        model_path: str,
                                        model_param: dict,
                                        start_timeout: int = 1500) -> Tuple[int, subprocess.Popen]:
        total_gpus_per_node = self.proc_per_node
        total_nodes = self.node_count

        ep = total_gpus_per_node * total_nodes
        dp = total_gpus_per_node * total_nodes

        backend = model_param.get('backend', 'turbomind')
        communicator = model_param.get('communicator', 'nccl')
        quant_policy = model_param.get('quant_policy', 0)

        full_command = [
            'lmdeploy', 'serve', 'api_server', model_path, '--backend', backend, '--tp',
            str(1), '--ep',
            str(ep), '--dp',
            str(dp), '--proxy-url', f'http://{self.master_addr}:{self.proxy_port}', '--nnodes',
            str(total_nodes), '--node-rank',
            str(self.node_rank), '--communicator', communicator
        ]

        if backend == 'turbomind':
            full_command.extend(['--quant-policy', str(quant_policy)])

        cmd = ' '.join(full_command)
        print(f'🎯 Node {self.node_rank} start command: {cmd}')

        env = os.environ.copy()
        env.update({
            'DEEPEP_MAX_BATCH_SIZE': '256',
        })

        if dp > 1:
            env.update({
                'LMDEPLOY_DP_MASTER_ADDR': self.master_addr,
                'LMDEPLOY_DP_MASTER_PORT': '29555',
            })

        log_file = open(self._lmdeploy_log_path, 'w')

        try:
            self._local_lmdeploy_process = subprocess.Popen(full_command,
                                                            stdout=log_file,
                                                            stderr=log_file,
                                                            env=env,
                                                            text=True,
                                                            encoding='utf-8')
            pid = self._local_lmdeploy_process.pid
            print(f'🚀 Node {self.node_rank} started lmdeploy api_server (PID: {pid}), log: {self._lmdeploy_log_path}')

            if self.health_check:
                expected_nodes = self.node_count
                ready = wait_for_model_service_ready(host=self.master_addr,
                                                     proxy_port=self.proxy_port,
                                                     model_name=model_path,
                                                     timeout_seconds=start_timeout,
                                                     expected_nodes=expected_nodes)
                if not ready:
                    print(f'❌ Node {self.node_rank}: Model service could not be ready within timeout, '
                          f'terminating local process')
                    self._local_lmdeploy_process.terminate()
                    try:
                        self._local_lmdeploy_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._local_lmdeploy_process.kill()
                    log_file.close()
                    return 0, self._local_lmdeploy_process

            log_file.close()
            return pid, self._local_lmdeploy_process

        except Exception as e:
            print(f'💥 Node {self.node_rank} failed to start lmdeploy api_server: {e}')
            log_file.close()
            raise

    def is_lmdeploy_running(self):
        return self._local_lmdeploy_process is not None and self._local_lmdeploy_process.poll() is None

    def _basic_health_check(self):
        print(f'🔍 Node {self.node_rank} performing basic health check...')
        if self.is_master:
            ok = self._check_service_health(self.proxy_port)
            status = '✅ lmdeploy proxy service healthy' if ok else '⚠️ lmdeploy proxy service may have issues'
        else:
            ok = self._check_connection_to_master(self.proxy_port)
            status = '✅ Connection to master node normal' if ok else '⚠️ Connection to master node may have issues'
        print(status)

    def _check_service_health(self, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(HEALTH_CHECK_TIMEOUT)
                return sock.connect_ex((self.master_addr, port)) == 0
        except Exception:
            return False

    def _check_connection_to_master(self, port: int = None) -> bool:
        p = port or self.proxy_port
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(CONNECTION_CHECK_TIMEOUT)
                return sock.connect_ex((self.master_addr, p)) == 0
        except Exception:
            return False

    def get_cluster_info(self) -> Dict[str, Any]:
        return {
            'node_rank': self.node_rank,
            'node_count': self.node_count,
            'master_addr': self.master_addr,
            'proc_per_node': self.proc_per_node,
            'total_gpus': self.total_gpus,
            'job_id': self.job_id,
            'is_master': self.is_master,
            'proxy_port': self.proxy_port
        }

    def cleanup(self, force: bool = True):
        """Clean up resources.

        Args:
            force (bool):
                - False: Only stop LMDeploy API Server (used after individual test completion)
                - True: Stop API Server + Proxy (if master) and mark as fully cleaned (session end)
        """
        if self._cleaned and force:
            return

        print(f'🧹 Node {self.node_rank} cleaning resources... (force={force})')

        # --- Stop local LMDeploy API Server (all nodes) ---
        if hasattr(self, '_local_lmdeploy_process') and self._local_lmdeploy_process is not None:
            if self._local_lmdeploy_process.poll() is None:
                try:
                    self._local_lmdeploy_process.terminate()
                    self._local_lmdeploy_process.wait(timeout=10)
                    print(f'✅ Node {self.node_rank}: LMDeploy API Server stopped')
                except subprocess.TimeoutExpired:
                    print(f'⚠️ Node {self.node_rank}: API Server stop timeout, forcing kill')
                    self._local_lmdeploy_process.kill()

        # --- Stop LMDeploy Proxy (master node only, only when force=True) ---
        if force and self.is_master:
            if hasattr(self, '_lmdeploy_proxy_process') and self._lmdeploy_proxy_process is not None:
                if self._lmdeploy_proxy_process.poll() is None:
                    try:
                        self._lmdeploy_proxy_process.terminate()
                        self._lmdeploy_proxy_process.wait(timeout=10)
                        print('✅ LMDeploy Proxy stopped')
                    except subprocess.TimeoutExpired:
                        print('⚠️ LMDeploy Proxy stop timeout, forcing kill')
                        self._lmdeploy_proxy_process.kill()

        # Mark as fully cleaned only on final cleanup
        if force:
            self._cleaned = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup(force=True)


def proxy_worker_node_wait(manager: ProxyDistributedManager, timeout_minutes: int = 60):
    print(f'⏸️ Worker node {manager.node_rank} entering monitoring mode...')

    max_checks = (timeout_minutes * 60) // WORKER_WAIT_INTERVAL
    consecutive_failures = 0
    max_consecutive_failures = 3

    for i in range(max_checks):
        if not manager._check_connection_to_master():
            consecutive_failures += 1
            print(f'⚠️ Master node connection failed ({consecutive_failures}/{max_consecutive_failures})')
            if consecutive_failures >= max_consecutive_failures:
                print('📡 Master node service stopped, worker node exiting')
                break
        else:
            consecutive_failures = 0

        if i % 4 == 0:
            elapsed = (i * WORKER_WAIT_INTERVAL) // 60
            print(f'⏳ Worker node {manager.node_rank} monitoring... Running for {elapsed} minutes')

        time.sleep(WORKER_WAIT_INTERVAL)
    else:
        print(f'⏰ Worker node {manager.node_rank} monitoring timed out ({timeout_minutes} minutes)')

    manager.cleanup(force=False)  # Worker node only cleans up its own API Server when exiting
    print(f'✅ Worker node {manager.node_rank} completed waiting')

import os
import random
import socket
import subprocess
import time
from time import time as time_time
from typing import Any, Dict

import requests

# Default constants
LM_DEPLOY_API_PORT = 8000
RAY_PORT = 6379
HEALTH_CHECK_TIMEOUT = 30
CONNECTION_CHECK_TIMEOUT = 5
WORKER_WAIT_INTERVAL = 30


def wait_for_model_service_ready(
    host: str,
    api_port: int,
    model_name: str,
    timeout_seconds: int = 1000,
) -> bool:
    """Wait for LMDeploy API Server to be ready and verify basic functionality.

    No longer checks multi-node registration (API Server is a single-point service).
    """
    print(f'⏳ Waiting for LMDeploy API Server to be ready (Model: {model_name}), Timeout: {timeout_seconds}s')

    start_time = time_time()
    check_count = 0
    last_progress_print = 0
    progress_print_interval = 30

    # Random initial delay to avoid multiple clients requesting simultaneously
    time.sleep(random.uniform(1, 5))

    while time_time() - start_time < timeout_seconds:
        check_count += 1
        current_time = time_time()

        try:
            # Check if port is open
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5)
                if sock.connect_ex((host, api_port)) != 0:
                    if current_time - last_progress_print >= progress_print_interval:
                        print(f'🔌 Check {check_count}: API port {api_port} not ready')
                        last_progress_print = current_time
                    time.sleep(10)
                    continue

            # Verify service functionality
            if verify_service_functionality(host, api_port, model_name, check_count):
                print('✅ LMDeploy API Server is fully ready!')
                return True

        except Exception as e:
            if current_time - last_progress_print >= progress_print_interval:
                print(f'🔧 Check {check_count}: Exception - {e}')
                last_progress_print = current_time

        sleep_time = 10 + random.uniform(-2, 2)
        time.sleep(sleep_time)

    print(f'❌ LMDeploy API Server startup timed out ({timeout_seconds} seconds)')
    return False


def verify_service_functionality(host: str, api_port: int, model_name: str, check_count: int) -> bool:
    """Verify that the API Server can respond to basic requests."""
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

        resp = requests.post(f'http://{host}:{api_port}/v1/chat/completions', json=test_data, timeout=15)

        if resp.status_code == 200:
            print(f'✅ Check {check_count}: Service functionality normal (received valid response)')
            return True
        elif resp.status_code == 400:
            print(f'✅ Check {check_count}: Service framework activated (received 400)')
            return True
        else:
            print(f'🔧 Check {check_count}: Service test failed, status code: {resp.status_code}')
            return False

    except requests.exceptions.RequestException as e:
        print(f'🔧 Check {check_count}: Service test exception - {e}')
        return False


class RayLMDeployManager:

    def __init__(
        self,
        master_addr: str,
        ray_port: int = RAY_PORT,
        api_port: int = LM_DEPLOY_API_PORT,
        log_dir: str = '.',
        health_check: bool = True,
    ):
        self.master_addr = master_addr
        self.ray_port = ray_port
        self.api_port = api_port
        self.log_dir = log_dir
        self.health_check = health_check
        self._cleaned = False

        # Determine if this is the master node (via environment variable NODE_RANK)
        self.node_rank = int(os.getenv('NODE_RANK', '0'))
        self.is_master = (self.node_rank == 0)

        os.makedirs(self.log_dir, exist_ok=True)
        print(f'📝 Node {self.node_rank} log directory: {self.log_dir}')

        # Print cluster information
        self.node_count = int(os.getenv('NODE_COUNT', '1'))
        self.job_id = os.getenv('JOB_ID', 'unknown')
        print(f'🎯 Node {self.node_rank} cluster information:')
        print(f'  - Total nodes: {self.node_count}')
        print(f"  - Role: {'Master node' if self.is_master else 'Worker node'}")
        print(f'  - Master address: {self.master_addr}')
        print(f'  - Ray port: {self.ray_port}')
        print(f'  - API port: {self.api_port}')
        print(f'  - Job ID: {self.job_id}')

    def start_ray_cluster(self):
        """Start or join Ray cluster."""
        if self.is_master:
            cmd = ['ray', 'start', '--head', '--port', str(self.ray_port)]
            print(f'🚀 Master node starting Ray cluster (Port: {self.ray_port})')
        else:
            cmd = ['ray', 'start', '--address', f'{self.master_addr}:{self.ray_port}']
            print(f'🔌 Worker node {self.node_rank} joining Ray cluster: {self.master_addr}:{self.ray_port}')

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print('✅ Ray started successfully')
        except subprocess.CalledProcessError as e:
            print(f'💥 Ray startup failed: {e.stderr}')
            raise

    def start_lmdeploy_api_server(self, model_path: str, model_param: dict):
        """
        Master node: Start LMDeploy API Server and wait for it to be ready.
        Worker nodes: Do not start the service, only verify that the master node's API Server is ready.
        """
        if self.is_master:
            # === Master node logic: Start service ===
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            log_path = os.path.join(self.log_dir, f'lmdeploy_api_{timestamp}.log')
            tp = model_param.get('tp_num', 1)
            backend = model_param.get('backend', 'turbomind')
            communicator = model_param.get('communicator', 'nccl')
            quant_policy = model_param.get('quant_policy', 0)

            with open(log_path, 'w') as log_file:
                cmd = [
                    'lmdeploy', 'serve', 'api_server', model_path, '--server-port',
                    str(self.api_port), '--tp',
                    str(tp), '--backend', backend
                ]

                if quant_policy != 0:
                    cmd += ['--quant-policy', str(self.quant_policy)]

                if backend == 'turbomind':
                    cmd.extend(['--communicator', str(communicator)])

                print(f"🚀 Master node starting LMDeploy API Server: {' '.join(cmd)}")
                self._api_process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            print(f'📝 API Server log: {log_path}')

            # Wait for service to be ready
            if self.health_check:
                ready = wait_for_model_service_ready(host=self.master_addr,
                                                     api_port=self.api_port,
                                                     model_name=model_path,
                                                     timeout_seconds=1000)
                if not ready:
                    print('❌ API Server failed to be ready, terminating process')
                    self._api_process.terminate()
                    try:
                        self._api_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._api_process.kill()
                    raise RuntimeError('LMDeploy API Server failed to start')
        else:
            # === Worker node logic: Only verify that the master node service is ready ===
            print(f'🔍 Worker node {self.node_rank} is verifying that the master node '
                  f'({self.master_addr}:{self.api_port}) API Server is ready...')
            if self.health_check:
                ready = wait_for_model_service_ready(host=self.master_addr,
                                                     api_port=self.api_port,
                                                     model_name=model_path,
                                                     timeout_seconds=1000)
                if not ready:
                    raise RuntimeError(f'Worker node {self.node_rank}: Master node API Server not ready '
                                       f'within 1000 seconds, cannot continue')
            else:
                print('⚠️ health_check=False, skipping API Server readiness check (not recommended)')

    def cleanup(self, force: bool = True):
        """Clean up resources.

        Args:
            force (bool):
                - False: Only stop LMDeploy API Server (used after individual test completion)
                - True: Stop API Server + Ray cluster (used for final cleanup at session end)
        """
        if self._cleaned and force:
            # Note: If this is just an intermediate cleanup with force=False, we shouldn't skip due to _cleaned
            # So only skip when force=True and already cleaned
            return

        print(f'🧹 Node {self.node_rank} cleaning resources... (force={force})')

        # Stop API Server (master node only)
        if hasattr(self, '_api_process') and self._api_process.poll() is None:
            self._api_process.terminate()
            try:
                self._api_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._api_process.kill()
            print('✅ LMDeploy API Server stopped')
            # Note: We don't clear the _api_process attribute here so it can be checked later

        # Stop Ray (only when force=True)
        if force:
            try:
                subprocess.run(['ray', 'stop', '--force'], check=False, capture_output=True)
                print('✅ Ray cluster stopped')
            except Exception as e:
                print(f'⚠️ Ray stop exception: {e}')
            self._cleaned = True  # Only mark as "fully cleaned" when force=True

    def get_cluster_info(self) -> Dict[str, Any]:
        return {
            'node_rank': self.node_rank,
            'node_count': self.node_count,
            'master_addr': self.master_addr,
            'ray_port': self.ray_port,
            'api_port': self.api_port,
            'is_master': self.is_master,
            'job_id': self.job_id,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def ray_worker_node_wait(manager: RayLMDeployManager, timeout_minutes: int = 60):
    """Worker node waits for Ray master node (Head Node) to be alive (by
    detecting GCS service port)"""
    if manager.is_master:
        return

    print(f'⏸️ Worker node {manager.node_rank} entering wait mode...')
    max_checks = (timeout_minutes * 60) // WORKER_WAIT_INTERVAL
    consecutive_failures = 0
    max_consecutive_failures = 3

    for i in range(max_checks):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(CONNECTION_CHECK_TIMEOUT)
                if sock.connect_ex((manager.master_addr, RAY_PORT)) == 0:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
        except Exception:
            consecutive_failures += 1

        if consecutive_failures >= max_consecutive_failures:
            print('📡 Ray master node GCS service unreachable, worker node exiting')
            break

        if i % 4 == 0:
            elapsed = (i * WORKER_WAIT_INTERVAL) // 60
            print(f'⏳ Worker node {manager.node_rank} waiting... Running for {elapsed} minutes')

        time.sleep(WORKER_WAIT_INTERVAL)
    else:
        print(f'⏰ Worker node {manager.node_rank} wait timeout ({timeout_minutes} minutes)')

    manager.cleanup()

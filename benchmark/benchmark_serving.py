import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import fire
import yaml


def get_launching_server_cmd(model_path, backend, server_config):
    if backend in ['turbomind', 'pytorch']:
        cmd = ['lmdeploy', 'serve', 'api_server', model_path, '--backend', backend]
    elif backend == 'sglang':
        cmd = ['python3', '-m', 'sglang.launch_server', '--model-path', model_path]
    elif backend == 'vllm':
        cmd = ['vllm', 'serve', model_path]
    else:
        raise ValueError(f'unknown backend: {backend}')
    for key, value in server_config.items():
        # Convert snake_case to kebab-case for command line args
        key = key.replace('_', '-')
        cmd.append(f'--{key}')
        if str(value):
            cmd.append(str(value))
    # Special handling for proxy server case
    if server_config.get('proxy_url') and server_config.get('dp'):
        cmd.append('--allow-terminate-by-client')
    return cmd


def get_output_file(model_path, backend, server_config):
    """Generate the benchmark output filename."""
    model_name = server_config.get('model_name', None) or os.path.basename(model_path)

    if backend not in ['turbomind', 'pytorch', 'sglang', 'vllm']:
        raise ValueError(f'Unknown backend: {backend}')

    if backend in ['sglang', 'vllm']:
        return f'benchmark_{model_name}_{backend}.csv'

    # For turbomind/pytorch backends
    params = [
        ('bs', server_config['max_batch_size']),
        ('tp', server_config.get('tp', 1)),
        ('cp', server_config.get('cp', 1)),
        ('cache', server_config.get('cache_max_entry_count', 0.8)),
        ('mptk', server_config.get('max_prefill_token_num', '')),
    ]
    params_str = '_'.join(f'{k}{v}' for k, v in params if v != '')
    # Turbomind-specific additions
    if backend == 'turbomind' and (comm := server_config.get('communicator')):
        params_str += f'_{comm}'

    return f'benchmark_{model_name}_{backend}_{params_str}.csv'


def get_server_ip_port(backend: str, server_config: Dict) -> Tuple[str, int]:
    if backend in ['turbomind', 'pytorch']:
        server_ip = server_config.get('server_ip', '0.0.0.0')
        server_port = server_config.get('server_port', 23333)
    elif backend == 'sglang':
        return (server_config.get('server_ip', '0.0.0.0'), server_config.get('port', 30000))
    elif backend == 'vllm':
        return (server_config.get('server_ip', '0.0.0.0'), server_config.get('port', 8000))
    else:
        raise ValueError(f'unknown backend: {backend}')
    return server_ip, server_port


def get_served_model_name(server_ip: str, server_port: int) -> bool:
    """Wait for the API server to become ready."""
    from openai import OpenAI
    while True:
        try:
            client = OpenAI(api_key='DUMMPY', base_url=f'http://{server_ip}:{server_port}/v1')
            model_name = client.models.list().data[0].id
            if model_name:
                print('Server is ready.')
                return model_name
        except Exception as e:
            print(f'connect to server http://{server_ip}:{server_port} failed {e}')
            time.sleep(5)


def get_client_cmd(backend: str, server_ip: str, server_port: int, client_config: Dict) -> List[str]:
    """Generate the client benchmark command."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if backend in ['turbomind', 'pytorch']:
        backend = 'lmdeploy'
    cmd = [
        'python3', f'{current_dir}/profile_restful_api.py', '--backend', backend, '--host', server_ip, '--port',
        str(server_port)
    ]
    for key, value in client_config.items():
        # change the key like 'dataset_path' to 'dataset-path' to suit the optional when performing
        # "python3 benchmark/profile_restful_api.py"
        key = key.replace('_', '-')
        if key == 'disable-warmup':
            if str(value).lower() == 'true':
                cmd.append(f'--{key}')
            continue
        cmd.append(f'--{key}')
        cmd.append(str(value))
    return cmd


def benchmark(model_path: str, backend: str, server_config: Dict, data_config: Dict | List[Dict]):
    """Benchmark the server with the given configuration.

    Args:
        model_path: Path to the model.
        backend: Backend to use.
        server_config: Configuration for the server and the inference engine.
        data_config: Configuration for the data.
    """
    if isinstance(data_config, Dict):
        data_config = [data_config]
    if not (isinstance(data_config, List) and all(isinstance(d, Dict) for d in data_config)):
        raise ValueError('data_config must be a dict or list of dicts')

    server_cmd = get_launching_server_cmd(model_path, backend, server_config)
    server_ip, server_port = get_server_ip_port(backend, server_config)
    proc = None

    try:

        print(f"Starting api_server: {' '.join(server_cmd)}", flush=True)
        proc = subprocess.Popen(server_cmd)
        # Wait for the server to be ready
        get_served_model_name(server_ip, server_port)
        # Run benchmarks
        output_file = get_output_file(model_path, backend, server_config)
        for data in data_config:
            data = data.copy()
            data['output_file'] = output_file
            client_cmd = get_client_cmd(backend, server_ip, server_port, data)
            print(f"Running benchmark: {' '.join(client_cmd)}")
            subprocess.run(client_cmd, check=True)
    except Exception as e:
        print(f'Unexpected error: {e}')
        raise
    finally:
        # Clean up server process
        if proc and proc.poll() is None:
            if server_config.get('proxy_url') and server_config.get('dp'):
                # Sending termination request to proxy_server. The request will be broadcasted to
                # api_server on each dp_rank by proxy server
                # Note that api_server is supposed to be launched with --allow-terminate-by-client
                print('Sending termination request to proxy server')
                subprocess.run(['curl', '-X', 'POST', f'{server_config["proxy_url"]}/nodes/terminate_all'],
                               check=True,
                               timeout=10)
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print('Server did not terminate gracefully - killing')
                proc.kill()


def benchmark_proxy(backend: str, server_config: Dict, data_config: Dict | List[Dict]):
    server_ip = server_config.get('server_ip', '0.0.0.0')
    server_port = server_config.get('server_port', 8000)

    if isinstance(data_config, Dict):
        data_config = [data_config]
    if not (isinstance(data_config, List) and all(isinstance(d, Dict) for d in data_config)):
        raise ValueError('data_config must be a dict or list of dicts')

    try:
        # Wait for the proxy_server to be ready
        model_name = get_served_model_name(server_ip, server_port)
        # Run benchmarks
        output_file = f'benchmark_proxy_{backend}.csv'
        for data in data_config:
            data = data.copy()
            data['output_file'] = output_file
            client_cmd = get_client_cmd(backend, server_ip, server_port, data)
            print(f"Running benchmark: {' '.join(client_cmd)}")
            subprocess.run(client_cmd, check=True)
    except Exception as e:
        print(f'Unexpected error: {e}')
        raise


def main(backend: str, config_path: str, model_path: Optional[str] = None):
    """Main entry point for the benchmark script.

    Args:
        backend: Backend to use
        config_path: Path to config file
        model_path: Optional override for model path
    Raises:
        BenchmarkConfigError: If required parameters are missing or config is invalid
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        data_config = config['data']
        server_config = config['server']
        server_type = server_config.get('type', 'api_server')
        if server_type == 'proxy':
            benchmark_proxy(backend, server_config, data_config)
            return
        
        engine_configs = config['engine']
        if isinstance(engine_configs, Dict):
            engine_configs = [engine_configs]
        assert isinstance(engine_configs, List) and all(isinstance(s, Dict) for s in engine_configs)
        for engine_config in engine_configs:
            server_config = server_config.copy()
            server_config.update(engine_config)  # Merge engine config with server config
            # The model_path provided by the user will override the model_path in the config file.
            model_path = model_path or server_config.pop('model_path')
            # Remove model_path from server_config to avoid passing it to the server command
            server_config.pop('model_path', None)
            benchmark(model_path, backend, server_config, data_config)


if __name__ == '__main__':
    fire.Fire(main)

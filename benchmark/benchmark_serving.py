import os
import subprocess
import time
from typing import Dict, List

import fire
import yaml


def get_launching_server_cmd(model_path, backend, server_config):
    if backend in ['turbomind', 'pytorch']:
        cmd = ['lmdeploy', 'serve', 'api_server', model_path, '--backend', backend]
    elif backend == 'sglang':
        pass
    elif backend == 'vllm':
        pass
    else:
        print(f'Unknown backend: {backend}')
        return
    for key, value in server_config.items():
        # change the key like 'cache_max_entry_count' to 'cache-max-entry-count' to suit the optional
        # arguments when performing "lmdeploy serve api_server"
        key = key.replace('_', '-')
        cmd.append(f'--{key}')
        cmd.append(str(value))
    return cmd


def get_server_ip_port(backend, server_config):
    if backend in ['turbomind', 'pytorch']:
        server_ip = server_config.get('server_ip', '0.0.0.0')
        server_port = server_config.get('server_port', 23333)
    elif backend == 'sglang':
        pass
    elif backend == 'vllm':
        pass
    else:
        print(f'Unknown backend: {backend}')
        return None, None
    return server_ip, server_port


def wait_server_ready(server_ip, server_port):
    import time

    from openai import OpenAI
    while True:
        try:
            client = OpenAI(api_key='DUMMPY', base_url=f'http://{server_ip}:{server_port}/v1')
            model_name = client.models.list().data[0].id
            if model_name:
                print('Server is ready.')
                return True
        except Exception as e:
            print(f'connect to server http://{server_ip}:{server_port} failed {e}')
            time.sleep(5)


def get_client_cmd(backend, server_ip, server_port, client_config):
    if backend in ['turbomind', 'pytorch']:
        cmd = [
            'python3', 'benchmark/profile_restful_api.py', '--backend', 'lmdeploy', '--host', server_ip, '--port',
            str(server_port)
        ]
    elif backend == 'sglang':
        pass
    elif backend == 'vllm':
        pass
    else:
        print(f'Unknown backend: {backend}')
        return
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


def benchmark(model_path, backend, server_config, data_config):
    """Benchmark the server with the given configuration.

    :param model_path: Path to the model.
    :param backend: Backend to use.
    :param server_config: Configuration for the server and the inference engine.
    :param data_config: Configuration for the data.
    """
    model_name = os.path.basename(model_path)
    max_batch_size = server_config['max_batch_size']
    cache_max_entry_count = server_config.get('cache_max_entry_count', 0.8)
    tp = server_config.get('tp', 1)
    output_file = f'benchmark_{model_name}_{backend}_bs{max_batch_size}_tp{tp}_cache{cache_max_entry_count}.csv'
    server_cmd = get_launching_server_cmd(model_path, backend, server_config)
    server_ip, server_port = get_server_ip_port(backend, server_config)
    try:
        print(f"Running server command: {' '.join(server_cmd)}")
        proc = subprocess.Popen(server_cmd)
        wait_server_ready(server_ip, server_port)
        # Run the benchmark script
        if isinstance(data_config, Dict):
            data_config = [data_config]
        assert isinstance(data_config, List) and all(isinstance(d, Dict) for d in data_config)
        for data in data_config:
            data['output_file'] = output_file
            client_cmd = get_client_cmd(backend, server_ip, server_port, data)
            print(f"Running client command: {' '.join(client_cmd)}")
            subprocess.run(client_cmd, check=True)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            time.sleep(30)


def main(model_path=None, backend=None, config_path=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        server_config = config['server']
        engine_configs = config['engine']
        data_config = config['data']
        if isinstance(engine_configs, Dict):
            engine_configs = [engine_configs]
        assert isinstance(engine_configs, List) and all(isinstance(s, Dict) for s in engine_configs)
        for engine_config in engine_configs:
            engine_config.update(server_config)  # Merge engine config with server config
            # The model_path provided by the user will override the model_path in the config file.
            model_path = model_path or engine_config.pop('model_path')
            engine_config.pop('model_path', '')
            benchmark(model_path, backend, engine_config, data_config)


if __name__ == '__main__':
    fire.Fire(main)

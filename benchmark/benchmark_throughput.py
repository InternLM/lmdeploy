import os
import subprocess
import time
from typing import Dict, List

import fire
import yaml


def get_cmd(model_path, backend, engine_config, data_config):
    assert backend in ['turbomind', 'pytorch']

    dataset_path = data_config.pop('dataset_path')

    cmd = ['python3', 'benchmark/profile_throughput.py', dataset_path, model_path, '--backend', backend]
    for key, value in engine_config.items():
        # profile_throughput.py uses "--concurrency" to pass the "max_batch_size" value
        if key == 'max_batch_size':
            key = 'concurrency'
        # change the key like 'cache_max_entry_count' to 'cache-max-entry-count' to suit the optional
        # arguments in "python3 benchmark/profile_throughput.py"
        key = key.replace('_', '-')
        cmd.append(f'--{key}')
        cmd.append(str(value))

    for key, value in data_config.items():
        # change the key like 'sharegpt_output_len' to 'sharegpt-output-len' to suit the optional
        # arguments in "python3 benchmark/profile_throughput.py"
        key = key.replace('_', '-')
        cmd.append(f'--{key}')
        cmd.append(str(value))
    return cmd


def benchmark(model_path, backend, engine_config, data_config):
    """Benchmark the performance with the given configuration.

    Args:
        model_path: Path to the model.
    :param backend: Backend to use.
    :param engine_config: Configuration for the inference engine.
    :param data_config: Configuration for the data.
    """
    model_name = os.path.basename(model_path)
    bs = engine_config['max_batch_size']
    cach_ratio = engine_config.get('cache_max_entry_count', 0.8)
    tp = engine_config.get('tp', 1)
    output_file = f'benchmark_throughput_{model_name}_{backend}_bs{bs}_tp{tp}_cache{cach_ratio}.csv'
    try:
        if isinstance(data_config, Dict):
            data_config = [data_config]
        assert isinstance(data_config, List) and all(isinstance(d, Dict) for d in data_config)
        for _data_config in data_config:
            _data_config['csv'] = output_file
            cmd = get_cmd(model_path, backend, engine_config, _data_config)
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
    finally:
        time.sleep(10)


def main(model_path=None, backend=None, config_path=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        engine_configs = config['engine']
        data_config = config['data']
        if isinstance(engine_configs, Dict):
            engine_configs = [engine_configs]
        assert isinstance(engine_configs, List) and all(isinstance(s, Dict) for s in engine_configs)
        for engine_config in engine_configs:
            # The model_path provided by the user will override the model_path in the config file.
            model_path = model_path or engine_config.pop('model_path')
            engine_config.pop('model_path', '')
            benchmark(model_path, backend, engine_config, data_config)


if __name__ == '__main__':
    fire.Fire(main)

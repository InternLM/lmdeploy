import csv
import os
import re
import subprocess
import time
import urllib.request

import fire
import yaml

SPEC_COUNTER_NAMES = {
    'drafts': 'lmdeploy:spec_decode_num_drafts_total',
    'draft_tokens': 'lmdeploy:spec_decode_num_draft_tokens_total',
    'accepted_tokens': 'lmdeploy:spec_decode_num_accepted_tokens_total',
    'accepted_tokens_per_pos': 'lmdeploy:spec_decode_num_accepted_tokens_per_pos_total',
}

def _is_spec_decoding(server_config: dict) -> bool:
    """Return whether this engine config enables speculative decoding."""
    return (server_config.get('speculative_algorithm') is not None
            or server_config.get('speculative_num_draft_tokens') is not None
            or server_config.get('speculative_draft_model') is not None)


def _parse_prometheus_labels(labels_text: str | None) -> dict[str, str]:
    """Parse the label section of one Prometheus exposition sample."""
    if not labels_text:
        return {}
    labels = {}
    for match in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"', labels_text):
        labels[match.group(1)] = bytes(match.group(2), 'utf-8').decode('unicode_escape')
    return labels


def _parse_prometheus_metrics(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Parse numeric samples from Prometheus exposition text."""
    metrics = {}
    sample_re = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)(?:\s|$)')
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        match = sample_re.match(line)
        if not match:
            continue
        name = match.group(1)
        # Ignore Prometheus client creation timestamps.
        if name.endswith('_created'):
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        labels = tuple(sorted(_parse_prometheus_labels(match.group(2)).items()))
        metrics[(name, labels)] = value
    return metrics


def _scrape_prometheus_metrics(server_ip: str,
                               server_port: int) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Scrape the API server /metrics endpoint."""
    metrics_ip = '127.0.0.1' if server_ip in {'0.0.0.0', '::'} else server_ip
    url = f'http://{metrics_ip}:{server_port}/metrics'
    with urllib.request.urlopen(url, timeout=5) as response:
        text = response.read().decode('utf-8', errors='replace')
    return _parse_prometheus_metrics(text)


def _metric_sum(metrics: dict[tuple[str, tuple[tuple[str, str], ...]], float],
                metric_name: str,
                position: str | None = None) -> float:
    total = 0.0
    for (name, labels_tuple), value in metrics.items():
        if name != metric_name:
            continue
        labels = dict(labels_tuple)
        if position is not None and labels.get('position') != position:
            continue
        total += value
    return total


def _metric_positions(metrics: dict[tuple[str, tuple[tuple[str, str], ...]], float], metric_name: str) -> set[str]:
    positions = set()
    for (name, labels_tuple) in metrics:
        if name != metric_name:
            continue
        labels = dict(labels_tuple)
        if 'position' in labels:
            positions.add(labels['position'])
    return positions


def _build_specdecode_summary(before: dict[tuple[str, tuple[tuple[str, str], ...]], float],
                              after: dict[tuple[str, tuple[tuple[str, str], ...]], float]) -> dict[str, str]:
    """Build a per-client-run speculative decoding summary from Prometheus
    counter deltas."""
    num_drafts = _metric_sum(after, SPEC_COUNTER_NAMES['drafts']) - _metric_sum(before, SPEC_COUNTER_NAMES['drafts'])
    draft_tokens = _metric_sum(after, SPEC_COUNTER_NAMES['draft_tokens']) - _metric_sum(
        before, SPEC_COUNTER_NAMES['draft_tokens'])
    accepted_tokens = _metric_sum(after, SPEC_COUNTER_NAMES['accepted_tokens']) - _metric_sum(
        before, SPEC_COUNTER_NAMES['accepted_tokens'])
    accept_rate = accepted_tokens / draft_tokens if draft_tokens > 0 else float('nan')
    mean_accept_length = 1 + accepted_tokens / num_drafts if num_drafts > 0 else float('nan')

    positions = sorted(
        _metric_positions(before, SPEC_COUNTER_NAMES['accepted_tokens_per_pos'])
        | _metric_positions(after, SPEC_COUNTER_NAMES['accepted_tokens_per_pos']),
        key=lambda item: int(item) if item.isdigit() else item)
    per_position = []
    for pos in positions:
        accepted_pos = _metric_sum(after, SPEC_COUNTER_NAMES['accepted_tokens_per_pos'], pos) - _metric_sum(
            before, SPEC_COUNTER_NAMES['accepted_tokens_per_pos'], pos)
        rate_pos = accepted_pos / num_drafts if num_drafts > 0 else float('nan')
        per_position.append(f'{pos}:{rate_pos:.6g}')

    return {
        'spec_num_drafts': f'{num_drafts:.0f}',
        'spec_num_draft_tokens': f'{draft_tokens:.0f}',
        'spec_num_accepted_tokens': f'{accepted_tokens:.0f}',
        'spec_draft_acceptance_rate': f'{accept_rate:.8g}',
        'spec_mean_acceptance_length': f'{mean_accept_length:.8g}',
        'spec_per_position_acceptance_rate': ';'.join(per_position),
    }


def _build_run_metadata(server_config: dict) -> dict[str, str]:
    """Build lightweight columns that identify the benchmark engine config."""
    spec_method = server_config.get('speculative_algorithm') or 'none'
    num_spec = server_config.get('speculative_num_draft_tokens')
    if num_spec is None:
        num_spec = 0
    return {
        'spec_method': str(spec_method),
        'num_spec': str(num_spec),
    }


def _enrich_output_csv(output_file: str, extra_columns: dict[str, str]) -> None:
    """Append lightweight engine-identification columns to the latest benchmark
    CSV row."""
    if not os.path.exists(output_file):
        return
    with open(output_file, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        return
    for field in extra_columns:
        if field not in fieldnames:
            fieldnames.append(field)
    rows[-1].update(extra_columns)
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_specdecode_summary_csv(model_path: str,
                                   backend: str,
                                   server_config: dict,
                                   data_config: dict,
                                   output_file: str,
                                   spec_summary: dict[str, str]) -> None:
    """Append a compact per-case speculative decoding summary CSV."""
    summary_file = 'specdecode_metrics_summary.csv'
    row = {
        'backend': backend,
        'model_name': server_config.get('model_name', os.path.basename(model_path)),
        'output_file': output_file,
        'dataset_name': data_config.get('dataset_name', ''),
        'num_prompts': data_config.get('num_prompts', ''),
        'random_input_len': data_config.get('random_input_len', ''),
        'random_output_len': data_config.get('random_output_len', ''),
        'sharegpt_output_len': data_config.get('sharegpt_output_len', ''),
        'request_rate': data_config.get('request_rate', ''),
    }
    row.update(_build_run_metadata(server_config))
    row.update(spec_summary)
    fieldnames = list(row)
    exists = os.path.exists(summary_file)
    with open(summary_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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
        ('dp', server_config.get('dp', '')),
        ('ep', server_config.get('ep', '')),
        ('cache', server_config.get('cache_max_entry_count', 0.8)),
        ('mptk', server_config.get('max_prefill_token_num', '')),
    ]
    params_str = '_'.join(f'{k}{v}' for k, v in params if v != '')
    # Turbomind-specific additions
    if backend == 'turbomind' and (comm := server_config.get('communicator')):
        params_str += f'_{comm}'

    return f'benchmark_{model_name}_{backend}_{params_str}.csv'


def get_server_ip_port(backend: str, server_config: dict) -> tuple[str, int]:
    if backend in ['turbomind', 'pytorch']:
        if server_config.get('proxy_url'):
            # If proxy_url is set, we use the proxy server's IP and port
            parts = server_config['proxy_url'].split(':')
            server_ip = parts[1].lstrip('//')
            server_port = int(parts[2])
        else:
            # Default to the server IP and port specified in the config
            server_ip = server_config.get('server_ip', '0.0.0.0')
            server_port = server_config.get('server_port', 23333)
    elif backend == 'sglang':
        return (server_config.get('server_ip', '0.0.0.0'), server_config.get('port', 30000))
    elif backend == 'vllm':
        return (server_config.get('server_ip', '0.0.0.0'), server_config.get('port', 8000))
    else:
        raise ValueError(f'unknown backend: {backend}')
    return server_ip, server_port


def wait_server_ready(server_ip: str, server_port: int, proc: subprocess.Popen | None = None) -> bool:
    """Wait for the API server to become ready."""
    from openai import OpenAI
    while True:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f'API server exited before becoming ready, returncode={proc.returncode}')
        try:
            client = OpenAI(api_key='DUMMPY', base_url=f'http://{server_ip}:{server_port}/v1')
            model_name = client.models.list().data[0].id
            if model_name:
                print('Server is ready.')
                return True
        except Exception as e:
            print(f'connect to server http://{server_ip}:{server_port} failed {e}')
            time.sleep(5)


def get_client_cmd(backend: str, server_ip: str, server_port: int, client_config: dict) -> list[str]:
    """Generate the client benchmark command."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    client_config = client_config.copy()
    client_backend = client_config.pop('client_backend', None) or client_config.pop('backend', None)
    if client_backend is None:
        client_backend = 'lmdeploy' if backend in ['turbomind', 'pytorch'] else backend
    cmd = [
        'python3', f'{current_dir}/profile_restful_api.py', '--backend', client_backend, '--host', server_ip, '--port',
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


def benchmark(model_path: str, backend: str, server_config: dict, data_config: dict | list[dict]):
    """Benchmark the server with the given configuration.

    Args:
        model_path: Path to the model.
        backend: Backend to use.
        server_config: Configuration for the server and the inference engine.
        data_config: Configuration for the data.
    """
    if isinstance(data_config, dict):
        data_config = [data_config]
    if not (isinstance(data_config, list) and all(isinstance(d, dict) for d in data_config)):
        raise ValueError('data_config must be a dict or list of dicts')

    server_cmd = get_launching_server_cmd(model_path, backend, server_config)
    server_ip, server_port = get_server_ip_port(backend, server_config)
    proc = None

    try:

        print(f"Starting api_server: {' '.join(server_cmd)}", flush=True)
        proc = subprocess.Popen(server_cmd)
        # Wait for the server to be ready
        wait_server_ready(server_ip, server_port, proc)
        # Run benchmarks
        output_file = get_output_file(model_path, backend, server_config)
        for data in data_config:
            data = data.copy()
            collect_spec_metrics = _is_spec_decoding(server_config)
            data['output_file'] = output_file
            client_cmd = get_client_cmd(backend, server_ip, server_port, data)
            print(f"Running benchmark: {' '.join(client_cmd)}")
            spec_metrics_before = _scrape_prometheus_metrics(server_ip, server_port) if collect_spec_metrics else None
            subprocess.run(client_cmd, check=True)
            _enrich_output_csv(output_file, _build_run_metadata(server_config))
            if spec_metrics_before is not None:
                try:
                    spec_metrics_after = _scrape_prometheus_metrics(server_ip, server_port)
                    spec_summary = _build_specdecode_summary(spec_metrics_before, spec_metrics_after)
                    _append_specdecode_summary_csv(model_path, backend, server_config, data, output_file, spec_summary)
                    print('Recorded specdecode metrics: '
                          f"accept_rate={spec_summary['spec_draft_acceptance_rate']}, "
                          f"mean_accept_len={spec_summary['spec_mean_acceptance_length']}")
                except Exception as e:
                    print(f'Warning: failed to record specdecode metrics after benchmark: {e}')
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


def validate_config(config: dict) -> None:
    """Validate the configuration structure.

    Args:
        config: Loaded configuration dictionary

    Raises:
        BenchmarkConfigError: If configuration is invalid
    """
    required_sections = ['api_server', 'engine', 'data']
    for section in required_sections:
        if section not in config:
            raise ValueError(f'Missing required config section: {section}')

    if not isinstance(config['engine'], (dict, list)):
        raise ValueError('engine config must be a dict or list of dicts')

    if not isinstance(config['data'], (dict, list)):
        raise ValueError('data config must be a dict or list of dicts')


def main(backend: str, config_path: str, model_path: str | None = None):
    """Main entry point for the benchmark script.

    Args:
        backend: Backend to use
        config_path: Path to config file
        model_path: Optional override for model path
    Raises:
        BenchmarkConfigError: If required parameters are missing or config is invalid
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
        base_server_config = config['server']
        engine_configs = config['engine']
        data_config = config['data']
        if isinstance(engine_configs, dict):
            engine_configs = [engine_configs]
        assert isinstance(engine_configs, list) and all(isinstance(s, dict) for s in engine_configs)
        user_model_path = model_path
        for engine_config in engine_configs:
            server_config = base_server_config.copy()
            server_config.update(engine_config)  # Merge engine config with server config
            # The model_path provided by the user will override the model_path in the config file.
            run_model_path = user_model_path or server_config.pop('model_path')
            # Remove model_path from server_config to avoid passing it to the server command
            server_config.pop('model_path', None)
            benchmark(run_model_path, backend, server_config, data_config)


if __name__ == '__main__':
    fire.Fire(main)

import os
import time

import allure
import utils.constant as constant
from utils.common_utils import execute_command_with_logging
from utils.config_utils import get_case_str_by_config, get_cli_common_param, get_cuda_prefix_by_workerid, get_workerid
from utils.run_restful_chat import health_check, start_openai_service, terminate_restful_api


def throughput_test(config, run_config, worker_id: str = '', is_smoke: bool = False):
    model = run_config.get('model')
    model_path = os.path.join(config.get('model_path'), model)
    dataset_path = config.get('dataset_path')

    case_name = get_case_str_by_config(run_config)
    benchmark_path = os.path.join(config.get('benchmark_path'), 'throughput')
    work_dir = os.path.join(benchmark_path, f'wk_{case_name}')
    os.makedirs(work_dir, exist_ok=True)

    max_cache_entry = get_max_cache_entry(model, run_config.get('backend'))
    if max_cache_entry is not None:
        if 'extra_params' not in run_config:
            run_config['extra_params'] = {}
        run_config['extra_params']['cache-max-entry-count'] = max_cache_entry

    cuda_prefix = get_cuda_prefix_by_workerid(worker_id, run_config.get('parallel_config'))

    command = f'{cuda_prefix} python3 benchmark/profile_throughput.py {dataset_path} {model_path} {get_cli_common_param(run_config)}'  # noqa

    if is_smoke:
        num_prompts = '--num-prompts 100'
    else:
        num_prompts = '--num-prompts 5000'

    env = os.environ.copy()
    env.update(run_config.get('env', {}))

    for batch in [128, 256]:
        csv_path = os.path.join(work_dir, f'{batch}.csv')
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        benchmark_log = os.path.join(benchmark_path, f'log_{case_name}_{batch}_{timestamp}.log')
        cmd = ' '.join([command, '--concurrency', str(batch), num_prompts, '--csv ', csv_path]).strip()

        result, stderr = execute_command_with_logging(cmd, benchmark_log, env=env)
        allure.attach.file(benchmark_log, name=benchmark_log, attachment_type=allure.attachment_type.TEXT)

        if result and not os.path.isfile(csv_path):
            return False, 'result is empty'
        if not result:
            return False, stderr

    return True, 'success'


def longtext_throughput_test(config, run_config, worker_id: str = ''):
    model = run_config.get('model')
    model_path = os.path.join(config.get('model_path'), model)
    dataset_path = config.get('dataset_path')

    case_name = get_case_str_by_config(run_config)
    benchmark_path = os.path.join(config.get('benchmark_path'), 'longtext-throughput')
    work_dir = os.path.join(benchmark_path, f'wk_{case_name}')
    os.makedirs(work_dir, exist_ok=True)

    max_cache_entry = get_max_cache_entry(model, run_config.get('backend'))
    if max_cache_entry is not None:
        if 'extra_params' not in run_config:
            run_config['extra_params'] = {}
        run_config['extra_params']['cache-max-entry-count'] = max_cache_entry
        run_config['extra_params'].pop('session-len', None)

    cuda_prefix = get_cuda_prefix_by_workerid(worker_id, run_config.get('parallel_config'))

    command = f'{cuda_prefix} python3 benchmark/profile_pipeline_api.py {dataset_path} {model_path} {get_cli_common_param(run_config)}'  # noqa

    env = os.environ.copy()
    env.update(run_config.get('env', {}))

    for input_len, out_len, num_prompts, session_info, concurrency in [(1, 32768, 10, '32k', 10),
                                                                       (1, 65536, 5, '64k', 5),
                                                                       (65536, 1024, 15, '64k-1k', 15),
                                                                       (198000, 1024, 3, '198k-1k', 1)]:
        session_len = input_len + out_len + 1
        csv_path = os.path.join(work_dir, f'{case_name}_{session_info}.csv')
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        benchmark_log = os.path.join(benchmark_path, f'log_{case_name}_{session_info}_{timestamp}.log')
        cmd = ' '.join([
            command, '--dataset-name random', f'--random-input-len {input_len}', f'--random-output-len {out_len}',
            f'--num-prompts {num_prompts}', f'--concurrency {concurrency}', '--stream-output',
            f'--session-len {session_len}', '--random-range-ratio 1', f'--csv {csv_path}'
        ]).strip()

        result, stderr = execute_command_with_logging(cmd, benchmark_log, timeout=7200, env=env)
        allure.attach.file(benchmark_log, name=benchmark_log, attachment_type=allure.attachment_type.TEXT)

        if result and not os.path.isfile(csv_path):
            return False, 'result is empty'
        if not result:
            return False, stderr
    return True, 'success'


def restful_test(config, run_config, worker_id: str = '', is_smoke: bool = False, is_mllm: bool = False):
    max_cache_entry = get_max_cache_entry(run_config.get('model'), run_config.get('backend'))
    if max_cache_entry is not None:
        if 'extra_params' not in run_config:
            run_config['extra_params'] = {}
        run_config['extra_params']['cache-max-entry-count'] = max_cache_entry

    pid, content = start_openai_service(config, run_config, worker_id, timeout=1200)
    try:
        if pid > 0:
            if is_mllm:
                return mllm_restful_profile(config,
                                            run_config,
                                            port=constant.DEFAULT_PORT + get_workerid(worker_id),
                                            is_smoke=is_smoke)
            else:
                return restful_profile(config,
                                       run_config,
                                       port=constant.DEFAULT_PORT + get_workerid(worker_id),
                                       is_smoke=is_smoke)
        else:
            assert False, f'Failed to start RESTful API server: {content}'
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)


BASE_HTTP_URL = f'http://{constant.DEFAULT_SERVER}'


def restful_profile(config, run_config, port, is_smoke: bool = False):
    model_path = os.path.join(config.get('model_path'), run_config.get('model'))
    case_name = get_case_str_by_config(run_config)
    dataset_path = config.get('dataset_path')
    benchmark_path = os.path.join(config.get('benchmark_path'), 'restful')
    work_dir = os.path.join(benchmark_path, f'wk_{case_name}')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    benchmark_log = os.path.join(benchmark_path, f'log_{case_name}_{timestamp}.log')
    os.makedirs(work_dir, exist_ok=True)

    http_url = f'{BASE_HTTP_URL}:{port}'  # noqa: E231
    if not health_check(http_url, case_name):
        return False, 'server not start'

    csv_path = f'{work_dir}/restful.csv'

    command = f'python benchmark/profile_restful_api.py --backend lmdeploy --dataset-name sharegpt --dataset-path {dataset_path} --tokenizer {model_path} --base-url {http_url} --output-file {csv_path}'  # noqa
    if is_smoke:
        command += ' --num-prompts 100'
    else:
        command += ' --num-prompts 5000'

    result, stderr = execute_command_with_logging(command, benchmark_log)
    allure.attach.file(benchmark_log, name=benchmark_log, attachment_type=allure.attachment_type.TEXT)

    if result and not os.path.isfile(csv_path):
        return False, 'result is empty'
    if not result:
        return False, stderr
    return True, 'success'


def mllm_restful_profile(config, run_config, port, is_smoke: bool = False):
    model_path = os.path.join(config.get('model_path'), run_config.get('model'))
    case_name = get_case_str_by_config(run_config)
    benchmark_path = os.path.join(config.get('benchmark_path'), 'mllm_restful')
    work_dir = os.path.join(benchmark_path, f'wk_{case_name}')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    benchmark_log = os.path.join(benchmark_path, f'log_{case_name}_{timestamp}.log')
    os.makedirs(work_dir, exist_ok=True)

    http_url = f'{BASE_HTTP_URL}:{port}'  # noqa: E231
    if not health_check(http_url, case_name):
        return False, 'server not start'

    csv_path = f'{work_dir}/mllm_restful.csv'

    command = f'python benchmark/profile_restful_api.py --backend lmdeploy-chat --dataset-name image --tokenizer {model_path} --model {case_name} --model-path {model_path} --random-input-len 100 --random-output-len 100 --random-range-ratio 1 --image-format jpeg --image-count 1 --image-content random --image-resolution 1024x1024 --base-url {http_url} --output-file {csv_path}'  # noqa
    if is_smoke:
        command += ' --num-prompts 100'
    else:
        command += ' --num-prompts 5000'

    result, stderr = execute_command_with_logging(command, benchmark_log)
    allure.attach.file(benchmark_log, name=benchmark_log, attachment_type=allure.attachment_type.TEXT)

    if result and not os.path.isfile(csv_path):
        return False, 'result is empty'
    if not result:
        return False, stderr
    return True, 'success'


def prefixcache_throughput_test(config, run_config, worker_id: str = ''):
    model = run_config.get('model')
    model_path = os.path.join(config.get('model_path'), model)
    dataset_path = config.get('prefix_dataset_path')

    case_name = get_case_str_by_config(run_config)
    benchmark_path = os.path.join(config.get('benchmark_path'), 'prefix-throughtput')
    work_dir = os.path.join(benchmark_path, f'wk_{case_name}')
    os.makedirs(work_dir, exist_ok=True)
    max_cache_entry = get_max_cache_entry(model, run_config.get('backend'))
    if max_cache_entry is not None:
        if 'extra_params' not in run_config:
            run_config['extra_params'] = {}
        run_config['extra_params']['cache-max-entry-count'] = max_cache_entry

    cuda_prefix = get_cuda_prefix_by_workerid(worker_id, run_config.get('parallel_config'))

    run_config_new = run_config.copy()
    if 'extra_params' not in run_config_new:
        run_config_new['extra_params'] = {}
    run_config_new['extra_params'].pop('enable-prefix-caching', None)
    run_config_new['extra_params']['session-len'] = 32768
    command = f'{cuda_prefix} python3 benchmark/profile_pipeline_api.py {dataset_path} {model_path} {get_cli_common_param(run_config_new)}'  # noqa

    env = os.environ.copy()
    env.update(run_config.get('env', {}))

    test_configs = [(8096, 256, 500, '8k', None)]
    for enable_prefix_caching in [False, True]:
        suffix = 'cache' if enable_prefix_caching else 'no_cache'

        for input_len, out_len, num_prompts, session_info, concurrency in test_configs:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            benchmark_log = os.path.join(benchmark_path, f'log_{case_name}_{session_info}_{suffix}_{timestamp}.log')
            csv_path = os.path.join(work_dir, f'{session_info}_{suffix}.csv')

            command = ' '.join([
                command, '--dataset-name random', f'--random-input-len {input_len}', f'--random-output-len {out_len}',
                '--random-range-ratio 1.0', f'--num-prompts {num_prompts}', '--stream-output', f'--csv {csv_path}'
            ]).strip()

            if enable_prefix_caching:
                command += ' --enable-prefix-caching'

            if concurrency:
                command += f' --concurrency {concurrency}'

            result, stderr = execute_command_with_logging(command, benchmark_log, env=env)
            allure.attach.file(benchmark_log, name=benchmark_log, attachment_type=allure.attachment_type.TEXT)

            if result and not os.path.isfile(csv_path):
                return False, 'result is empty'
            if not result:
                return False, stderr
    return True, 'success'


def get_max_cache_entry(model, backend):
    if backend == 'pytorch':
        return 0.8
    if 'Llama-2' in model:
        return 0.95
    elif 'internlm2' in model:
        return 0.9
    elif 'Qwen/Qwen3-235B-A22B' == model or 'internlm/Intern-S1' == model:
        return 0.7
    else:
        return None

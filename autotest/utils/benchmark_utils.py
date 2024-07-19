import os
import subprocess
from subprocess import PIPE

from utils.config_utils import get_workerid

DEFAULT_PORT = 23333

GENERATION_CONFIG = ' -c 8 256 -ct 128 128 2048 128 -pt 1 128 128 2048'
GENERATION_LONGTEXT_CONFIG = ' -c 1 --session-len 200000 -ct 1024 -pt 198000'

# GENERATION_CONFIG = ' -c 8 -ct 128 -pt 128'
# GENERATION_LONGTEXT_CONFIG = ' -c 1 --session-len 100 -ct 128 -pt 128'


def generation_test(config,
                    run_id,
                    run_config,
                    is_longtext: bool = False,
                    cuda_prefix: str = None,
                    worker_id: str = '',
                    is_smoke: bool = False):
    model = run_config['model']
    backend = run_config['backend']
    tp_num = run_config['tp_num']
    model_path = '/'.join([config.get('model_path'), model])
    log_path = config.get('log_path')
    benchmark_log = os.path.join(
        log_path, 'benchmark_' + model.split('/')[1] + worker_id + '.log')
    benchmark_path = '/'.join([
        config.get('benchmark_path'), run_id, model,
        f'benchmark-generation-{backend}'
    ])

    create_multi_level_directory(benchmark_path)

    command = f'python3 benchmark/profile_generation.py {model_path} '
    command = get_command_with_extra(command, cuda_prefix)

    run_config = ''
    if backend == 'pytorch':
        command += ' --backend pytorch'
    else:
        if is_longtext:
            run_config = '--rope-scaling-factor 1.0'
        if '4bit' in model:
            command += ' --model-format awq'

    if is_longtext:
        run_config = run_config + GENERATION_LONGTEXT_CONFIG
        csv_path = f'{benchmark_path}/generation_longtext.csv'
    else:
        run_config = run_config + GENERATION_CONFIG
        csv_path = f'{benchmark_path}/generation.csv'
    if is_smoke:
        run_config = ' -c 1 -ct 128 -pt 128'

    cmd = ' '.join(
        [command, run_config, '--tp',
         str(tp_num), '--csv', csv_path])

    with open(benchmark_log, 'w') as f:
        f.writelines('reproduce command: ' + cmd + '\n')
        print('reproduce command: ' + cmd)

        benchmark_res = subprocess.run([cmd],
                                       stdout=f,
                                       stderr=PIPE,
                                       shell=True,
                                       text=True)
        f.writelines(benchmark_res.stderr)

    if benchmark_res.returncode == 0 and os.path.isfile(csv_path) is False:
        return False, benchmark_log, 'result is empty'
    return benchmark_res.returncode == 0, benchmark_log, benchmark_res.stderr


def throughput_test(config,
                    run_id,
                    run_config,
                    cuda_prefix: str = None,
                    worker_id: str = '',
                    is_smoke: bool = False):
    model = run_config['model']
    backend = run_config['backend']
    tp_num = run_config['tp_num']
    if backend == 'turbomind':
        quant_policy = run_config['quant_policy']
    model_path = '/'.join([config.get('model_path'), model])
    log_path = config.get('log_path')
    dataset_path = config.get('dataset_path')
    benchmark_log = os.path.join(
        log_path, 'benchmark_' + model.split('/')[1] + worker_id + '.log')
    if backend == 'turbomind' and quant_policy != 0:
        benchmark_path = '/'.join([
            config.get('benchmark_path'), run_id, model,
            f'benchmark-throughput-{backend}-kvint{quant_policy}'
        ])
    else:
        benchmark_path = '/'.join([
            config.get('benchmark_path'), run_id, model,
            f'benchmark-throughput-{backend}'
        ])

    create_multi_level_directory(benchmark_path)

    command = f'python3 benchmark/profile_throughput.py {dataset_path} {model_path} '  # noqa: F401, E501
    command = get_command_with_extra(command, cuda_prefix)

    if is_smoke:
        run_config = '--num-prompts 300'
    else:
        run_config = '--num-prompts 3000'
    if backend == 'pytorch':
        command += ' --backend pytorch'
    else:
        if '4bit' in model:
            command += ' --model-format awq'
        run_config = run_config + f' --quant-policy {quant_policy}'

    for batch in [128, 256]:
        csv_path = f'{benchmark_path}/throughput_batch_{batch}_1th.csv'
        cmd = ' '.join(
            [command, run_config, '--tp',
             str(tp_num), '--csv ', csv_path])

        with open(benchmark_log, 'w') as f:
            f.writelines('reproduce command: ' + cmd + '\n')
            print('reproduce command: ' + cmd)

            benchmark_res = subprocess.run([cmd],
                                           stdout=f,
                                           stderr=PIPE,
                                           shell=True,
                                           text=True,
                                           encoding='utf-8')
            f.writelines(benchmark_res.stderr)
    if benchmark_res.returncode == 0 and os.path.isfile(csv_path) is False:
        return False, benchmark_log, 'result is empty'
    return benchmark_res.returncode == 0, benchmark_log, benchmark_res.stderr


def restful_test(config,
                 run_id,
                 run_config,
                 worker_id: str = '',
                 is_smoke: bool = False):
    model = run_config['model']
    backend = run_config['backend']
    if backend == 'turbomind':
        quant_policy = run_config['quant_policy']
    model_path = '/'.join([config.get('model_path'), model])
    log_path = config.get('log_path')
    dataset_path = config.get('dataset_path')
    benchmark_log = os.path.join(
        log_path, 'benchmark_' + model.split('/')[1] + worker_id + '.log')
    if backend == 'turbomind' and quant_policy != 0:
        benchmark_path = '/'.join([
            config.get('benchmark_path'), run_id, model,
            f'benchmark-restful-{backend}-kvint{quant_policy}'
        ])
    else:
        benchmark_path = '/'.join([
            config.get('benchmark_path'), run_id, model,
            f'benchmark-restful-{backend}'
        ])

    create_multi_level_directory(benchmark_path)

    worker_num = get_workerid(worker_id)
    if worker_num is None:
        port = DEFAULT_PORT
    else:
        port = DEFAULT_PORT + worker_num

    command = f'python3 benchmark/profile_restful_api.py localhost:{port} {model_path} {dataset_path} --stream-output True'  # noqa: F401, E501
    if is_smoke:
        command += ' --num-prompts 300'

    for batch in [128, 256]:
        csv_path = f'{benchmark_path}/restful_batch_{batch}_1th.csv'
        cmd = ' '.join(
            [command, '--concurrency',
             str(batch), '--csv', csv_path])

        with open(benchmark_log, 'w') as f:
            f.writelines('reproduce command: ' + cmd + '\n')
            print('reproduce command: ' + cmd)

            benchmark_res = subprocess.run([cmd],
                                           stdout=f,
                                           stderr=PIPE,
                                           shell=True,
                                           text=True,
                                           encoding='utf-8')
            f.writelines(benchmark_res.stderr)
    if benchmark_res.returncode == 0 and os.path.isfile(csv_path) is False:
        return False, benchmark_log, 'result is empty'
    return benchmark_res.returncode == 0, benchmark_log, benchmark_res.stderr


def get_command_with_extra(cmd, cuda_prefix: str = None):
    if cuda_prefix is not None and len(cuda_prefix) > 0:
        cmd = ' '.join([cuda_prefix, cmd])
    return cmd


def create_multi_level_directory(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        return

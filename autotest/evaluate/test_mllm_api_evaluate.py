import os

import pytest
import utils.constant as constant
from utils.config_utils import get_case_str_by_config, get_func_config_list, get_workerid
from utils.evaluate_utils import mllm_eval_test
from utils.run_restful_chat import start_openai_service, start_proxy_server, stop_restful_api, terminate_restful_api


def run_eval_test_new(config, run_config, worker_id, test_type='infer', eval_config_name='default'):
    extra_command = constant.MLLM_EVAL_CONFIGS.get(eval_config_name, {})
    eval_path = config.get('mllm_eval_path')
    case_name = get_case_str_by_config(run_config)

    if test_type == 'infer':
        proxy_pid, proxy_process = start_proxy_server(config.get('server_log_path'), constant.PROXY_PORT)
        work_num = int(8 / run_config.get('parallel_config', {}).get('tp', 1))
        run_config_new = run_config.copy()
        if 'extra_params' not in run_config_new:
            run_config_new['extra_params'] = {}
        run_config_new['extra_params']['proxy-url'] = f'http://127.0.0.1:{constant.PROXY_PORT}'

        from concurrent.futures import ThreadPoolExecutor

        def run_openai_service_start(i):
            return start_openai_service(config, run_config_new, f'gw{i}')

        with ThreadPoolExecutor(max_workers=work_num) as executor:
            futures = [executor.submit(run_openai_service_start, i) for i in range(int(work_num))]
        results = []
        for future in futures:
            pid, content = future.result()
            results.append((pid, content))

        try:
            model_path = os.path.join(config.get('model_path'), run_config.get('model'))
            extra_command += f' --api-nproc {work_num * 16}'
            mllm_eval_test(model_path,
                           eval_path,
                           case_name,
                           port=constant.PROXY_PORT,
                           test_type=test_type,
                           extra_command=extra_command)
        finally:
            for i in range(work_num):
                terminate_restful_api(f'gw{i}')
            stop_restful_api(proxy_pid, proxy_process)
    else:  # eval
        port = constant.PROXY_PORT + get_workerid(worker_id)
        proxy_pid, proxy_process = start_proxy_server(config.get('server_log_path'), port)
        eval_run_config = constant.EVAL_RUN_CONFIG.copy()
        if 'extra_params' not in eval_run_config:
            eval_run_config['extra_params'] = {}
        eval_run_config['extra_params']['proxy-url'] = f'http://127.0.0.1:{port}'
        pid, content = start_openai_service(config, eval_run_config, worker_id)
        try:
            if pid > 0:
                model_path = os.path.join(config.get('model_path'), eval_run_config.get('model'))
                mllm_eval_test(model_path, eval_path, case_name, port=port, test_type=test_type)
            else:
                assert False, f'Failed to start RESTful API server: {content}'
        finally:
            if pid > 0:
                terminate_restful_api(worker_id)
            stop_restful_api(proxy_pid, proxy_process)


TURBOMIND = 'turbomind'
PYTORCH = 'pytorch'


def get_models(backend, parallel_config):
    return get_func_config_list(backend,
                                parallel_config,
                                model_type='vl_model',
                                func_type='mllm_evaluate',
                                extra={
                                    'session-len': 65536,
                                    'cache-max-entry-count': 0.6
                                })


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 1}))
def test_turbomind_vl_eval_tp1(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 2}))
def test_turbomind_vl_eval_tp2(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 4}))
def test_turbomind_vl_eval_tp4(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 8}))
def test_turbomind_vl_eval_tp8(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 1}))
def test_pytorch_vl_eval_tp1(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 2}))
def test_pytorch_vl_eval_tp2(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 4}))
def test_pytorch_vl_eval_tp4(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 8}))
def test_pytorch_vl_eval_tp8(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 16}))
def test_pytorch_vl_eval_tp16(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'infer')


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 1}))
def test_turbomind_eval_tp1(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 2}))
def test_turbomind_eval_tp2(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 4}))
def test_turbomind_eval_tp4(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 8}))
def test_turbomind_eval_tp8(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 1}))
def test_pytorch_eval_tp1(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 2}))
def test_pytorch_eval_tp2(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 4}))
def test_pytorch_eval_tp4(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 8}))
def test_pytorch_eval_tp8(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 16}))
def test_pytorch_eval_tp16(config, run_config, worker_id):
    run_eval_test_new(config, run_config, worker_id, 'eval')

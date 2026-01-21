import os
import time

import pytest
import utils.constant as constant
from utils.config_utils import get_case_str_by_config, get_func_config_list, get_workerid
from utils.evaluate_utils import eval_test
from utils.proxy_distributed_utils import ApiServerPerTest, proxy_worker_node_wait
from utils.ray_distributed_utils import ray_worker_node_wait
from utils.run_restful_chat import start_openai_service, start_proxy_server, stop_restful_api, terminate_restful_api


def _run_ray_distributed_test(
        config,
        run_config,
        worker_id,
        test_type='infer',
        manager=None,  # ← New parameter: pass in shared manager
        eval_config_name='default'):
    """Universal distributed test executor (using shared Ray cluster)"""
    assert manager is not None, 'Manager instance must be provided'
    if 'gpt' in run_config.get('model', '').lower():
        eval_config_name = 'gpt'
        preset_config = constant.EVAL_CONFIGS.get(eval_config_name, {})

    if manager.is_master:
        model_name = run_config['model']
        model_path = os.path.join(config['model_path'], model_name)
        preset_config = constant.EVAL_CONFIGS.get(eval_config_name, {})
        eval_path = config.get('eval_path')

        # Start API Server for current model (master node starts/stops, worker nodes verify)
        manager.start_lmdeploy_api_server(model_path=model_path, run_config=run_config)

        try:
            print(f'🧪 Master node executing {test_type} test ({eval_config_name})...')
            case_name = get_case_str_by_config(run_config)

            result, msg = eval_test(model_path,
                                    eval_path,
                                    case_name,
                                    port=constant.PROXY_PORT,
                                    test_type=test_type,
                                    **preset_config)
            assert result, f'❌ {test_type} test failed: {msg}'
            print(f'✅ {test_type} test passed')

        finally:
            # Clean up API Server for current model (worker nodes skip)
            manager.cleanup(force=False)
    else:
        time.sleep(10)
        ray_worker_node_wait(manager, timeout_minutes=4880)


def _run_proxy_distributed_test(config,
                                run_config,
                                worker_id,
                                test_type='infer',
                                manager=None,
                                eval_config_name='default'):
    assert manager is not None, 'Manager instance must be provided'

    if 'gpt' in run_config.get('model', '').lower():
        eval_config_name = 'gpt'

    preset_config = constant.EVAL_CONFIGS.get(eval_config_name, {})
    model_name = run_config['model']
    model_path = os.path.join(config['model_path'], model_name)

    api_server = ApiServerPerTest(proxy_manager=manager, model_path=model_path, run_config=run_config)
    api_server.start()

    try:
        if manager.is_master:
            api_server.wait_until_ready()
            print(f'🧪 Master node executing {test_type} test ({eval_config_name})...')
            eval_path = config.get('eval_path')
            case_name = get_case_str_by_config(run_config)

            result, msg = eval_test(model_path,
                                    eval_path,
                                    case_name,
                                    port=constant.PROXY_PORT,
                                    test_type=test_type,
                                    **preset_config)
            assert result, f'❌ {test_type} test failed: {msg}'
            print(f'✅ {test_type} test passed')

        else:
            print(f'⏸️ Worker node {manager.node_rank} waiting for master to complete test...')
            proxy_worker_node_wait(manager, timeout_minutes=4880)

    finally:
        api_server.cleanup()
        if manager.is_master:
            time.sleep(1)


def run_eval_test(config, run_config, worker_id, test_type='infer', eval_config_name='default'):
    """Run test with specified evaluation configuration."""
    if 'gpt' in run_config.get('model', '').lower():
        eval_config_name = 'gpt'
    if str(config.get('env_tag')) == 'a100':
        eval_config_name = f'{eval_config_name}-32k'
    preset_config = constant.EVAL_CONFIGS.get(eval_config_name, {})
    eval_path = config.get('eval_path')
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
            eval_test(model_path, eval_path, case_name, port=constant.PROXY_PORT, test_type=test_type, **preset_config)
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
                eval_test(model_path, eval_path, case_name, port=port, test_type=test_type, **preset_config)
            else:
                assert False, f'Failed to start RESTful API server: {content}'
        finally:
            if pid > 0:
                terminate_restful_api(worker_id)
            stop_restful_api(proxy_pid, proxy_process)


TURBOMIND = 'turbomind'
PYTORCH = 'pytorch'


def get_models(backend, parallel_config):
    return get_func_config_list(backend, parallel_config, func_type='evaluate', extra={'session_len': 65536})


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 1}))
def test_turbomind_infer_tp1(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 2}))
def test_turbomind_infer_tp2(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 4}))
def test_turbomind_infer_tp4(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 8}))
def test_turbomind_infer_tp8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 1}))
def test_pytorch_restful_tp1(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 2}))
def test_pytorch_restful_tp2(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 4}))
def test_pytorch_restful_tp4(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 8}))
def test_pytorch_restful_tp8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 16}))
def test_pytorch_restful_tp16(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_tp16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 16}))
def test_pytorch_restful_distributed_tp16(shared_ray_manager, config, run_config, worker_id):
    _run_ray_distributed_test(config=config,
                              run_config=run_config,
                              worker_id=worker_id,
                              test_type='infer',
                              manager=shared_ray_manager)


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_dpep8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'dp': 8, 'ep': 8}))
def test_pytorch_restful_distributed_dpep8(shared_proxy_manager, config, run_config, worker_id):
    _run_proxy_distributed_test(config=config,
                                run_config=run_config,
                                worker_id=worker_id,
                                test_type='infer',
                                manager=shared_proxy_manager)


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_dpep16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'dp': 16, 'ep': 16}))
def test_pytorch_restful_distributed_dpep16(shared_proxy_manager, config, run_config, worker_id):
    _run_proxy_distributed_test(config=config,
                                run_config=run_config,
                                worker_id=worker_id,
                                test_type='infer',
                                manager=shared_proxy_manager)


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 1}))
def test_turbomind_eval_tp1(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 2}))
def test_turbomind_eval_tp2(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 4}))
def test_turbomind_eval_tp4(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'tp': 8}))
def test_turbomind_eval_tp8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 1}))
def test_pytorch_eval_tp1(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 2}))
def test_pytorch_eval_tp2(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 4}))
def test_pytorch_eval_tp4(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 8}))
def test_pytorch_eval_tp8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 16}))
def test_pytorch_eval_tp16(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_tp16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'tp': 16}))
def test_pytorch_eval_distributed_tp16(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_dpep8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(PYTORCH, {'dp': 8, 'ep': 8}))
def test_pytorch_eval_distributed_dpep8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(TURBOMIND, {'cp': 2, 'tp': 8}))
def test_turbomind_eval_cp2tp8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')

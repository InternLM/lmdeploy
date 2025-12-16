import os
import time

import pytest
import utils.constant as constant
from utils.config_utils import get_evaluate_pytorch_model_list, get_evaluate_turbomind_model_list, get_workerid
from utils.evaluate_utils import eval_test
from utils.proxy_distributed_utils import ApiServerPerTest, proxy_worker_node_wait
from utils.ray_distributed_utils import ray_worker_node_wait
from utils.run_restful_chat import start_proxy_server, start_restful_api, stop_restful_api


@pytest.fixture(scope='function')
def prepare_environment(request, config, worker_id):
    param = request.param
    model = param['model']
    backend = param['backend']
    model_path = config.get('model_path') + '/' + model
    pid, startRes = start_restful_api(config, param, model, model_path, backend, worker_id)
    yield param
    stop_restful_api(pid, startRes, param)


@pytest.fixture(scope='function')
def prepare_environment_judge_evaluate(request, config, worker_id):
    if get_workerid(worker_id) is None:
        port = constant.PROXY_PORT
    else:
        port = constant.PROXY_PORT + get_workerid(worker_id)
    judge_config = {
        'model': 'Qwen/Qwen2.5-32B-Instruct',
        'backend': 'turbomind',
        'param': {
            'tp_num':
            2,
            'extra':
            '--server-name 127.0.0.1 --proxy-url http://127.0.0.1:{} --session-len 65536 '
            '--cache-max-entry-count 0.7 '.format(port),
            'cuda_prefix':
            None
        },
        'log_path': config.get('log_path'),
    }

    param = judge_config['param']
    model = judge_config['model']
    backend = judge_config['backend']
    model_path = config.get('model_path') + '/' + model

    proxy_pid, proxy_process = start_proxy_server(config, worker_id)

    judge_pid, judge_start_res = start_restful_api(config, param, model, model_path, backend, worker_id)

    try:
        yield request.param
    finally:
        stop_restful_api(judge_pid, judge_start_res, request.param)
        stop_restful_api(proxy_pid, proxy_process, request.param)


def _run_ray_distributed_test(
        config,
        run_id,
        model_param,
        worker_id,
        test_type='infer',
        manager=None,  # ← New parameter: pass in shared manager
        eval_config_name='default'):
    """Universal distributed test executor (using shared Ray cluster)"""
    assert manager is not None, 'Manager instance must be provided'
    if 'gpt' in model_param.get('model', '').lower():
        eval_config_name = 'gpt'
        preset_config = constant.EVAL_CONFIGS.get(eval_config_name, {})

    if manager.is_master:
        model_name = model_param['model']
        model_path = os.path.join(config['model_path'], model_name)
        preset_config = constant.EVAL_CONFIGS.get(eval_config_name, {})

        # Start API Server for current model (master node starts/stops, worker nodes verify)
        manager.start_lmdeploy_api_server(model_path=model_path, model_param=model_param)

        try:
            print(f'🧪 Master node executing {test_type} test ({eval_config_name})...')
            result, msg = eval_test(config,
                                    run_id,
                                    model_param,
                                    worker_id=worker_id,
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
                                run_id,
                                model_param,
                                worker_id,
                                test_type='infer',
                                manager=None,
                                eval_config_name='default'):
    assert manager is not None, 'Manager instance must be provided'

    if 'gpt' in model_param.get('model', '').lower():
        eval_config_name = 'gpt'

    preset_config = constant.EVAL_CONFIGS.get(eval_config_name, {})
    model_name = model_param['model']
    model_path = os.path.join(config['model_path'], model_name)

    api_server = ApiServerPerTest(proxy_manager=manager, model_path=model_path, model_param=model_param)
    api_server.start()

    try:
        if manager.is_master:
            api_server.wait_until_ready()
            print(f'🧪 Master node executing {test_type} test ({eval_config_name})...')

            result, msg = eval_test(config,
                                    run_id,
                                    model_param,
                                    worker_id=worker_id,
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


def get_turbomind_model_list(tp_num):
    model_list = get_evaluate_turbomind_model_list(tp_num, kvint_list=[4, 8])
    new_model_list = []
    for model in model_list:
        if 'Qwen3-235B-A22B-Thinking-2507' in model['model']:
            model['extra'] += '--session-len 65536 --cache-max-entry-count 0.9 --max-batch-size 1024 '
        else:
            model['extra'] += '--session-len 65536 --cache-max-entry-count 0.9 '
        model['cuda_prefix'] = None
        new_model_list.append(model)
    return new_model_list


def get_pytorch_model_list(tp_num):
    model_list = get_evaluate_pytorch_model_list(tp_num, kvint_list=[4, 8])
    new_model_list = []
    for model in model_list:
        if 'Qwen3-235B-A22B-Thinking-2507' in model['model']:
            model['extra'] += '--session-len 65536 --cache-max-entry-count 0.9 --max-batch-size 1024 '
        else:
            model['extra'] += '--session-len 65536 --cache-max-entry-count 0.9 '
        model['cuda_prefix'] = None
        new_model_list.append(model)
    return new_model_list


def run_test(config, run_id, prepare_environment, worker_id, test_type='infer', eval_config_name='default'):
    """Run test with specified evaluation configuration."""
    if 'gpt' in prepare_environment.get('model', '').lower():
        eval_config_name = 'gpt'
    preset_config = constant.EVAL_CONFIGS.get(eval_config_name, {})

    if test_type == 'infer':
        port = constant.DEFAULT_PORT
    else:  # eval
        port = constant.PROXY_PORT

    if get_workerid(worker_id) is None:
        result, msg = eval_test(config,
                                run_id,
                                prepare_environment,
                                worker_id=worker_id,
                                port=port,
                                test_type=test_type,
                                **preset_config)
    else:
        result, msg = eval_test(config,
                                run_id,
                                prepare_environment,
                                worker_id=worker_id,
                                port=port + get_workerid(worker_id),
                                test_type=test_type,
                                **preset_config)
    return result, msg


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_turbomind_model_list(tp_num=1), indirect=True)
def test_turbomind_restful_tp1(config, run_id, prepare_environment, worker_id):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer')
    assert result, msg


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_turbomind_model_list(tp_num=2), indirect=True)
def test_turbomind_restful_tp2(config, run_id, prepare_environment, worker_id):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer')
    assert result, msg


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_turbomind_model_list(tp_num=4), indirect=True)
def test_turbomind_restful_tp4(config, run_id, prepare_environment, worker_id):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer')
    assert result, msg


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_turbomind_model_list({'cp': 2, 'tp': 8}), indirect=True)
def test_turbomind_restful_cp2tp8(config, run_id, prepare_environment, worker_id):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer')
    assert result, msg


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_turbomind_model_list(tp_num=8), indirect=True)
def test_turbomind_restful_tp8(config, run_id, prepare_environment, worker_id):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer')
    assert result, msg


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=1), indirect=True)
def test_pytorch_restful_tp1(config, run_id, prepare_environment, worker_id):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer')
    assert result, msg


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=2), indirect=True)
def test_pytorch_restful_tp2(config, run_id, prepare_environment, worker_id):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer')
    assert result, msg


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=4), indirect=True)
def test_pytorch_restful_tp4(config, run_id, prepare_environment, worker_id):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer')
    assert result, msg


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=8), indirect=True)
def test_pytorch_restful_tp8(config, run_id, prepare_environment, worker_id):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer')
    assert result, msg


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=16), indirect=True)
def test_pytorch_restful_tp16(config, run_id, prepare_environment, worker_id):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer')
    assert result, msg


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_tp16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model_param', get_pytorch_model_list(tp_num=16))
def test_pytorch_restful_distributed_tp16(shared_ray_manager, config, run_id, model_param, worker_id):
    _run_ray_distributed_test(config=config,
                              run_id=run_id,
                              model_param=model_param,
                              worker_id=worker_id,
                              test_type='infer',
                              manager=shared_ray_manager)


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_dpep8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model_param', get_pytorch_model_list({'dp': 8, 'ep': 8}))
def test_pytorch_restful_distributed_dpep8(shared_proxy_manager, config, run_id, model_param, worker_id):
    _run_proxy_distributed_test(config=config,
                                run_id=run_id,
                                model_param=model_param,
                                worker_id=worker_id,
                                test_type='infer',
                                manager=shared_proxy_manager)


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_dpep16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model_param', get_pytorch_model_list({'dp': 16, 'ep': 16}))
def test_pytorch_restful_distributed_dpep16(shared_proxy_manager, config, run_id, model_param, worker_id):
    _run_proxy_distributed_test(config=config,
                                run_id=run_id,
                                model_param=model_param,
                                worker_id=worker_id,
                                test_type='infer',
                                manager=shared_proxy_manager)


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=1), indirect=True)
def test_pytorch_judgeeval_tp1(config, run_id, prepare_environment_judge_evaluate, worker_id):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval')
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=2), indirect=True)
def test_pytorch_judgeeval_tp2(config, run_id, prepare_environment_judge_evaluate, worker_id):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval')
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=4), indirect=True)
def test_pytorch_judgeeval_tp4(config, run_id, prepare_environment_judge_evaluate, worker_id):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval')
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=8), indirect=True)
def test_pytorch_judgeeval_tp8(config, run_id, prepare_environment_judge_evaluate, worker_id):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval')
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=16), indirect=True)
def test_pytorch_judgeeval_tp16(config, run_id, prepare_environment_judge_evaluate, worker_id):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval')
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_tp16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=16), indirect=True)
def test_pytorch_judgeeval_distributed_tp16(config, run_id, prepare_environment_judge_evaluate, worker_id):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval')
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_dpep8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate',
                         get_pytorch_model_list({
                             'dp': 8,
                             'ep': 8
                         }),
                         indirect=True)
def test_pytorch_judgeeval_distributed_dpep8(config, run_id, prepare_environment_judge_evaluate, worker_id):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval')
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_turbomind_model_list(tp_num=1), indirect=True)
def test_turbomind_judgeeval_tp1(config, run_id, prepare_environment_judge_evaluate, worker_id):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval')
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_turbomind_model_list(tp_num=2), indirect=True)
def test_turbomind_judgeeval_tp2(config, run_id, prepare_environment_judge_evaluate, worker_id):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval')
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_turbomind_model_list(tp_num=4), indirect=True)
def test_turbomind_judgeeval_tp4(config, run_id, prepare_environment_judge_evaluate, worker_id):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval')
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_turbomind_model_list(tp_num=8), indirect=True)
def test_turbomind_judgeeval_tp8(config, run_id, prepare_environment_judge_evaluate, worker_id):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval')
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate',
                         get_turbomind_model_list({
                             'cp': 2,
                             'tp': 8
                         }),
                         indirect=True)
def test_turbomind_judgeeval_cp2tp8(config, run_id, prepare_environment_judge_evaluate, worker_id):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval')
    assert result, msg

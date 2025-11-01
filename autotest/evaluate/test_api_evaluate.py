import pytest
from utils.config_utils import get_evaluate_pytorch_model_list, get_evaluate_turbomind_model_list, get_workerid
from utils.evaluate_utils import restful_test
from utils.run_restful_chat import start_proxy_server, start_restful_api, stop_restful_api

DEFAULT_PORT = 23333
PROXY_PORT = 8000

EVAL_CONFIGS = {
    'default': {
        'query_per_second': 4,
        'max_out_len': 32768,
        'max_seq_len': 32768,
        'batch_size': 500,
        'temperature': 0.6,
    }
}


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
        port = PROXY_PORT
    else:
        port = PROXY_PORT + get_workerid(worker_id)
    judge_config = {
        'model': 'Qwen/Qwen2.5-32B-Instruct',
        'backend': 'turbomind',
        'param': {
            'tp_num':
            2,
            'extra':
            '--server-name 127.0.0.1 --proxy-url http://127.0.0.1:{} --session-len 46000 '
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


def get_turbomind_model_list(tp_num):
    model_list = get_evaluate_turbomind_model_list(tp_num, kvint_list=[4, 8])
    new_model_list = []
    for model in model_list:
        model['cuda_prefix'] = None
        new_model_list.append(model)
    return new_model_list


def get_pytorch_model_list(tp_num):
    model_list = get_evaluate_pytorch_model_list(tp_num, kvint_list=[4, 8])
    new_model_list = []
    for model in model_list:
        model['cuda_prefix'] = None
        new_model_list.append(model)
    return new_model_list


def run_test(config, run_id, prepare_environment, worker_id, test_type='infer', eval_config_name='default'):
    """Run test with specified evaluation configuration."""
    preset_config = EVAL_CONFIGS.get(eval_config_name, {})

    if test_type == 'infer':
        port = DEFAULT_PORT
    else:  # eval
        port = PROXY_PORT

    if get_workerid(worker_id) is None:
        result, msg = restful_test(config,
                                   run_id,
                                   prepare_environment,
                                   worker_id=worker_id,
                                   port=port,
                                   test_type=test_type,
                                   **preset_config)
    else:
        result, msg = restful_test(config,
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
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_restful_tp1(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer', eval_config)
    assert result, msg


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_turbomind_model_list(tp_num=2), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_restful_tp2(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer', eval_config)
    assert result, msg


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_turbomind_model_list(tp_num=4), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_restful_tp4(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer', eval_config)
    assert result, msg


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_turbomind_model_list(tp_num=8), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_restful_tp8(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer', eval_config)
    assert result, msg


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=1), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_restful_tp1(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer', eval_config)
    assert result, msg


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=2), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_restful_tp2(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer', eval_config)
    assert result, msg


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=4), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_restful_tp4(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer', eval_config)
    assert result, msg


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', get_pytorch_model_list(tp_num=8), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_restful_tp8(config, run_id, prepare_environment, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment, worker_id, 'infer', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=1), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_judgeeval_tp1(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=2), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_judgeeval_tp2(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=4), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_judgeeval_tp4(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_pytorch_model_list(tp_num=8), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_pytorch_judgeeval_tp8(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_turbomind_model_list(tp_num=1), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_judgeeval_tp1(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_turbomind_model_list(tp_num=2), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_judgeeval_tp2(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_turbomind_model_list(tp_num=4), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_judgeeval_tp4(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment_judge_evaluate', get_turbomind_model_list(tp_num=8), indirect=True)
@pytest.mark.parametrize('eval_config', list(EVAL_CONFIGS.keys()))
def test_turbomind_judgeeval_tp8(config, run_id, prepare_environment_judge_evaluate, worker_id, eval_config):
    result, msg = run_test(config, run_id, prepare_environment_judge_evaluate, worker_id, 'eval', eval_config)
    assert result, msg

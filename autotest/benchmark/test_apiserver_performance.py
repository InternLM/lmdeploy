import pytest
from utils.benchmark_utils import restful_test
from utils.config_utils import get_benchmark_model_list
from utils.run_restful_chat import start_restful_api, stop_restful_api


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    param = request.param
    model = param['model']
    backend = param['backend']
    model_path = config.get('model_path') + '/' + model

    pid, startRes = start_restful_api(config, param, model, model_path,
                                      backend, worker_id)
    yield param
    stop_restful_api(pid, startRes, param)


def getModelList(tp_num):
    model_list = get_benchmark_model_list(tp_num, kvint_list=[4, 8])
    new_model_list = []
    for model in model_list:
        if model['backend'] == 'pytorch':
            model['extra'] = '--max-batch-size 256 --cache-max-entry-count 0.8'
        elif 'Llama-2' in model['model']:
            model[
                'extra'] = '--max-batch-size 256 --cache-max-entry-count 0.95'
        elif 'internlm2' in model['model']:
            model['extra'] = '--max-batch-size 256 --cache-max-entry-count 0.9'
        else:
            model['extra'] = '--max-batch-size 256'
        model['cuda_prefix'] = None
        new_model_list.append(model)
    return new_model_list


@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment',
                         getModelList(tp_num=1),
                         indirect=True)
def test_restful_tp1(config, run_id, prepare_environment, worker_id):
    result, msg = restful_test(config,
                               run_id,
                               prepare_environment,
                               worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment',
                         getModelList(tp_num=2),
                         indirect=True)
def test_restful_tp2(config, run_id, prepare_environment, worker_id):
    result, msg = restful_test(config,
                               run_id,
                               prepare_environment,
                               worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment',
                         getModelList(tp_num=4),
                         indirect=True)
def test_restful_tp4(config, run_id, prepare_environment, worker_id):
    result, msg = restful_test(config,
                               run_id,
                               prepare_environment,
                               worker_id=worker_id)

    assert result, msg


@pytest.mark.function
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('prepare_environment', [{
    'model': 'internlm/internlm2_5-20b-chat',
    'backend': 'pytorch',
    'tp_num': 2,
    'extra': '--max-batch-size 256 --cache-max-entry-count 0.9',
    'cuda_prefix': None
}, {
    'model': 'internlm/internlm2_5-20b-chat-inner-4bits',
    'backend': 'turbomind',
    'quant_policy': 0,
    'tp_num': 2,
    'extra': '--max-batch-size 256 --cache-max-entry-count 0.9',
    'cuda_prefix': None
}],
                         indirect=True)
def test_restful_func_tp2(config, run_id, prepare_environment, worker_id):
    result, msg = restful_test(config,
                               run_id,
                               prepare_environment,
                               worker_id=worker_id,
                               is_smoke=True)

    assert result, msg

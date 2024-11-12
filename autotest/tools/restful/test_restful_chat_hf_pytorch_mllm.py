import pytest
from utils.config_utils import get_torch_model_list, get_workerid
from utils.run_restful_chat import (run_vl_testcase, start_restful_api,
                                    stop_restful_api)

BASE_HTTP_URL = 'http://localhost'
DEFAULT_PORT = 23333


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    param = request.param
    model = param['model']
    model_path = config.get('model_path') + '/' + model

    pid, startRes = start_restful_api(config, param, model, model_path,
                                      'pytorch', worker_id)
    yield
    stop_restful_api(pid, startRes, param)


def getModelList(tp_num):
    return [{
        'model': item,
        'cuda_prefix': None,
        'tp_num': tp_num,
    } for item in get_torch_model_list(tp_num, model_type='vl_model')]


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment',
                         getModelList(tp_num=1),
                         indirect=True)
def test_restful_chat_tp1(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment',
                         getModelList(tp_num=2),
                         indirect=True)
def test_restful_chat_tp2(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


def getKvintModelList(tp_num, quant_policy: int = None):
    return [{
        'model': item,
        'cuda_prefix': None,
        'tp_num': tp_num,
        'extra': f'--quant-policy {quant_policy}'
    } for item in get_torch_model_list(
        tp_num, quant_policy=quant_policy, model_type='vl_model')
            if 'qwen2' not in item.lower() or quant_policy == 8]


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment',
                         getKvintModelList(tp_num=1, quant_policy=4),
                         indirect=True)
def test_restful_chat_kvint4_tp1(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment',
                         getKvintModelList(tp_num=2, quant_policy=4),
                         indirect=True)
def test_restful_chat_kvint4_tp2(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment',
                         getKvintModelList(tp_num=1, quant_policy=8),
                         indirect=True)
def test_restful_chat_kvint8_tp1(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment',
                         getKvintModelList(tp_num=2, quant_policy=8),
                         indirect=True)
def test_restful_chat_kvint8_tp2(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))

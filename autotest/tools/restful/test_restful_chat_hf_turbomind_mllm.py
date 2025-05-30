import pytest
from utils.config_utils import get_communicator_list, get_turbomind_model_list, get_workerid
from utils.run_restful_chat import run_vl_testcase, start_restful_api, stop_restful_api

BASE_HTTP_URL = 'http://localhost'
DEFAULT_PORT = 23333


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    param = request.param
    model = param['model']
    model_path = config.get('model_path') + '/' + model

    pid, startRes = start_restful_api(config, param, model, model_path, 'turbomind', worker_id)
    yield
    stop_restful_api(pid, startRes, param)


def getModelList(tp_num):
    model_list = []
    for communicator in get_communicator_list():
        model_list += [{
            'model': item,
            'cuda_prefix': None,
            'tp_num': tp_num,
            'extra': f'--communicator {communicator}'
        } for item in get_turbomind_model_list(tp_num, model_type='vl_model')]
    return model_list


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=1), indirect=True)
def test_restful_chat_tp1(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=2), indirect=True)
def test_restful_chat_tp2(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=4), indirect=True)
def test_restful_chat_tp4(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


def getKvintModelList(tp_num, quant_policy: int = None):
    model_list = []
    for communicator in get_communicator_list():
        model_list += [{
            'model': item,
            'cuda_prefix': None,
            'tp_num': tp_num,
            'extra': f'--quant-policy {quant_policy} --communicator {communicator}'
        } for item in get_turbomind_model_list(tp_num, quant_policy=quant_policy, model_type='vl_model')]
    return model_list


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=1, quant_policy=4), indirect=True)
def test_restful_chat_kvint4_tp1(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=2, quant_policy=4), indirect=True)
def test_restful_chat_kvint4_tp2(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=4, quant_policy=4), indirect=True)
def test_restful_chat_kvint4_tp4(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=1, quant_policy=8), indirect=True)
def test_restful_chat_kvint8_tp1(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=2, quant_policy=8), indirect=True)
def test_restful_chat_kvint8_tp2(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=4, quant_policy=8), indirect=True)
def test_restful_chat_kvint8_tp4(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment', [
    {
        'model': 'OpenGVLab/InternVL2-4B',
        'cuda_prefix': None,
        'tp_num': 1,
    },
    {
        'model': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'cuda_prefix': None,
        'tp_num': 1
    },
    {
        'model': 'Qwen/Qwen2-VL-7B-Instruct-inner-4bits',
        'cuda_prefix': None,
        'tp_num': 1
    },
    {
        'model': 'THUDM/glm-4v-9b',
        'cuda_prefix': None,
        'tp_num': 1
    },
    {
        'model': 'THUDM/glm-4v-9b-inner-4bits',
        'cuda_prefix': None,
        'tp_num': 1
    },
    {
        'model': 'OpenGVLab/InternVL2-4B',
        'cuda_prefix': None,
        'tp_num': 1,
        'extra': ' --quant-policy 8 --communicator native'
    },
    {
        'model': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'cuda_prefix': None,
        'tp_num': 1,
        'extra': ' --quant-policy 8 --communicator native'
    },
    {
        'model': 'Qwen/Qwen2-VL-7B-Instruct-inner-4bits',
        'cuda_prefix': None,
        'tp_num': 1,
        'extra': ' --quant-policy 8 --communicator native'
    },
    {
        'model': 'THUDM/glm-4v-9b',
        'cuda_prefix': None,
        'tp_num': 1,
        'extra': ' --quant-policy 8 --communicator native'
    },
    {
        'model': 'THUDM/glm-4v-9b-inner-4bits',
        'cuda_prefix': None,
        'tp_num': 1,
        'extra': ' --quant-policy 8 --communicator native'
    },
],
                         indirect=True)
def test_restful_chat_fallback_backend_tp1(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.restful_api_vl
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', [
    {
        'model': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
        'cuda_prefix': None,
        'tp_num': 2
    },
    {
        'model': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
        'cuda_prefix': None,
        'tp_num': 2,
        'extra': ' --communicator native'
    },
    {
        'model': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
        'cuda_prefix': None,
        'tp_num': 2,
        'extra': ' --quant-policy 8 --communicator native'
    },
],
                         indirect=True)
def test_restful_chat_fallback_backend_tp2(config, worker_id):
    if get_workerid(worker_id) is None:
        run_vl_testcase(config)
    else:
        run_vl_testcase(config, port=DEFAULT_PORT + get_workerid(worker_id))

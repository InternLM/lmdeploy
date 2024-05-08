import pytest
from utils.config_utils import get_torch_model_list, get_workerid
from utils.run_restful_chat import (run_all_step, start_restful_api,
                                    stop_restful_api)

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
        'tp_num': tp_num
    } for item in get_torch_model_list(tp_num)]


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment',
                         getModelList(tp_num=1),
                         indirect=True)
def test_restful_chat_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config,
                     common_case_config,
                     worker_id=worker_id,
                     port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment',
                         getModelList(tp_num=2),
                         indirect=True)
def test_restful_chat_tp2(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config,
                     common_case_config,
                     worker_id=worker_id,
                     port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment', [{
    'model': 'Qwen/Qwen-7B-Chat',
    'cuda_prefix': None,
    'tp_num': 2,
    'modelscope': True
}],
                         indirect=True)
def test_modelscope_restful_chat_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config,
                     common_case_config,
                     worker_id=worker_id,
                     port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.parametrize(
    'prepare_environment',
    [{
        'model': 'meta-llama/Llama-2-7b-chat-hf',
        'cuda_prefix': None,
        'tp_num': 1,
        'extra': ' --adapters lora/Llama2-Chinese-7b-Chat-LoRA'
    }],
    indirect=True)
def test_restful_chat_with_lora_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config,
                     common_case_config,
                     worker_id=worker_id,
                     port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.gpu_num_2
@pytest.mark.parametrize(
    'prepare_environment',
    [{
        'model': 'baichuan-inc/Baichuan2-13B-Chat',
        'cuda_prefix': None,
        'tp_num': 2,
        'extra':
        ' --adapters a=lora/2024-01-25_self_dup b=lora/2024-01-25_self'
    }],
    indirect=True)
def test_restful_chat_with_lora_tp2(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config,
                     common_case_config,
                     worker_id=worker_id,
                     port=DEFAULT_PORT + get_workerid(worker_id))

import pytest
from utils.config_utils import get_torch_model_list, get_workerid
from utils.run_restful_chat import run_all_step, run_reasoning_case, run_tools_case, start_restful_api, stop_restful_api

DEFAULT_PORT = 23333


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    param = request.param
    model = param['model']
    model_path = config.get('model_path') + '/' + model

    pid, startRes = start_restful_api(config, param, model, model_path, 'pytorch', worker_id)
    yield
    stop_restful_api(pid, startRes, param)


def getModelList(tp_num):
    return [{
        'model': item,
        'cuda_prefix': None,
        'tp_num': tp_num
    } for item in get_torch_model_list(tp_num, exclude_dup=True)]


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.test_ascend
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=1), indirect=True)
def test_restful_chat_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=2), indirect=True)
def test_restful_chat_tp2(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=4), indirect=True)
def test_restful_chat_tp4(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=8), indirect=True)
def test_restful_chat_tp8(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


def getKvintModelList(tp_num, quant_policy):
    return [{
        'model': item,
        'cuda_prefix': None,
        'tp_num': tp_num,
        'extra': f'--quant-policy {quant_policy}'
    } for item in get_torch_model_list(tp_num, quant_policy=quant_policy, exclude_dup=True)]


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=1, quant_policy=4), indirect=True)
def test_restful_chat_kvint4_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=2, quant_policy=4), indirect=True)
def test_restful_chat_kvint4_tp2(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=4, quant_policy=4), indirect=True)
def test_restful_chat_kvint4_tp4(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=1, quant_policy=8), indirect=True)
def test_restful_chat_kvint8_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=2, quant_policy=8), indirect=True)
def test_restful_chat_kvint8_tp2(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=4, quant_policy=8), indirect=True)
def test_restful_chat_kvint8_tp4(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=8, quant_policy=8), indirect=True)
def test_restful_chat_kvint8_tp8(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [{
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'cuda_prefix': None,
    'tp_num': 1,
    'modelscope': True
}],
                         indirect=True)
def test_modelscope_restful_chat_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [{
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
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.gpu_num_2
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment',
                         [{
                             'model': 'baichuan-inc/Baichuan2-13B-Chat',
                             'cuda_prefix': None,
                             'tp_num': 2,
                             'extra': ' --adapters a=lora/2024-01-25_self_dup b=lora/2024-01-25_self'
                         }],
                         indirect=True)
def test_restful_chat_with_lora_tp2(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [
    {
        'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'cuda_prefix': None,
        'tp_num': 1,
        'extra': ' --reasoning-parser deepseek-r1'
    },
],
                         indirect=True)
def test_restful_chat_reasoning_tp1(config, worker_id):
    if get_workerid(worker_id) is None:
        run_reasoning_case(config)
    else:
        run_reasoning_case(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [
    {
        'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'cuda_prefix': None,
        'tp_num': 2,
        'extra': ' --reasoning-parser deepseek-r1'
    },
],
                         indirect=True)
def test_restful_chat_reasoning_tp2(config, worker_id):
    if get_workerid(worker_id) is None:
        run_reasoning_case(config)
    else:
        run_reasoning_case(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [
    {
        'model': 'internlm/internlm2_5-7b-chat',
        'cuda_prefix': None,
        'tp_num': 1,
        'extra': ' --tool-call-parser internlm'
    },
    {
        'model': 'Qwen/Qwen2.5-7B-Instruct',
        'cuda_prefix': None,
        'tp_num': 1,
        'extra': ' --tool-call-parser qwen'
    },
],
                         indirect=True)
def test_restful_chat_tools_tp1(config, worker_id):
    if get_workerid(worker_id) is None:
        run_tools_case(config)
    else:
        run_tools_case(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [
    {
        'model': 'internlm/internlm2_5-20b-chat',
        'cuda_prefix': None,
        'tp_num': 2,
        'extra': ' --tool-call-parser internlm'
    },
],
                         indirect=True)
def test_restful_chat_tools_tp2(config, worker_id):
    if get_workerid(worker_id) is None:
        run_tools_case(config)
    else:
        run_tools_case(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_4
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [
    {
        'model': 'meta-llama/Meta-Llama-3-1-70B-Instruct',
        'cuda_prefix': None,
        'tp_num': 4,
        'extra': ' --tool-call-parser llama3'
    },
    {
        'model': 'Qwen/Qwen2.5-72B-Instruct',
        'cuda_prefix': None,
        'tp_num': 4,
        'extra': ' --tool-call-parser qwen'
    },
],
                         indirect=True)
def test_restful_chat_tools_tp4(config, worker_id):
    if get_workerid(worker_id) is None:
        run_tools_case(config)
    else:
        run_tools_case(config, port=DEFAULT_PORT + get_workerid(worker_id))

import pytest
from utils.config_utils import get_communicator_list, get_turbomind_model_list, get_workerid
from utils.run_restful_chat import (run_all_step, run_reasoning_case, run_tools_case, start_restful_api,
                                    stop_restful_api, test_logprobs)

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
        } for item in get_turbomind_model_list(tp_num)]
    return model_list


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=1), indirect=True)
def test_restful_chat_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=2), indirect=True)
def test_restful_chat_tp2(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=4), indirect=True)
def test_restful_chat_tp4(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=8), indirect=True)
def test_restful_chat_tp8(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


def getKvintModelList(tp_num, quant_policy):
    model_list = []
    for communicator in get_communicator_list(tp_num):
        model_list += [{
            'model': item,
            'cuda_prefix': None,
            'tp_num': tp_num,
            'extra': f'--quant-policy {quant_policy} --communicator {communicator}'
        } for item in get_turbomind_model_list(tp_num, quant_policy=quant_policy)]
    return model_list


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
@pytest.mark.parametrize('prepare_environment', [
    {
        'model': 'microsoft/Phi-3-mini-4k-instruct',
        'cuda_prefix': None,
        'tp_num': 1,
    },
    {
        'model': 'microsoft/Phi-3-mini-4k-instruct-inner-4bits',
        'cuda_prefix': None,
        'tp_num': 1
    },
    {
        'model': 'microsoft/Phi-3-mini-4k-instruct-inner-w8a8',
        'cuda_prefix': None,
        'tp_num': 1
    },
    {
        'model': 'microsoft/Phi-3-mini-4k-instruct',
        'cuda_prefix': None,
        'tp_num': 1,
        'extra': ' --quant-policy 8'
    },
    {
        'model': 'microsoft/Phi-3-mini-4k-instruct-inner-4bits',
        'cuda_prefix': None,
        'tp_num': 1,
        'extra': ' --quant-policy 8'
    },
    {
        'model': 'microsoft/Phi-3-mini-4k-instruct-inner-w8a8',
        'cuda_prefix': None,
        'tp_num': 1,
        'extra': ' --quant-policy 8'
    },
],
                         indirect=True)
def test_restful_chat_fallback_backend_tp1(config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    if get_workerid(worker_id) is None:
        run_all_step(config, case_config)
    else:
        run_all_step(config, case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_2
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [
    {
        'model': 'google/gemma-2-27b-it',
        'cuda_prefix': None,
        'tp_num': 2
    },
    {
        'model': 'deepseek-ai/deepseek-moe-16b-chat',
        'cuda_prefix': None,
        'tp_num': 2
    },
    {
        'model': 'google/gemma-2-27b-it',
        'cuda_prefix': None,
        'tp_num': 2,
        'extra': ' --communicator cuda-ipc'
    },
    {
        'model': 'deepseek-ai/deepseek-moe-16b-chat',
        'cuda_prefix': None,
        'tp_num': 2,
        'extra': ' --communicator cuda-ipc'
    },
    {
        'model': 'google/gemma-2-27b-it',
        'cuda_prefix': None,
        'tp_num': 2,
        'extra': ' --quant-policy 8 --communicator cuda-ipc'
    },
    {
        'model': 'deepseek-ai/deepseek-moe-16b-chat',
        'cuda_prefix': None,
        'tp_num': 2,
        'extra': ' --quant-policy 8 --communicator cuda-ipc'
    },
],
                         indirect=True)
def test_restful_chat_fallback_backend_tp2(config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    if get_workerid(worker_id) is None:
        run_all_step(config, case_config)
    else:
        run_all_step(config, case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('prepare_environment', [
    {
        'model': 'internlm/internlm2_5-20b-chat',
        'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
        'tp_num': 2
    },
    {
        'model': 'internlm/internlm2_5-20b-chat-inner-4bits',
        'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
        'tp_num': 2
    },
    {
        'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
        'tp_num': 2
    },
    {
        'model': 'internlm/internlm2_5-20b-chat',
        'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
        'tp_num': 2,
        'extra': ' --communicator cuda-ipc'
    },
    {
        'model': 'internlm/internlm2_5-20b-chat-inner-4bits',
        'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
        'tp_num': 2,
        'extra': ' --communicator cuda-ipc'
    },
    {
        'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
        'tp_num': 2,
        'extra': ' --communicator cuda-ipc'
    },
],
                         indirect=True)
def test_restful_chat_pr(config, common_case_config):
    run_all_step(config, {key: value for key, value in common_case_config.items() if key == 'memory_test'})


@pytest.mark.order(7)
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('prepare_environment', [{
    'model': 'internlm/internlm2_5-20b-chat',
    'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
    'tp_num': 2
}],
                         indirect=True)
def test_restful_logprobs(worker_id):

    test_logprobs(worker_id)


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

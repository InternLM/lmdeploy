import os
import time

import pytest
from utils.config_utils import get_torch_model_list, get_workerid
from utils.proxy_distributed_utils import ApiServerPerTest, proxy_worker_node_wait
from utils.ray_distributed_utils import ray_worker_node_wait
from utils.run_restful_chat import run_all_step, run_reasoning_case, run_tools_case, start_restful_api, stop_restful_api

DEFAULT_PORT = 23333
PROXY_PORT = 8000


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    if hasattr(request, 'param'):
        param = request.param
        model = param['model']
        model_path = config.get('model_path') + '/' + model

        pid, startRes = start_restful_api(config, param, model, model_path, 'pytorch', worker_id)
        yield
        stop_restful_api(pid, startRes, param)
    else:
        yield


def getModelList(tp_num):
    return [{
        'model': item,
        'cuda_prefix': None,
        'tp_num': tp_num
    } for item in get_torch_model_list(tp_num, exclude_dup=True)]


def getPrefixCacheModelList(tp_num):
    return [{
        'model': item,
        'cuda_prefix': None,
        'tp_num': tp_num,
        'extra': '--enable-prefix-caching'
    } for item in get_torch_model_list(tp_num, exclude_dup=True)]


def _run_ray_distributed_test(
        config,
        model_param,
        common_case_config,
        worker_id,
        manager=None,  # ← New parameter: pass in shared manager
):
    """Universal distributed test executor (using shared Ray cluster)"""
    assert manager is not None, 'Manager instance must be provided'

    if manager.is_master:
        model_name = model_param['model']
        model_path = os.path.join(config['model_path'], model_name)

        # Start API Server for current model (master node starts/stops, worker nodes verify)
        manager.start_lmdeploy_api_server(model_path=model_path, model_param=model_param)

        try:
            run_all_step(config, common_case_config, worker_id=worker_id, port=PROXY_PORT)

        finally:
            # Clean up API Server for current model (worker nodes skip)
            manager.cleanup(force=False)
    else:
        time.sleep(10)
        ray_worker_node_wait(manager, timeout_minutes=4880)


def _run_proxy_distributed_test(
        config,
        model_param,
        common_case_config,
        worker_id,
        manager=None,  # ← New parameter: pass in shared manager
):
    """Universal distributed test executor (using shared Ray cluster)"""
    assert manager is not None, 'Manager instance must be provided'
    model_name = model_param['model']
    model_path = os.path.join(config['model_path'], model_name)

    api_server = ApiServerPerTest(proxy_manager=manager, model_path=model_path, model_param=model_param)
    api_server.start()

    try:

        if manager.is_master:
            api_server.wait_until_ready()

            run_all_step(config, common_case_config, worker_id=worker_id, port=PROXY_PORT)

        else:
            print(f'⏸️ Worker node {manager.node_rank} waiting for master to complete test...')
            proxy_worker_node_wait(manager, timeout_minutes=4880)
    finally:
        api_server.cleanup()
        if manager.is_master:
            time.sleep(1)


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.prefix_cache_test
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment', getPrefixCacheModelList(tp_num=1), indirect=True)
def test_restful_chat_pytorch_prefix_cache_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


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


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=16), indirect=True)
def test_restful_chat_tp16(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_distributed_tp16
@pytest.mark.parametrize('model_param', getModelList(tp_num=16))
def test_restful_chat_distributed_tp16(shared_ray_manager, config, model_param, common_case_config, worker_id):
    _run_ray_distributed_test(config=config,
                              model_param=model_param,
                              common_case_config=common_case_config,
                              worker_id=worker_id,
                              manager=shared_ray_manager)


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_distributed_dpep16
@pytest.mark.parametrize('model_param', getModelList({'dp': 16, 'ep': 16}))
def test_restful_chat_distributed_dpep16(shared_proxy_manager, config, model_param, common_case_config, worker_id):
    _run_proxy_distributed_test(config=config,
                                model_param=model_param,
                                common_case_config=common_case_config,
                                worker_id=worker_id,
                                manager=shared_proxy_manager)


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

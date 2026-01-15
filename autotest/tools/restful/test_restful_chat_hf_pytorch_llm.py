import os
import time

import pytest
from tools.common_case_config import (MODELSCOPE_CONFIG, PYTORCH_LORA_TEST_LLM_GPU1, PYTORCH_LORA_TEST_LLM_GPU2,
                                      PYTORCH_PR_TEST_LLM_GPU1, PYTORCH_PR_TEST_LLM_GPU2, REASONING_TEST_LLM,
                                      TOOLCALL_TEST_LLM)
from utils.config_utils import get_case_str_by_config, get_func_config_list
from utils.constant import PROXY_PORT
from utils.proxy_distributed_utils import ApiServerPerTest, proxy_worker_node_wait
from utils.ray_distributed_utils import ray_worker_node_wait
from utils.run_restful_chat import run_all_step, run_llm_test, run_reasoning_case, run_tools_case

BACKEND = 'pytorch'


def _run_ray_distributed_test(
        config,
        run_config,
        common_case_config,
        manager=None,  # ← New parameter: pass in shared manager
):
    """Universal distributed test executor (using shared Ray cluster)"""
    assert manager is not None, 'Manager instance must be provided'

    if manager.is_master:
        model_name = run_config['model']
        model_path = os.path.join(config['model_path'], model_name)

        # Start API Server for current model (master node starts/stops, worker nodes verify)
        manager.start_lmdeploy_api_server(model_path=model_path, run_config=run_config)

        try:
            case_name = get_case_str_by_config(run_config)
            run_all_step(config.get('log_path'), case_name, common_case_config, port=PROXY_PORT)

        finally:
            # Clean up API Server for current model (worker nodes skip)
            manager.cleanup(force=False)
    else:
        time.sleep(10)
        ray_worker_node_wait(manager, timeout_minutes=4880)


def _run_proxy_distributed_test(
        config,
        run_config,
        common_case_config,
        manager=None,  # ← New parameter: pass in shared manager
):
    """Universal distributed test executor (using shared Ray cluster)"""
    assert manager is not None, 'Manager instance must be provided'
    model_name = run_config['model']
    model_path = os.path.join(config['model_path'], model_name)

    api_server = ApiServerPerTest(proxy_manager=manager, model_path=model_path, run_config=run_config)
    api_server.start()

    try:

        if manager.is_master:
            api_server.wait_until_ready()
            case_name = get_case_str_by_config(run_config)
            run_all_step(config.get('log_path'), case_name, common_case_config, port=PROXY_PORT)

        else:
            print(f'⏸️ Worker node {manager.node_rank} waiting for master to complete test...')
            proxy_worker_node_wait(manager, timeout_minutes=4880)
    finally:
        api_server.cleanup()
        if manager.is_master:
            time.sleep(1)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 1}))
def test_restful_chat_tp1(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 2}))
def test_restful_chat_tp2(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 4}))
def test_restful_chat_tp4(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 8}))
def test_restful_chat_tp8(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 16}))
def test_restful_chat_tp16(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_distributed_tp16
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 16}))
def test_restful_chat_distributed_tp16(shared_ray_manager, config, run_config, common_case_config, worker_id):
    _run_ray_distributed_test(config=config,
                              run_config=run_config,
                              common_case_config=common_case_config,
                              worker_id=worker_id,
                              manager=shared_ray_manager)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api_pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_distributed_dpep16
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'dp': 16, 'ep': 16}))
def test_restful_chat_distributed_dpep16(shared_proxy_manager, config, run_config, common_case_config, worker_id):
    _run_proxy_distributed_test(config=config,
                                run_config=run_config,
                                common_case_config=common_case_config,
                                worker_id=worker_id,
                                manager=shared_proxy_manager)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config',
                         get_func_config_list(BACKEND, {'tp': 2}, extra={'enable_prefix_caching': True}),
                         indirect=True)
def test_restful_chat_turbomind_prefix_cache_tp2(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', PYTORCH_PR_TEST_LLM_GPU2)
def test_hf_turbomind_chat_pr_tp2(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', PYTORCH_PR_TEST_LLM_GPU1)
def test_hf_turbomind_chat_pr_tp1(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('run_config', [item for item in MODELSCOPE_CONFIG if item['backend'] == BACKEND],
                         indirect=True)
def test_modelscope_restful_chat_tp1(config, run_config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_llm_test(config, run_config, case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', PYTORCH_LORA_TEST_LLM_GPU1)
def test_pytorch_chat_with_lora_tp1(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', PYTORCH_LORA_TEST_LLM_GPU2)
def test_pytorch_chat_with_lora_tp2(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.parametrize(
    'run_config',
    [item for item in REASONING_TEST_LLM if item['backend'] == BACKEND and item['parallel_config']['tp'] == 1],
    indirect=True)
def test_restful_chat_reasoning_tp1(config, run_config, worker_id):
    run_reasoning_case(config, run_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.parametrize(
    'run_config',
    [item for item in REASONING_TEST_LLM if item['backend'] == BACKEND and item['parallel_config']['tp'] == 2],
    indirect=True)
def test_restful_chat_reasoning_tp2(config, run_config, worker_id):
    run_reasoning_case(config, run_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.parametrize(
    'run_config',
    [item for item in TOOLCALL_TEST_LLM if item['backend'] == BACKEND and item['parallel_config']['tp'] == 1],
    indirect=True)
def test_restful_chat_tools_tp1(config, run_config, worker_id):
    run_tools_case(config, run_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.parametrize(
    'run_config',
    [item for item in TOOLCALL_TEST_LLM if item['backend'] == BACKEND and item['parallel_config']['tp'] == 2],
    indirect=True)
def test_restful_chat_tools_tp2(config, run_config, worker_id):
    run_tools_case(config, run_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_4
@pytest.mark.parametrize(
    'run_config',
    [item for item in TOOLCALL_TEST_LLM if item['backend'] == BACKEND and item['parallel_config']['tp'] == 4],
    indirect=True)
def test_restful_chat_tools_tp4(config, run_config, worker_id):
    run_tools_case(config, run_config, worker_id)

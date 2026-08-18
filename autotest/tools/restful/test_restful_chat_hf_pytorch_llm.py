import time

import pytest
from tools.common_case_config import (
    MODELSCOPE_CONFIG,
    PYTORCH_LORA_TEST_LLM_GPU1,
    PYTORCH_LORA_TEST_LLM_GPU2,
    PYTORCH_PR_TEST_LLM_GPU1,
    PYTORCH_PR_TEST_LLM_GPU2,
)
from utils.config_utils import get_case_str_by_config, get_workerid
from utils.constant import PROXY_PORT
from utils.proxy_distributed_utils import ApiServerPerTest, proxy_worker_node_wait
from utils.pytest_layout_utils import (
    DISTRIBUTED_DP_EP_EQUAL_LAYOUTS,
    DISTRIBUTED_TP_DP_EP_LAYOUTS,
    LOCAL_TP_LAYOUTS,
    build_layout_params,
    layout_mark,
)
from utils.ray_distributed_utils import ray_worker_node_wait
from utils.run_restful_chat import run_all_step, run_llm_test

BACKEND = 'pytorch'
_PREFIX_CACHE_EXTRA = {'enable-prefix-caching': None}
_DISTRIBUTED_EXTRA_MARKS = [pytest.mark.restful_api_pytorch, pytest.mark.flaky(reruns=0)]


def _run_ray_distributed_test(config, run_config, common_case_config, manager=None):
    assert manager is not None, 'Manager instance must be provided'

    if manager.is_master:
        manager.start_lmdeploy_api_server(config=config, run_config=run_config)
        try:
            case_name = get_case_str_by_config(run_config)
            run_all_step(config.get('log_path'), case_name, common_case_config, port=PROXY_PORT)
        finally:
            manager.cleanup(force=False)
    else:
        time.sleep(10)
        ray_worker_node_wait(manager, timeout_minutes=4880)


def _run_proxy_distributed_test(config, run_config, common_case_config, manager=None):
    assert manager is not None, 'Manager instance must be provided'

    api_server = ApiServerPerTest(proxy_manager=manager, config=config, run_config=run_config)
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


def _chat_layout_marks(layout: dict[str, int]):
    marks = [pytest.mark.test_ascend]
    if layout == {'tp': 1}:
        marks.append(pytest.mark.test_3090)
    return marks


_CHAT_PARAMS = build_layout_params(
    BACKEND,
    LOCAL_TP_LAYOUTS,
    layout_extra_marks=_chat_layout_marks,
)
_PREFIX_CACHE_LOCAL_PARAMS = build_layout_params(
    BACKEND,
    LOCAL_TP_LAYOUTS[:2],
    extra=_PREFIX_CACHE_EXTRA,
    layout_extra_marks=lambda _layout: [pytest.mark.test_ascend],
)
_PROXY_PARAMS = build_layout_params(
    BACKEND,
    DISTRIBUTED_DP_EP_EQUAL_LAYOUTS + DISTRIBUTED_TP_DP_EP_LAYOUTS,
    param_marks=_DISTRIBUTED_EXTRA_MARKS,
    layout_extra_marks=lambda layout: (
        [pytest.mark.test_ascend] if layout == {'dp': 32, 'ep': 32} else []
    ),
)
_RAY_PARAMS = build_layout_params(
    BACKEND,
    ({'tp': 16},),
    param_marks=_DISTRIBUTED_EXTRA_MARKS,
    layout_extra_marks=_chat_layout_marks,
)
_PREFIX_CACHE_PROXY_PARAMS = build_layout_params(
    BACKEND,
    DISTRIBUTED_TP_DP_EP_LAYOUTS,
    extra=_PREFIX_CACHE_EXTRA,
    param_marks=_DISTRIBUTED_EXTRA_MARKS,
)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.parametrize('run_config', _CHAT_PARAMS)
def test_restful_chat(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.distributed
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.parametrize('run_config', _RAY_PARAMS)
def test_restful_chat_ray(shared_ray_manager, config, run_config, common_case_config, worker_id):
    del worker_id
    _run_ray_distributed_test(
        config=config,
        run_config=run_config,
        common_case_config=common_case_config,
        manager=shared_ray_manager,
    )


@pytest.mark.distributed
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.parametrize('run_config', _PROXY_PARAMS)
def test_restful_chat_proxy(shared_proxy_manager, config, run_config, common_case_config, worker_id):
    del worker_id
    _run_proxy_distributed_test(
        config=config,
        run_config=run_config,
        common_case_config=common_case_config,
        manager=shared_proxy_manager,
    )


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.parametrize('run_config', _PREFIX_CACHE_LOCAL_PARAMS)
def test_restful_chat_prefix_cache_local(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.distributed
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.parametrize('run_config', _PREFIX_CACHE_PROXY_PARAMS)
def test_restful_chat_prefix_cache_proxy(
        shared_proxy_manager, config, run_config, common_case_config, worker_id):
    del worker_id
    _run_proxy_distributed_test(
        config=config,
        run_config=run_config,
        common_case_config=common_case_config,
        manager=shared_proxy_manager,
    )


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pr_test
@layout_mark({'tp': 2})
@pytest.mark.parametrize('run_config', PYTORCH_PR_TEST_LLM_GPU2)
def test_hf_pytorch_chat_pr_tp2(config, run_config, common_case_config, worker_id):
    worker_id = 'gw' + str(3 + get_workerid(worker_id))
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pr_test
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', PYTORCH_PR_TEST_LLM_GPU1)
def test_hf_pytorch_chat_pr_tp1(config, run_config, common_case_config, worker_id):
    worker_id = 'gw' + str(6 + get_workerid(worker_id))
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', [item for item in MODELSCOPE_CONFIG if item['backend'] == BACKEND])
def test_modelscope_restful_chat_tp1(config, run_config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_llm_test(config, run_config, case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', PYTORCH_LORA_TEST_LLM_GPU1)
def test_pytorch_chat_with_lora_tp1(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@layout_mark({'tp': 2})
@pytest.mark.parametrize('run_config', PYTORCH_LORA_TEST_LLM_GPU2)
def test_pytorch_chat_with_lora_tp2(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)

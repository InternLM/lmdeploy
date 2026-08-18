import os
import time

import pytest
import utils.constant as constant
from utils.config_utils import (
    get_case_str_by_config,
    get_eval_preset_config,
    get_model_path_from_config,
    get_workerid,
    resolve_eval_config_name,
)
from utils.evaluate_utils import build_eval_judge_run_config, eval_test
from utils.proxy_distributed_utils import ApiServerPerTest, proxy_worker_node_wait
from utils.pytest_layout_utils import (
    DISTRIBUTED_CP_TP_LAYOUTS,
    DISTRIBUTED_DP_EP_EQUAL_LAYOUTS,
    DISTRIBUTED_TP_DP_EP_LAYOUTS,
    LOCAL_TP_LAYOUTS,
    build_eval_longtext_params,
    build_eval_stage_params,
)
from utils.ray_distributed_utils import ray_worker_node_wait
from utils.run_restful_chat import start_openai_service, start_proxy_server, stop_restful_api, terminate_restful_api

TURBOMIND_LOCAL_LAYOUTS = LOCAL_TP_LAYOUTS[:4]
PYTORCH_LOCAL_LAYOUTS = LOCAL_TP_LAYOUTS
_PREFIX_CACHE_EXTRA = {'enable-prefix-caching': True}


def _pytorch_ascend_marks(_layout: dict[str, int]):
    return [pytest.mark.test_ascend]


def _run_ray_distributed_test(
        config,
        run_config,
        worker_id,
        test_type='infer',
        manager=None,
        eval_config_name='default'):
    """Universal distributed test executor (using shared Ray cluster)"""
    assert manager is not None, 'Manager instance must be provided'
    eval_config_name = resolve_eval_config_name(config, run_config, eval_config_name)

    preset_config = get_eval_preset_config(config, run_config, eval_config_name)

    if manager.is_master:
        model_path = get_model_path_from_config(config, run_config['model'])
        eval_path = config.get('eval_path')

        manager.start_lmdeploy_api_server(config=config, run_config=run_config)

        try:
            print(f'🧪 Master node executing {test_type} test ({eval_config_name})...')
            case_name = get_case_str_by_config(run_config)

            result, msg = eval_test(model_path,
                                    eval_path,
                                    case_name,
                                    port=constant.PROXY_PORT,
                                    test_type=test_type,
                                    **preset_config)
            assert result, f'❌ {test_type} test failed: {msg}'
            print(f'✅ {test_type} test passed')

        finally:
            manager.cleanup(force=False)
    else:
        time.sleep(10)
        ray_worker_node_wait(manager, timeout_minutes=4880)


def _run_proxy_distributed_test(config,
                                run_config,
                                worker_id,
                                test_type='infer',
                                manager=None,
                                eval_config_name='default',
                                eval_subpath=None):
    assert manager is not None, 'Manager instance must be provided'

    if eval_subpath is None:
        eval_config_name = resolve_eval_config_name(config, run_config, eval_config_name)

    preset_config = get_eval_preset_config(config, run_config, eval_config_name)
    model_name = run_config['model']
    model_path = get_model_path_from_config(config, model_name)

    api_server = ApiServerPerTest(proxy_manager=manager, config=config, run_config=run_config)
    api_server.start()

    try:
        if manager.is_master:
            api_server.wait_until_ready()
            print(f'🧪 Master node executing {test_type} test ({eval_config_name})...')
            eval_path = config.get('eval_path')
            if eval_subpath:
                eval_path = os.path.join(eval_path, eval_subpath)
                os.makedirs(eval_path, exist_ok=True)
            case_name = get_case_str_by_config(run_config)

            extra_config = {'max-num-workers': 16}

            result, msg = eval_test(model_path,
                                    eval_path,
                                    case_name,
                                    port=constant.PROXY_PORT,
                                    test_type=test_type,
                                    extra_config=extra_config,
                                    eval_config_name=eval_config_name,
                                    **preset_config)
            assert result, f'❌ {test_type} test failed: {msg}'
            print(f'✅ {test_type} test passed')

        else:
            print(f'⏸️ Worker node {manager.node_rank} waiting for master to complete test...')
            proxy_worker_node_wait(manager, timeout_minutes=4880)

    finally:
        api_server.cleanup()
        if manager.is_master:
            time.sleep(1)


def run_eval_test(config, run_config, worker_id, test_type='infer', eval_config_name='default', eval_subpath=None):
    """Run test with specified evaluation configuration."""
    eval_config_name = resolve_eval_config_name(config, run_config, eval_config_name)
    preset_config = get_eval_preset_config(config, run_config, eval_config_name)
    eval_path = config.get('eval_path')
    if eval_subpath:
        eval_path = os.path.join(eval_path, eval_subpath)
        os.makedirs(eval_path, exist_ok=True)

    total_gpus = int(os.environ.get('TOTAL_GPU_COUNT', '8'))
    work_num = int(total_gpus / run_config.get('parallel_config', {}).get('tp', 1))

    extra_config = {'max-num-workers': min(work_num * 16, 64)}

    case_name = get_case_str_by_config(run_config)

    if test_type == 'infer':
        proxy_pid, proxy_process = start_proxy_server(config.get('server_log_path'), constant.PROXY_PORT,
                                                      f'{case_name}_infer')
        run_config_new = run_config.copy()
        if 'extra_params' not in run_config_new:
            run_config_new['extra_params'] = {}
        run_config_new['extra_params']['proxy-url'] = f'http://{constant.DEFAULT_SERVER}:{constant.PROXY_PORT}'
        run_config_new['extra_params']['server-name'] = constant.DEFAULT_SERVER

        from concurrent.futures import ThreadPoolExecutor

        def run_openai_service_start(i):
            return start_openai_service(config, run_config_new, f'gw{i}')

        with ThreadPoolExecutor(max_workers=work_num) as executor:
            futures = [executor.submit(run_openai_service_start, i) for i in range(int(work_num))]
        results = []
        for future in futures:
            pid, content = future.result()
            results.append((pid, content))

        try:
            model_path = get_model_path_from_config(config, run_config.get('model'))
            eval_test(model_path,
                      eval_path,
                      case_name,
                      port=constant.PROXY_PORT,
                      test_type=test_type,
                      extra_config=extra_config,
                      eval_config_name=eval_config_name,
                      **preset_config)
        finally:
            for i in range(work_num):
                terminate_restful_api(f'gw{i}')
            stop_restful_api(proxy_pid, proxy_process)
    else:
        model_path = get_model_path_from_config(config, run_config.get('model'))
        if eval_config_name in ('longtext-256k', 'longtext-512k'):
            eval_test(model_path,
                      eval_path,
                      case_name,
                      port=constant.PROXY_PORT,
                      test_type=test_type,
                      extra_config=extra_config,
                      eval_config_name=eval_config_name,
                      **preset_config)
            return

        port = constant.PROXY_PORT + get_workerid(worker_id)
        proxy_pid, proxy_process = start_proxy_server(config.get('server_log_path'), port, f'{case_name}_eval')
        eval_run_config = build_eval_judge_run_config(
            config, f'http://{constant.DEFAULT_SERVER}:{port}')

        pid, content = start_openai_service(config, eval_run_config, worker_id)
        try:
            if pid > 0:
                model_path = get_model_path_from_config(config, eval_run_config.get('model'))
                eval_test(model_path,
                          eval_path,
                          case_name,
                          port=port,
                          test_type=test_type,
                          extra_config=extra_config,
                          eval_config_name=eval_config_name,
                          **preset_config)
            else:
                assert False, f'Failed to start RESTful API server: {content}'
        finally:
            if pid > 0:
                terminate_restful_api(worker_id)
            stop_restful_api(proxy_pid, proxy_process)


_LOCAL_EVAL_PARAMS = (
    build_eval_stage_params('turbomind', TURBOMIND_LOCAL_LAYOUTS)
    + build_eval_stage_params(
        'turbomind',
        DISTRIBUTED_CP_TP_LAYOUTS + DISTRIBUTED_TP_DP_EP_LAYOUTS,
    )
    + build_eval_stage_params(
        'pytorch',
        PYTORCH_LOCAL_LAYOUTS,
        layout_extra_marks=_pytorch_ascend_marks,
    )
    + build_eval_stage_params(
        'pytorch',
        DISTRIBUTED_DP_EP_EQUAL_LAYOUTS + DISTRIBUTED_TP_DP_EP_LAYOUTS,
        test_types=('eval',),
        layout_extra_marks=_pytorch_ascend_marks,
    )
)

_PROXY_INFER_PARAMS = build_eval_stage_params(
    'pytorch',
    DISTRIBUTED_DP_EP_EQUAL_LAYOUTS + DISTRIBUTED_TP_DP_EP_LAYOUTS,
    test_types=('infer',),
    layout_extra_marks=lambda layout: (
        [pytest.mark.test_ascend] if layout == {'dp': 32, 'ep': 32} else []
    ),
)

_RAY_INFER_PARAMS = build_eval_stage_params(
    'pytorch',
    ({'tp': 16},),
    test_types=('infer',),
    layout_extra_marks=_pytorch_ascend_marks,
)

_LONGTEXT_LOCAL_PARAMS = (
    build_eval_longtext_params(
        'pytorch',
        {'tp': 2},
        session_len=400000,
        eval_config_name='longtext-256k',
        eval_subpath='longtext',
    )
    + build_eval_longtext_params(
        'pytorch',
        {'tp': 2},
        session_len=700000,
        eval_config_name='longtext-512k',
        eval_subpath='longtext-512k',
    )
    + build_eval_longtext_params(
        'pytorch',
        {'tp': 2, 'dp': 4, 'ep': 8},
        session_len=400000,
        eval_config_name='longtext-256k',
        eval_subpath='longtext',
        test_types=('eval',),
    )
    + build_eval_longtext_params(
        'pytorch',
        {'tp': 2, 'dp': 4, 'ep': 8},
        session_len=700000,
        eval_config_name='longtext-512k',
        eval_subpath='longtext-512k',
        test_types=('eval',),
    )
)

_LONGTEXT_PROXY_INFER_PARAMS = (
    build_eval_longtext_params(
        'pytorch',
        {'tp': 2, 'dp': 4, 'ep': 8},
        session_len=400000,
        eval_config_name='longtext-256k',
        eval_subpath='longtext',
        test_types=('infer',),
        use_proxy=True,
    )
    + build_eval_longtext_params(
        'pytorch',
        {'tp': 2, 'dp': 4, 'ep': 8},
        session_len=700000,
        eval_config_name='longtext-512k',
        eval_subpath='longtext-512k',
        test_types=('infer',),
        use_proxy=True,
    )
)

_PREFIX_CACHE_LOCAL_PARAMS = (
    build_eval_stage_params(
        'pytorch',
        PYTORCH_LOCAL_LAYOUTS[:2],
        extra=_PREFIX_CACHE_EXTRA,
    )
    + build_eval_stage_params(
        'pytorch',
        DISTRIBUTED_TP_DP_EP_LAYOUTS,
        test_types=('eval',),
        extra=_PREFIX_CACHE_EXTRA,
    )
    + build_eval_stage_params(
        'turbomind',
        DISTRIBUTED_TP_DP_EP_LAYOUTS,
        extra=_PREFIX_CACHE_EXTRA,
    )
)

_PREFIX_CACHE_PROXY_INFER_PARAMS = build_eval_stage_params(
    'pytorch',
    DISTRIBUTED_TP_DP_EP_LAYOUTS,
    test_types=('infer',),
    extra=_PREFIX_CACHE_EXTRA,
)


@pytest.mark.parametrize('test_type, run_config', _LOCAL_EVAL_PARAMS)
def test_api_evaluate_local(config, run_config, worker_id, test_type):
    run_eval_test(config, run_config, worker_id, test_type)


@pytest.mark.distributed
@pytest.mark.parametrize('test_type, run_config', _PROXY_INFER_PARAMS)
def test_api_evaluate_proxy_infer(shared_proxy_manager, config, run_config, worker_id, test_type):
    _run_proxy_distributed_test(
        config=config,
        run_config=run_config,
        worker_id=worker_id,
        test_type=test_type,
        manager=shared_proxy_manager,
    )


@pytest.mark.distributed
@pytest.mark.parametrize('test_type, run_config', _RAY_INFER_PARAMS)
def test_api_evaluate_ray_infer(shared_ray_manager, config, run_config, worker_id, test_type):
    _run_ray_distributed_test(
        config=config,
        run_config=run_config,
        worker_id=worker_id,
        test_type=test_type,
        manager=shared_ray_manager,
    )


@pytest.mark.parametrize(
    'test_type, run_config, eval_config_name, eval_subpath',
    _LONGTEXT_LOCAL_PARAMS,
)
def test_api_evaluate_longtext_local(
        config, run_config, worker_id, test_type, eval_config_name, eval_subpath):
    run_eval_test(
        config,
        run_config,
        worker_id,
        test_type,
        eval_config_name=eval_config_name,
        eval_subpath=eval_subpath,
    )


@pytest.mark.distributed
@pytest.mark.parametrize(
    'test_type, run_config, eval_config_name, eval_subpath',
    _LONGTEXT_PROXY_INFER_PARAMS,
)
def test_api_evaluate_longtext_proxy_infer(
        shared_proxy_manager,
        config,
        run_config,
        worker_id,
        test_type,
        eval_config_name,
        eval_subpath):
    _run_proxy_distributed_test(
        config=config,
        run_config=run_config,
        worker_id=worker_id,
        test_type=test_type,
        manager=shared_proxy_manager,
        eval_config_name=eval_config_name,
        eval_subpath=eval_subpath,
    )


@pytest.mark.parametrize('test_type, run_config', _PREFIX_CACHE_LOCAL_PARAMS)
def test_api_evaluate_prefix_cache_local(config, run_config, worker_id, test_type):
    run_eval_test(config, run_config, worker_id, test_type, eval_subpath='prefix_cache')


@pytest.mark.distributed
@pytest.mark.parametrize('test_type, run_config', _PREFIX_CACHE_PROXY_INFER_PARAMS)
def test_api_evaluate_prefix_cache_proxy_infer(
        shared_proxy_manager, config, run_config, worker_id, test_type):
    _run_proxy_distributed_test(
        config=config,
        run_config=run_config,
        worker_id=worker_id,
        test_type=test_type,
        manager=shared_proxy_manager,
        eval_subpath='prefix_cache',
    )

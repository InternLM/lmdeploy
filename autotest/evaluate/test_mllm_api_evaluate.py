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
from utils.evaluate_utils import build_eval_judge_run_config, mllm_eval_test
from utils.proxy_distributed_utils import ApiServerPerTest, proxy_worker_node_wait
from utils.pytest_layout_utils import (
    DISTRIBUTED_TP_DP_EP_LAYOUTS,
    LOCAL_TP_LAYOUTS,
    build_eval_stage_params,
)
from utils.run_restful_chat import start_openai_service, start_proxy_server, stop_restful_api, terminate_restful_api

TURBOMIND_LOCAL_LAYOUTS = LOCAL_TP_LAYOUTS[:4]
_MLLM_EXTRA = {'session-len': 65536, 'cache-max-entry-count': 0.6}


def _pytorch_ascend_marks(_layout: dict[str, int]):
    return [pytest.mark.test_ascend]


def run_eval_test(config, run_config, worker_id, test_type='infer', eval_config_name='default', eval_subpath=None):
    eval_config_name = resolve_eval_config_name(config, run_config, eval_config_name)
    extra_config = get_eval_preset_config(config, run_config, eval_config_name, mllm=True)
    eval_path = config.get('mllm_eval_path')
    if eval_subpath:
        eval_path = os.path.join(eval_path, eval_subpath)
        os.makedirs(eval_path, exist_ok=True)
    case_name = get_case_str_by_config(run_config)
    if test_type == 'infer':
        proxy_pid, proxy_process = start_proxy_server(config.get('server_log_path'), constant.PROXY_PORT,
                                                      f'{case_name}_infer')
        total_gpus = int(os.environ.get('TOTAL_GPU_COUNT', '8'))
        work_num = int(total_gpus / run_config.get('parallel_config', {}).get('tp', 1))
        run_config_new = run_config.copy()
        if 'extra_params' not in run_config_new:
            run_config_new['extra_params'] = {}
        run_config_new['extra_params']['proxy-url'] = f'http://{constant.DEFAULT_SERVER}:{constant.PROXY_PORT}'

        from concurrent.futures import ThreadPoolExecutor

        def run_openai_service_start(i):
            return start_openai_service(config, run_config_new, f'gw{i}')

        with ThreadPoolExecutor(max_workers=work_num) as executor:
            futures = [executor.submit(run_openai_service_start, i) for i in range(int(work_num))]
        for future in futures:
            future.result()

        try:
            model_path = get_model_path_from_config(config, run_config.get('model'))
            extra_config['api-nproc'] = work_num * 16
            mllm_eval_test(model_path,
                           eval_path,
                           case_name,
                           port=constant.PROXY_PORT,
                           test_type=test_type,
                           extra_config=extra_config)
        finally:
            for i in range(work_num):
                terminate_restful_api(f'gw{i}')
            stop_restful_api(proxy_pid, proxy_process)
    else:
        port = constant.PROXY_PORT + get_workerid(worker_id)
        proxy_pid, proxy_process = start_proxy_server(config.get('server_log_path'), port, f'{case_name}_eval')
        eval_run_config = build_eval_judge_run_config(
            config, f'http://{constant.DEFAULT_SERVER}:{port}')
        pid, content = start_openai_service(config, eval_run_config, worker_id)
        try:
            if pid > 0:
                model_path = get_model_path_from_config(config, eval_run_config.get('model'))
                mllm_eval_test(model_path, eval_path, case_name, port=port, test_type=test_type)
            else:
                assert False, f'Failed to start RESTful API server: {content}'
        finally:
            if pid > 0:
                terminate_restful_api(worker_id)
            stop_restful_api(proxy_pid, proxy_process)


def _run_proxy_distributed_mllm_test(
        config,
        run_config,
        worker_id,
        test_type='infer',
        manager=None,
        eval_config_name='default'):
    assert manager is not None, 'Manager instance must be provided'

    eval_config_name = resolve_eval_config_name(config, run_config, eval_config_name)

    preset_config = get_eval_preset_config(config, run_config, eval_config_name, mllm=True)
    model_name = run_config['model']
    model_path = get_model_path_from_config(config, model_name)

    api_server = ApiServerPerTest(proxy_manager=manager, config=config, run_config=run_config)
    api_server.start()

    try:
        if manager.is_master:
            api_server.wait_until_ready()
            print(f'🧪 Master node executing mllm {test_type} test ({eval_config_name})...')
            eval_path = config.get('mllm_eval_path')
            case_name = get_case_str_by_config(run_config)
            extra_config = {'api-nproc': 16}
            extra_config.update(preset_config)

            result, msg = mllm_eval_test(model_path,
                                         eval_path,
                                         case_name,
                                         port=constant.PROXY_PORT,
                                         test_type=test_type,
                                         extra_config=extra_config)
            assert result, f'❌ mllm {test_type} test failed: {msg}'
            print(f'✅ mllm {test_type} test passed')

        else:
            print(f'⏸️ Worker node {manager.node_rank} waiting for master to complete mllm test...')
            proxy_worker_node_wait(manager, timeout_minutes=4880)

    finally:
        api_server.cleanup()
        if manager.is_master:
            time.sleep(1)


_LOCAL_EVAL_PARAMS = (
    build_eval_stage_params(
        'turbomind',
        TURBOMIND_LOCAL_LAYOUTS,
        model_type='vl_model',
        func_type='mllm_evaluate',
        extra=_MLLM_EXTRA,
    )
    + build_eval_stage_params(
        'pytorch',
        LOCAL_TP_LAYOUTS,
        model_type='vl_model',
        func_type='mllm_evaluate',
        extra=_MLLM_EXTRA,
        layout_extra_marks=_pytorch_ascend_marks,
    )
    + build_eval_stage_params(
        'pytorch',
        DISTRIBUTED_TP_DP_EP_LAYOUTS,
        test_types=('eval',),
        model_type='vl_model',
        func_type='mllm_evaluate',
        extra=_MLLM_EXTRA,
        layout_extra_marks=_pytorch_ascend_marks,
    )
    + build_eval_stage_params(
        'turbomind',
        DISTRIBUTED_TP_DP_EP_LAYOUTS,
        test_types=('eval',),
        model_type='vl_model',
        func_type='mllm_evaluate',
        extra=_MLLM_EXTRA,
    )
)

_PROXY_INFER_PARAMS = (
    build_eval_stage_params(
        'pytorch',
        DISTRIBUTED_TP_DP_EP_LAYOUTS,
        test_types=('infer',),
        model_type='vl_model',
        func_type='mllm_evaluate',
        extra=_MLLM_EXTRA,
        layout_extra_marks=_pytorch_ascend_marks,
    )
    + build_eval_stage_params(
        'turbomind',
        DISTRIBUTED_TP_DP_EP_LAYOUTS,
        test_types=('infer',),
        model_type='vl_model',
        func_type='mllm_evaluate',
        extra=_MLLM_EXTRA,
    )
)


@pytest.mark.parametrize('test_type, run_config', _LOCAL_EVAL_PARAMS)
def test_mllm_api_evaluate_local(config, run_config, worker_id, test_type):
    run_eval_test(config, run_config, worker_id, test_type)


@pytest.mark.distributed
@pytest.mark.parametrize('test_type, run_config', _PROXY_INFER_PARAMS)
def test_mllm_api_evaluate_proxy_infer(
        shared_proxy_manager, config, run_config, worker_id, test_type):
    _run_proxy_distributed_mllm_test(
        config=config,
        run_config=run_config,
        worker_id=worker_id,
        test_type=test_type,
        manager=shared_proxy_manager,
    )

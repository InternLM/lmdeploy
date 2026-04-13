import copy
import os
import time

import pytest
import utils.constant as constant
from utils.config_utils import get_case_str_by_config, get_func_config_list, get_workerid
from utils.evaluate_utils import eval_test
from utils.proxy_distributed_utils import ApiServerPerTest, proxy_worker_node_wait
from utils.ray_distributed_utils import ray_worker_node_wait
from utils.run_restful_chat import start_openai_service, start_proxy_server, stop_restful_api, terminate_restful_api


def _run_ray_distributed_test(
        config,
        run_config,
        worker_id,
        test_type='infer',
        manager=None,  # ← New parameter: pass in shared manager
        eval_config_name='default'):
    """Universal distributed test executor (using shared Ray cluster)"""
    assert manager is not None, 'Manager instance must be provided'
    if 'gpt' in run_config.get('model', '').lower():
        eval_config_name = 'gpt'
    elif 'intern-s1-pro' in run_config.get('model', '').lower():
        eval_config_name = 'intern-s1-pro'
    elif 'qwen3.5' in run_config.get('model', '').lower():
        eval_config_name = 'qwen3.5'
    if str(config.get('env_tag')) == 'ascend':
        eval_config_name = f'{eval_config_name}-2batch'

    preset_config = constant.EVAL_CONFIGS.get(eval_config_name, {})

    if manager.is_master:
        model_path = os.path.join(config['model_path'], run_config['model'])
        eval_path = config.get('eval_path')

        # Start API Server for current model (master node starts/stops, worker nodes verify)
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
            # Clean up API Server for current model (worker nodes skip)
            manager.cleanup(force=False)
    else:
        time.sleep(10)
        ray_worker_node_wait(manager, timeout_minutes=4880)


def _run_proxy_distributed_test(config,
                                run_config,
                                worker_id,
                                test_type='infer',
                                manager=None,
                                eval_config_name='default'):
    assert manager is not None, 'Manager instance must be provided'

    if 'gpt' in run_config.get('model', '').lower():
        eval_config_name = 'gpt'
    elif 'intern-s1-pro' in run_config.get('model', '').lower():
        eval_config_name = 'intern-s1-pro'
    elif 'qwen3.5' in run_config.get('model', '').lower():
        eval_config_name = 'qwen3.5'

    if str(config.get('env_tag')) == 'ascend':
        eval_config_name = f'{eval_config_name}-2batch'

    preset_config = constant.EVAL_CONFIGS.get(eval_config_name, {})
    model_name = run_config['model']
    model_path = os.path.join(config['model_path'], model_name)

    api_server = ApiServerPerTest(proxy_manager=manager, config=config, run_config=run_config)
    api_server.start()

    try:
        if manager.is_master:
            api_server.wait_until_ready()
            print(f'🧪 Master node executing {test_type} test ({eval_config_name})...')
            eval_path = config.get('eval_path')
            case_name = get_case_str_by_config(run_config)

            extra_config = {'max-num-workers': 16}

            result, msg = eval_test(model_path,
                                    eval_path,
                                    case_name,
                                    port=constant.PROXY_PORT,
                                    test_type=test_type,
                                    extra_config=extra_config,
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
    if eval_config_name == 'default':
        longtext_key = run_config.get('_longtext_eval_config_name')
        if longtext_key:
            eval_config_name = longtext_key
        else:
            if 'gpt' in run_config.get('model', '').lower():
                eval_config_name = 'gpt'
            elif 'sdar' in run_config.get('model', '').lower():
                eval_config_name = 'sdar'
            elif 'intern-s1-pro' in run_config.get('model', '').lower():
                eval_config_name = 'intern-s1-pro'
            elif 'qwen3.5' in run_config.get('model', '').lower():
                eval_config_name = 'qwen3.5'
            if str(config.get('env_tag')) == 'a100':
                eval_config_name = f'{eval_config_name}-32k'
            elif str(config.get('env_tag')) == 'ascend':
                eval_config_name = f'{eval_config_name}-2batch'
    preset_config = constant.EVAL_CONFIGS.get(eval_config_name, {})
    eval_path = config.get('eval_path')
    if eval_subpath:
        rel = eval_subpath
        nested = run_config.get('_eval_path_subdir')
        if nested:
            rel = os.path.join(rel, nested)
        eval_path = os.path.join(eval_path, rel)
        os.makedirs(eval_path, exist_ok=True)

    total_gpus = int(os.environ.get('TOTAL_GPU_COUNT', '8'))
    work_num = int(total_gpus / run_config.get('parallel_config', {}).get('tp', 1))

    # Set max-num-workers to 8 for qwen3.5 models
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
            model_path = os.path.join(config.get('model_path'), run_config.get('model'))
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
    else:  # eval
        port = constant.PROXY_PORT + get_workerid(worker_id)
        proxy_pid, proxy_process = start_proxy_server(config.get('server_log_path'), port, f'{case_name}_eval')
        eval_run_config = constant.EVAL_RUN_CONFIG.copy()
        if 'extra_params' not in eval_run_config:
            eval_run_config['extra_params'] = {}
        eval_run_config['extra_params']['proxy-url'] = f'http://{constant.DEFAULT_SERVER}:{port}'

        pid, content = start_openai_service(config, eval_run_config, worker_id)
        try:
            if pid > 0:
                model_path = os.path.join(config.get('model_path'), eval_run_config.get('model'))
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


def get_models(backend, parallel_config, session_len='auto'):
    if session_len == 'auto':
        configs = get_func_config_list(backend, parallel_config, func_type='evaluate', extra={})
        result = []
        for config in configs:
            model = config.get('model', '')
            if 'Qwen3.5' not in model:
                if 'extra_params' not in config:
                    config['extra_params'] = {}
                config['extra_params']['session_len'] = 65536
            result.append(config)
        return result
    else:
        extra = {'session_len': session_len} if session_len is not None else {}
        return get_func_config_list(backend, parallel_config, func_type='evaluate', extra=extra)


def _resolve_longtext_eval_config_name(run_config: dict) -> str | None:
    """Map longtext_evaluate config to EVAL_CONFIGS key; add branches when new
    longtext families ship."""
    ep = run_config.get('extra_params') or {}
    raw = ep.get('session_len', ep.get('session-len'))
    if raw is None:
        return None
    try:
        sl = int(raw)
    except (TypeError, ValueError):
        return None
    model_lower = (run_config.get('model') or '').lower()
    if 'qwen3.5' in model_lower:
        if sl >= 600000:
            return 'longtext-512k'
        if sl >= 300000:
            return 'longtext-256k'
    return None


def get_longtext_models(backend, parallel_config, session_len='auto'):
    if session_len == 'auto':
        session_len = 65536
    extra = {'session_len': session_len} if session_len is not None else {}
    configs = get_func_config_list(backend, parallel_config, func_type='longtext_evaluate', extra=extra)
    for cfg in configs:
        preset_key = _resolve_longtext_eval_config_name(cfg)
        if preset_key:
            cfg['_longtext_eval_config_name'] = preset_key
    return configs


def get_mtp_models(backend, parallel_config):
    base_configs = get_func_config_list(backend, parallel_config, func_type='mtp_evaluate', extra={})
    for cfg in base_configs:
        if 'qwen3.5' in cfg.get('model', '').lower():
            cfg['extra_params'].update(constant.QWEN35_MTP_SERVER_EXTRA)

    result_configs = []
    for config in base_configs:
        result_configs.append(config)

        if config.get('model') == 'Qwen/Qwen3.5-35B-A3B' and parallel_config.get('tp') == 2:
            fp8_config = copy.deepcopy(config)
            fp8_config['extra_params']['max-prefill-token-num'] = 1024
            fp8_config['extra_params']['model-format'] = 'fp8'
            fp8_config['_eval_path_subdir'] = 'serve_fp8'
            result_configs.append(fp8_config)

    return result_configs


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('turbomind', {'tp': 1}))
def test_turbomind_infer_tp1(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('turbomind', {'tp': 2}))
def test_turbomind_infer_tp2(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('turbomind', {'tp': 4}))
def test_turbomind_infer_tp4(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('turbomind', {'tp': 8}))
def test_turbomind_infer_tp8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.turbomind
@pytest.mark.gpu_num_distributed_cp2tp8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('turbomind', {'cp': 2, 'tp': 8}))
def test_turbomind_infer_cp2tp8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'tp': 1}))
def test_pytorch_restful_tp1(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'tp': 2}))
def test_pytorch_restful_tp2(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_longtext_models('pytorch', {'tp': 2}, session_len=400000))
def test_pytorch_restful_tp2_longtext(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_longtext_models('pytorch', {'tp': 2}, session_len=700000))
def test_pytorch_restful_tp2_longtext_512k(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.mtp
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_mtp_models('pytorch', {'tp': 2}))
def test_pytorch_restful_tp2_mtp(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer', eval_subpath='mtp')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.mtp
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_mtp_models('pytorch', {'tp': 1}))
def test_pytorch_restful_tp1_mtp(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer', eval_subpath='mtp')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.mtp
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_mtp_models('pytorch', {'tp': 2}))
def test_pytorch_eval_tp2_mtp(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval', eval_subpath='mtp')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.mtp
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_mtp_models('pytorch', {'tp': 1}))
def test_pytorch_eval_tp1_mtp(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval', eval_subpath='mtp')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'tp': 4}))
def test_pytorch_restful_tp4(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'tp': 8}))
def test_pytorch_restful_tp8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'tp': 16}))
def test_pytorch_restful_tp16(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'infer')


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_tp16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'tp': 16}))
def test_pytorch_restful_distributed_tp16(shared_ray_manager, config, run_config, worker_id):
    _run_ray_distributed_test(config=config,
                              run_config=run_config,
                              worker_id=worker_id,
                              test_type='infer',
                              manager=shared_ray_manager)


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_dpep8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'dp': 8, 'ep': 8}))
def test_pytorch_restful_distributed_dpep8(shared_proxy_manager, config, run_config, worker_id):
    _run_proxy_distributed_test(config=config,
                                run_config=run_config,
                                worker_id=worker_id,
                                test_type='infer',
                                manager=shared_proxy_manager)


@pytest.mark.infer
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_dpep16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'dp': 16, 'ep': 16}))
def test_pytorch_restful_distributed_dpep16(shared_proxy_manager, config, run_config, worker_id):
    _run_proxy_distributed_test(config=config,
                                run_config=run_config,
                                worker_id=worker_id,
                                test_type='infer',
                                manager=shared_proxy_manager)


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('turbomind', {'tp': 1}))
def test_turbomind_eval_tp1(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('turbomind', {'tp': 2}))
def test_turbomind_eval_tp2(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('turbomind', {'tp': 4}))
def test_turbomind_eval_tp4(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('turbomind', {'tp': 8}))
def test_turbomind_eval_tp8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'tp': 1}))
def test_pytorch_eval_tp1(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'tp': 2}))
def test_pytorch_eval_tp2(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'tp': 4}))
def test_pytorch_eval_tp4(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'tp': 8}))
def test_pytorch_eval_tp8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'tp': 16}))
def test_pytorch_eval_tp16(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_tp16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'tp': 16}))
def test_pytorch_eval_distributed_tp16(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_dpep8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'dp': 8, 'ep': 8}))
def test_pytorch_eval_distributed_dpep8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_dpep16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('pytorch', {'dp': 16, 'ep': 16}))
def test_pytorch_eval_distributed_dpep16(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_longtext_models('pytorch', {'tp': 2}, session_len=400000))
def test_pytorch_eval_tp2_longtext(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_longtext_models('pytorch', {'tp': 2}, session_len=700000))
def test_pytorch_eval_tp2_longtext_512k(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')


@pytest.mark.eval
@pytest.mark.turbomind
@pytest.mark.gpu_num_distributed_cp2tp8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models('turbomind', {'cp': 2, 'tp': 8}))
def test_turbomind_eval_cp2tp8(config, run_config, worker_id):
    run_eval_test(config, run_config, worker_id, 'eval')

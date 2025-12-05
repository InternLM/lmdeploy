import os

import pytest
from utils.benchmark_utils import throughput_test
from utils.config_utils import get_benchmark_model_list, get_cuda_id_by_workerid, get_cuda_prefix_by_workerid


@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_benchmark_model_list(parallel_config=1, kvint_list=[4, 8]))
def test_throughput_tp1(config, run_id, run_config, worker_id):
    result, msg = throughput_test(config,
                                  run_id,
                                  run_config,
                                  cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=1),
                                  worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_benchmark_model_list(parallel_config=2, kvint_list=[4, 8]))
def test_throughput_tp2(config, run_id, run_config, worker_id):
    result, msg = throughput_test(config,
                                  run_id,
                                  run_config,
                                  cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2),
                                  worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_benchmark_model_list(parallel_config=4, kvint_list=[4, 8]))
def test_throughput_tp4(config, run_id, run_config, worker_id):
    result, msg = throughput_test(config,
                                  run_id,
                                  run_config,
                                  cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=4),
                                  worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_benchmark_model_list(parallel_config=4, kvint_list=[4, 8]))
def test_throughput_tp8(config, run_id, run_config, worker_id):
    result, msg = throughput_test(config,
                                  run_id,
                                  run_config,
                                  cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=8),
                                  worker_id=worker_id)

    assert result, msg


@pytest.mark.function
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', [{
    'model': 'internlm/internlm2_5-20b-chat',
    'backend': 'pytorch',
    'tp_num': 2
}, {
    'model': 'internlm/internlm2_5-20b-chat-inner-4bits',
    'backend': 'turbomind',
    'quant_policy': 0,
    'tp_num': 2
}])
def test_throughput_func_tp2(config, run_id, run_config, worker_id):
    result, msg = throughput_test(config,
                                  run_id,
                                  run_config,
                                  cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2),
                                  worker_id=worker_id,
                                  is_smoke=True)

    assert result, msg


@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', [{
    'model': 'meta-llama/Meta-Llama-3-1-8B-Instruct',
    'backend': 'pytorch',
    'tp_num': 1
}, {
    'model': 'meta-llama/Meta-Llama-3-1-8B-Instruct',
    'backend': 'turbomind',
    'quant_policy': 0,
    'tp_num': 1
}])
def test_throughput_prtest_tp1(config, run_id, run_config, worker_id):
    device_type = os.environ.get('DEVICE', 'cuda')
    if device_type == 'ascend':
        env_var = 'ASCEND_RT_VISIBLE_DEVICES='
    else:
        env_var = 'CUDA_VISIBLE_DEVICES='
    result, msg = throughput_test(config,
                                  run_id,
                                  run_config,
                                  cuda_prefix=f'{env_var}' + str(int(get_cuda_id_by_workerid(worker_id)) + 5),
                                  worker_id=worker_id,
                                  is_smoke=True)

    assert result, msg

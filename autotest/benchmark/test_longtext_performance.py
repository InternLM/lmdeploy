import pytest
from utils.benchmark_utils import longtext_throughput_test
from utils.config_utils import get_benchmark_model_list, get_cuda_prefix_by_workerid


@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_benchmark_model_list(parallel_config=1, kvint_list=[4, 8], is_longtext=True))
def test_longtext_tp1(config, run_id, run_config, worker_id):
    result, msg = longtext_throughput_test(config,
                                           run_id,
                                           run_config,
                                           cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=1),
                                           worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_benchmark_model_list(parallel_config=2, kvint_list=[4, 8], is_longtext=True))
def test_longtext_tp2(config, run_id, run_config, worker_id):
    result, msg = longtext_throughput_test(config,
                                           run_id,
                                           run_config,
                                           cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2),
                                           worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_benchmark_model_list(parallel_config=4, kvint_list=[4, 8], is_longtext=True))
def test_longtext_tp4(config, run_id, run_config, worker_id):
    result, msg = longtext_throughput_test(config,
                                           run_id,
                                           run_config,
                                           cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=4),
                                           worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_benchmark_model_list(parallel_config=8, kvint_list=[4, 8], is_longtext=True))
def test_longtext_tp8(config, run_id, run_config, worker_id):
    result, msg = longtext_throughput_test(config,
                                           run_id,
                                           run_config,
                                           cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=8),
                                           worker_id=worker_id)

    assert result, msg

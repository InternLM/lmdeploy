import pytest
from utils.benchmark_utils import generation_test
from utils.config_utils import (get_benchmark_model_list,
                                get_cuda_prefix_by_workerid)


@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_benchmark_model_list(tp_num=1))
def test_generation_tp1(config, run_id, run_config, worker_id):
    result, msg = generation_test(config,
                                  run_id,
                                  run_config,
                                  cuda_prefix=get_cuda_prefix_by_workerid(
                                      worker_id, tp_num=1),
                                  worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config',
                         get_benchmark_model_list(tp_num=1, is_longtext=True))
def test_generation_longtext_tp1(config, run_id, run_config, worker_id):
    result, msg = generation_test(config,
                                  run_id,
                                  run_config,
                                  is_longtext=True,
                                  cuda_prefix=get_cuda_prefix_by_workerid(
                                      worker_id, tp_num=1),
                                  worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_benchmark_model_list(tp_num=2))
def test_generation_tp2(config, run_id, run_config, worker_id):
    result, msg = generation_test(config,
                                  run_id,
                                  run_config,
                                  cuda_prefix=get_cuda_prefix_by_workerid(
                                      worker_id, tp_num=2),
                                  worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_2
@pytest.mark.longtext
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config',
                         get_benchmark_model_list(tp_num=2, is_longtext=True))
def test_generation_longtext_tp2(config, run_id, run_config, worker_id):
    result, msg = generation_test(config,
                                  run_id,
                                  run_config,
                                  is_longtext=True,
                                  cuda_prefix=get_cuda_prefix_by_workerid(
                                      worker_id, tp_num=2),
                                  worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_benchmark_model_list(tp_num=4))
def test_generation_tp4(config, run_id, run_config, worker_id):
    result, msg = generation_test(config,
                                  run_id,
                                  run_config,
                                  cuda_prefix=get_cuda_prefix_by_workerid(
                                      worker_id, tp_num=4),
                                  worker_id=worker_id)

    assert result, msg


@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config',
                         get_benchmark_model_list(tp_num=4, is_longtext=True))
def test_generation_longtext_tp4(config, run_id, run_config, worker_id):
    result, msg = generation_test(config,
                                  run_id,
                                  run_config,
                                  is_longtext=True,
                                  cuda_prefix=get_cuda_prefix_by_workerid(
                                      worker_id, tp_num=4),
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
def test_generation_fun_tp2(config, run_id, run_config, worker_id):
    result, msg = generation_test(config,
                                  run_id,
                                  run_config,
                                  cuda_prefix=get_cuda_prefix_by_workerid(
                                      worker_id, tp_num=2),
                                  worker_id=worker_id,
                                  is_smoke=True)

    assert result, msg

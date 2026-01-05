import pytest
from utils.config_utils import get_func_config_list
from utils.run_client_chat import run_tests

BACKEND = 'turbomind'


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 1}))
def test_hf_turbomind_chat_tp1(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 2}))
def test_hf_turbomind_chat_tp2(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 4}))
def test_hf_turbomind_chat_tp4(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 8}))
def test_hf_turbomind_chat_tp8(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('run_config', [{
    'model': 'microsoft/Phi-4-mini-instruct',
    'backend': BACKEND,
    'communicator': 'cuda-ipc',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'microsoft/Phi-4-mini-instruct-inner-4bits',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 4,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'microsoft/Phi-4-mini-instruct-inner-w8a8',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}])
def test_hf_turbomind_chat_fallback_backend_tp1(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', [{
    'model': 'google/gemma-2-27b-it',
    'backend': BACKEND,
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'deepseek-ai/deepseek-moe-16b-chat',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}])
def test_hf_turbomind_chat_fallback_backend_tp2(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 1}, 'base_model'))
def test_hf_turbomind_base_tp1(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'base_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 2}, 'base_model'))
def test_hf_turbomind_base_tp2(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'base_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', [{
    'model': 'internlm/internlm2_5-20b-chat',
    'backend': BACKEND,
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'internlm/internlm2_5-20b-chat-inner-4bits',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}])
def test_hf_turbomind_chat_pr_tp2(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', [{
    'model': 'OpenGVLab/InternVL3-8B',
    'backend': BACKEND,
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL3-8B',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}])
def test_hf_turbomind_chat_pr_tp1(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('run_config', [{
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'backend': BACKEND,
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}])
def test_modelscope_turbomind_chat_tp1(config, run_config, cli_case_config, worker_id):
    run_config['env'] = {'LMDEPLOY_USE_MODELSCOPE': 'True'}
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)

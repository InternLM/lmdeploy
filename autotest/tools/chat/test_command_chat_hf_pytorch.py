import pytest
from utils.config_utils import get_cuda_id_by_workerid, get_func_config_list
from utils.run_client_chat import run_tests

BACKEND = 'pytorch'


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('run_config', get_func_config_list('pytorch', {'tp': 1}))
def test_hf_pytorch_chat_tp1(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 2}))
def test_hf_pytorch_chat_tp2(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 4}))
def test_hf_pytorch_chat_tp4(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 8}))
def test_hf_pytorch_chat_tp8(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_16
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 16}))
def test_hf_pytorch_chat_tp16(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 1}, 'base_model'))
def test_hf_pytorch_base_tp1(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'base_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 2}, 'base_model'))
def test_hf_pytorch_base_tp2(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'base_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', [{
    'model': 'internlm/internlm2_5-20b-chat',
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
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id, gpu_num='5,6')


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', [{
    'model': 'meta-llama/Llama-2-7b-chat-hf',
    'backend': BACKEND,
    'communicator': 'nccl',
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
    gpu_num = get_cuda_id_by_workerid(worker_id)
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id, gpu_num=str(int(gpu_num) + 5))


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('run_config', [{
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'backend': BACKEND,
    'communicator': 'nccl',
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


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('run_config', [{
    'model': 'meta-llama/Llama-2-7b-chat-hf',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'adapters': 'lora/Llama2-Chinese-7b-Chat-LoRA'
    }
}])
def test_pytorch_chat_with_lora_tp1(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('run_config', [{
    'model': 'baichuan-inc/Baichuan2-13B-Chat',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'adapters': 'a=lora/2024-01-25_self_dup b=lora/2024-01-25_self'
    }
}])
def test_pytorch_chat_with_lora_tp2(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)

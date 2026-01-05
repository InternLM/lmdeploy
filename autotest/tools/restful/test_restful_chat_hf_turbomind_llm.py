import pytest
from utils.config_utils import get_func_config_list, get_workerid
from utils.constant import DEFAULT_PORT
from utils.run_restful_chat import (run_all_step, run_reasoning_case, run_tools_case, start_openai_service,
                                    terminate_restful_api, test_logprobs)


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    if hasattr(request, 'param'):
        run_config = request.param

        pid, startRes = start_openai_service(config, run_config, worker_id)
        try:
            yield run_config
        finally:
            if pid > 0:
                terminate_restful_api(worker_id, run_config)
    else:
        yield


BACKEND = 'turbomind'


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('prepare_environment', get_func_config_list(BACKEND, {'tp': 1}), indirect=True)
def test_restful_chat_tp1(config, common_case_config, worker_id):
    run_all_step(config, common_case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', get_func_config_list(BACKEND, {'tp': 2}), indirect=True)
def test_restful_chat_tp2(config, common_case_config, worker_id):
    run_all_step(config, common_case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment', get_func_config_list(BACKEND, {'tp': 4}), indirect=True)
def test_restful_chat_tp4(config, common_case_config, worker_id):
    run_all_step(config, common_case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('prepare_environment', get_func_config_list(BACKEND, {'tp': 8}), indirect=True)
def test_restful_chat_tp8(config, common_case_config, worker_id):
    run_all_step(config, common_case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment',
                         get_func_config_list(BACKEND, {'tp': 2}, extra={'enable_prefix_caching': None}),
                         indirect=True)
def test_restful_chat_prefix_cache_tp2(config, common_case_config, worker_id):
    run_all_step(config, common_case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment', [{
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
}],
                         indirect=True)
def test_restful_chat_fallback_backend_tp1(config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_all_step(config, case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', [{
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
}],
                         indirect=True)
def test_restful_chat_fallback_backend_tp2(config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_all_step(config, case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@pytest.mark.parametrize('prepare_environment', [{
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
}],
                         indirect=True)
def test_restful_chat_pr(config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_all_step(config, case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('prepare_environment', [{
    'model': 'internlm/internlm2_5-20b-chat',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL3-38B',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}],
                         indirect=True)
def test_restful_logprobs(worker_id):
    test_logprobs(worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment', [{
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'backend': BACKEND,
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {},
    'env': {
        'LMDEPLOY_USE_MODELSCOPE': 'True'
    }
}, {
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {},
    'env': {
        'LMDEPLOY_USE_MODELSCOPE': 'True'
    }
}],
                         indirect=True)
def test_modelscope_restful_chat_tp1(config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_all_step(config, case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [{
    'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'reasoning-parser': ' deepseek-r1'
    }
}],
                         indirect=True)
def test_restful_chat_reasoning_tp1(config, worker_id):
    run_reasoning_case(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [{
    'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'reasoning-parser': ' deepseek-r1'
    }
}],
                         indirect=True)
def test_restful_chat_reasoning_tp2(config, worker_id):
    run_reasoning_case(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [{
    'model': 'internlm/internlm2_5-7b-chat',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'tool-call-parser': 'internlm'
    }
}, {
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'tool-call-parser': 'qwen'
    }
}],
                         indirect=True)
def test_restful_chat_tools_tp1(config, worker_id):
    run_tools_case(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [
    {
        'model': 'internlm/internlm2_5-20b-chat',
        'backend': BACKEND,
        'communicator': 'nccl',
        'quant_policy': 0,
        'parallel_config': {
            'tp': 2
        },
        'extra_params': {
            'tool-call-parser': 'internlm'
        }
    },
],
                         indirect=True)
def test_restful_chat_tools_tp2(config, worker_id):
    run_tools_case(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_4
@pytest.mark.other
@pytest.mark.parametrize('prepare_environment', [{
    'model': 'meta-llama/Meta-Llama-3-1-70B-Instruct',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 4
    },
    'extra_params': {
        'tool-call-parser': 'llama3'
    }
}, {
    'model': 'Qwen/Qwen2.5-72B-Instruct',
    'backend': BACKEND,
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 4
    },
    'extra_params': {
        'tool-call-parser': 'qwen'
    }
}],
                         indirect=True)
def test_restful_chat_tools_tp4(config, worker_id):
    run_tools_case(config, port=DEFAULT_PORT + get_workerid(worker_id))

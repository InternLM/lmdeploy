import os

import numpy as np
import pytest
from utils.config_utils import set_device_env_variable
from utils.get_run_config import close_pipeline, get_tp_num

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline

SESSION_LEN = 198000
SESSION_LEN_128K = 128000
SESSION_LEN_32K = 32000

SESSION_LEN_CONFIG = {
    'Qwen/Qwen2.5-7B-Instruct': SESSION_LEN_32K,
    'Qwen/Qwen2.5-32B-Instruct': SESSION_LEN_32K,
    'Qwen/Qwen2.5-72B-Instruct': SESSION_LEN_32K,
    'Qwen/Qwen3-235B-A22B': SESSION_LEN_128K,
    'Qwen/Qwen3-30B-A3B': SESSION_LEN_128K,
    'Qwen/Qwen3-32B': SESSION_LEN_128K,
    'meta-llama/Meta-Llama-3-1-8B-Instruct': SESSION_LEN_128K,
    'internlm/Intern-S1-mini': SESSION_LEN_128K,
    'internlm/Intern-S1': SESSION_LEN_128K,
    'meta-llama/Meta-Llama-3-1-70B-Instruct': SESSION_LEN_128K
}


@pytest.mark.gpu_num_1
@pytest.mark.parametrize(
    'model', ['internlm/Intern-S1-mini', 'internlm/internlm2_5-7b-chat', 'internlm/internlm2_5-7b-chat-inner-4bits'])
def test_history_issue_tp1(config, model, worker_id):
    log_name = ''.join(['pipeline_longtext_issue_', worker_id, '.log'])
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    stream_infer_basic(config, model, log_name)


@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', ['Qwen/Qwen3-32B', 'Qwen/Qwen3-32B-inner-4bits', 'Qwen/Qwen3-30B-A3B'])
def test_history_issue_tp2(config, model, worker_id):
    log_name = ''.join(['pipeline_longtext_issue_', worker_id, '.log'])
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    stream_infer_basic(config, model, log_name)


def stream_infer_basic(config, model, log_name):
    tp_num = get_tp_num(config, model)
    model_path = '/'.join([config.get('model_path'), model])

    backend_config = TurbomindEngineConfig(session_len=SESSION_LEN, tp=tp_num)
    pipe = pipeline(model_path, backend_config=backend_config)
    prompt = '今 天 心 ' * int(SESSION_LEN / 6)

    gen_config = GenerationConfig(top_k=40)
    # stream infer
    for outputs in pipe.stream_infer(prompt, gen_config=gen_config):
        continue
    print(outputs)

    prompts = ['今 天 心 ' * int(SESSION_LEN / 6)] * 2
    # stream infer
    for outputs in pipe.stream_infer(prompts, gen_config=gen_config):
        continue
    print(outputs)

    close_pipeline(pipe)


@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', [
    'internlm/Intern-S1-mini', 'internlm/internlm2_5-7b-chat', 'internlm/internlm2_5-7b-chat-inner-4bits',
    'Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Meta-Llama-3-1-8B-Instruct'
])
@pytest.mark.parametrize('backend', ['turbomind', 'pytorch'])
def test_long_test_passkey_tp1(config, model, backend, worker_id):
    log_name = ''.join(['pipeline_longtext_passkey_', worker_id, '.log'])
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    passkey_retrival(config, model, backend, log_name, 1, SESSION_LEN_CONFIG.get(model, SESSION_LEN_128K))


@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', ['Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen3-30B-A3B', 'Qwen/Qwen3-32B'])
@pytest.mark.parametrize('backend', ['turbomind', 'pytorch'])
def test_long_test_passkey_tp2(config, model, backend, worker_id):
    log_name = ''.join(['pipeline_longtext_passkey_', worker_id, '.log'])
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    passkey_retrival(config, model, backend, log_name, 2, SESSION_LEN_CONFIG.get(model, SESSION_LEN_128K))


@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', ['Qwen/Qwen2.5-72B-Instruct'])
@pytest.mark.parametrize('backend', ['turbomind', 'pytorch'])
def test_long_test_passkey_tp4(config, model, backend, worker_id):
    log_name = ''.join(['pipeline_longtext_passkey_', worker_id, '.log'])
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    passkey_retrival(config, model, backend, log_name, 4, SESSION_LEN_CONFIG.get(model, SESSION_LEN_128K))


@pytest.mark.gpu_num_8
@pytest.mark.parametrize('model',
                         ['Qwen/Qwen3-235B-A22B', 'internlm/Intern-S1', 'meta-llama/Meta-Llama-3-1-70B-Instruct'])
@pytest.mark.parametrize('backend', ['turbomind', 'pytorch'])
def test_long_test_passkey_tp8(config, model, backend, worker_id):
    log_name = ''.join(['pipeline_longtext_passkey_', worker_id, '.log'])
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=8)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    passkey_retrival(config, model, backend, log_name, 8, SESSION_LEN_CONFIG.get(model, SESSION_LEN_128K))


YARN_CONFIG = {'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}}

NTK_CONFIG = {
    'rope_scaling': {
        'type': 'dynamic',
        'factor': 2.0
    },
}


def passkey_retrival(config, model, backend, log_name, tp_num, session_len: int = SESSION_LEN_128K):
    model_path = '/'.join([config.get('model_path'), model])
    if backend == 'turbomind':
        if 'qwen' in model.lower():
            backend_config = TurbomindEngineConfig(session_len=session_len,
                                                   max_batch_size=1,
                                                   cache_max_entry_count=0.7,
                                                   tp=tp_num,
                                                   hf_overrides=YARN_CONFIG)
        elif 'intern-s1' in model.lower():
            backend_config = TurbomindEngineConfig(session_len=session_len,
                                                   max_batch_size=1,
                                                   cache_max_entry_count=0.7,
                                                   tp=tp_num,
                                                   hf_overrides={'text_config': NTK_CONFIG})
        else:
            backend_config = TurbomindEngineConfig(session_len=session_len,
                                                   max_batch_size=1,
                                                   cache_max_entry_count=0.7,
                                                   tp=tp_num)
    else:
        if 'qwen' in model.lower():
            backend_config = PytorchEngineConfig(session_len=session_len,
                                                 tp=tp_num,
                                                 max_batch_size=1,
                                                 hf_overrides=YARN_CONFIG)
        elif 'intern-s1' in model.lower():
            backend_config = TurbomindEngineConfig(session_len=session_len,
                                                   max_batch_size=1,
                                                   cache_max_entry_count=0.7,
                                                   tp=tp_num,
                                                   hf_overrides={'text_config': NTK_CONFIG})
        else:
            backend_config = PytorchEngineConfig(session_len=session_len, tp=tp_num, max_batch_size=1)

    pipe = pipeline(model_path, backend_config=backend_config)

    gen_config = GenerationConfig(top_k=40)
    # inference
    pass_key1, prompt = get_passkey_prompt(pipe, session_len)
    response1 = pipe(prompt, gen_config=gen_config)

    # inference
    pass_key2, prompt = get_passkey_prompt(pipe, session_len)
    response2 = pipe([prompt] * 2, gen_config=gen_config)

    close_pipeline(pipe)

    assert str(pass_key1) in response1.text, str(response1)
    assert str(pass_key2) in response2[0].text and str(pass_key2) in response2[1].text, str(response2)


def get_passkey_prompt(pipe, session_len):
    # create long context input
    tok = pipe.tokenizer
    task_description = 'There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.'  # noqa: E501
    garbage = 'The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.'  # noqa: E501

    n_times = (session_len - 1000) // len(tok.encode(garbage))
    n_garbage_prefix = np.random.randint(0, n_times)
    n_garbage_suffix = n_times - n_garbage_prefix
    garbage_prefix = ' '.join([garbage] * n_garbage_prefix)
    garbage_suffix = ' '.join([garbage] * n_garbage_suffix)
    pass_key = np.random.randint(1, 50000)
    information_line = f'The pass key is {pass_key}. Remember it. {pass_key} is the pass key.'  # noqa: E501
    final_question = 'What is the pass key? The pass key is'
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]

    # inference
    prompt = ' '.join(lines)
    return pass_key, prompt

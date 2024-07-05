import os
from multiprocessing import Process

import numpy as np
import pytest
from utils.config_utils import get_cuda_id_by_workerid
from utils.get_run_config import get_tp_num
from utils.pipeline_chat import (assert_pipeline_common_log,
                                 save_pipeline_common_log)

from lmdeploy import (GenerationConfig, PytorchEngineConfig,
                      TurbomindEngineConfig, pipeline)

SESSION_LEN = 198000
SESSION_LEN_PASSKEY = 168000
SESSION_LEN_PASSKEY_1M = 1048576


@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', [
    'internlm/internlm2-chat-7b', 'internlm/internlm2-7b',
    'internlm/internlm2-chat-1_8b', 'internlm/internlm2-1_8b'
])
def test_history_issue_tp1(config, model, worker_id):
    log_name = ''.join(['pipeline_longtext_issue_', worker_id, '.log'])
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    p = Process(target=stream_infer_basic, args=(config, model, log_name))
    p.start()
    p.join()

    assert_pipeline_common_log(config, log_name)


@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', [
    'internlm/internlm2-chat-20b', 'internlm/internlm2-chat-20b-inner-4bits',
    'internlm/internlm2-20b', 'internlm/internlm2-20b-inner-4bits'
])
def test_history_issue_tp2(config, model, worker_id):
    log_name = ''.join(['pipeline_longtext_issue_', worker_id, '.log'])
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id,
                                                                     tp_num=2)
    p = Process(target=stream_infer_basic, args=(config, model, log_name))
    p.start()
    p.join()

    assert_pipeline_common_log(config, log_name)


def stream_infer_basic(config, model, log_name):
    tp_num = get_tp_num(config, model)
    model_path = '/'.join([config.get('model_path'), model])

    backend_config = TurbomindEngineConfig(rope_scaling_factor=2.0,
                                           session_len=SESSION_LEN,
                                           tp=tp_num)
    pipe = pipeline(model_path, backend_config=backend_config)
    prompt = '今 天 心 ' * int(SESSION_LEN / 6)

    gen_config = GenerationConfig(top_k=40)
    # stream infer
    for outputs in pipe.stream_infer(prompt, gen_config=gen_config):
        continue

    save_pipeline_common_log(config, log_name, True, str(outputs))

    prompts = ['今 天 心 ' * int(SESSION_LEN / 6)] * 2
    # stream infer
    for outputs in pipe.stream_infer(prompts, gen_config=gen_config):
        continue

    save_pipeline_common_log(config,
                             log_name,
                             True,
                             str(outputs),
                             write_type='a')
    assert False


@pytest.mark.gpu_num_1
@pytest.mark.parametrize(
    'model', ['internlm/internlm2-chat-7b', 'Qwen/Qwen2-7B-Instruct'])
@pytest.mark.parametrize('backend', ['turbomind'])
def test_long_test_passkey_tp1(config, model, backend, worker_id):
    log_name = ''.join(['pipeline_longtext_passkey_', worker_id, '.log'])
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    p = Process(target=passkey_retrival,
                args=(config, model, backend, log_name, 1))
    p.start()
    p.join()

    assert_pipeline_common_log(config, log_name)


@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', [
    'internlm/internlm2-chat-20b', 'internlm/internlm2-chat-20b-inner-4bits',
    'Qwen/Qwen2-7B-Instruct'
])
@pytest.mark.parametrize('backend', ['turbomind'])
def test_long_test_passkey_tp2(config, model, backend, worker_id):
    log_name = ''.join(['pipeline_longtext_passkey_', worker_id, '.log'])
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id,
                                                                     tp_num=2)
    p = Process(target=passkey_retrival,
                args=(config, model, backend, log_name, 2))
    p.start()
    p.join()

    assert_pipeline_common_log(config, log_name)


@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', ['internlm/internlm2_5-7b-chat-1m'])
@pytest.mark.parametrize('backend', ['turbomind'])
def test_long_test_passkey_tp4(config, model, backend, worker_id):
    log_name = ''.join(['pipeline_longtext_passkey_', worker_id, '.log'])
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id,
                                                                     tp_num=4)
    p = Process(target=passkey_retrival,
                args=(config, model, backend, log_name, 4,
                      SESSION_LEN_PASSKEY_1M))
    p.start()
    p.join()

    assert_pipeline_common_log(config, log_name)


def passkey_retrival(config,
                     model,
                     backend,
                     log_name,
                     tp_num,
                     session_len: int = SESSION_LEN_PASSKEY):
    model_path = '/'.join([config.get('model_path'), model])
    if backend == 'turbomind':
        if 'internlm2_5' in model and '-1m' in model:
            backend_config = TurbomindEngineConfig(rope_scaling_factor=2.5,
                                                   session_len=session_len,
                                                   max_batch_size=1,
                                                   cache_max_entry_count=0.7,
                                                   tp=tp_num)
        else:
            backend_config = TurbomindEngineConfig(rope_scaling_factor=2.0,
                                                   session_len=session_len,
                                                   use_logn_attn=True,
                                                   tp=tp_num)
    else:
        backend_config = PytorchEngineConfig(session_len=session_len,
                                             tp=tp_num)

    pipe = pipeline(model_path, backend_config=backend_config)

    gen_config = GenerationConfig(top_k=40)
    # inference
    pass_key, prompt = get_passkey_prompt(pipe, session_len)
    response = pipe(prompt, gen_config=gen_config)
    save_pipeline_common_log(config, log_name,
                             str(pass_key) in response.text, str(response))

    # inference
    pass_key, prompt = get_passkey_prompt(pipe, session_len)
    response = pipe([prompt] * 2, gen_config=gen_config)
    save_pipeline_common_log(config,
                             log_name,
                             str(pass_key) in response[0].text
                             and str(pass_key) in response[1].text,
                             str(response),
                             write_type='a')


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

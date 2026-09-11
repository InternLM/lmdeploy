import multiprocessing as mp
import os

import numpy as np
import pytest
from transformers import AutoTokenizer
from utils.config_utils import get_model_path_from_config, set_device_env_variable, unset_device_env_variable
from utils.pytest_layout_utils import layout_mark

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from lmdeploy.messages import Response

SESSION_LEN = 198000
SESSION_LEN_128K = 128000
SESSION_LEN_32K = 32000

SESSION_LEN_CONFIG = {
    'Qwen/Qwen2.5-7B-Instruct': SESSION_LEN_32K,
    'Qwen/Qwen3-30B-A3B': SESSION_LEN_128K,
    'Qwen/Qwen3-32B': SESSION_LEN_128K,
    'Qwen/Qwen3.5-35B-A3B': SESSION_LEN,
    'Qwen/Qwen3.5-27B': SESSION_LEN,
    'meta-llama/Meta-Llama-3-1-8B-Instruct': SESSION_LEN_128K,
    'meta-llama/Meta-Llama-3-1-70B-Instruct': SESSION_LEN_128K,
}


def run_case_in_spawn(target, args):
    ctx = mp.get_context('spawn')
    process = ctx.Process(target=target, args=args)
    process.start()
    process.join()
    if process.exitcode != 0:
        name = getattr(target, '__name__', repr(target))
        raise AssertionError(f'spawn worker {name!r} failed with exit code {process.exitcode!r}')


_PASSKEY_CASES = (
    (
        ['Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Meta-Llama-3-1-8B-Instruct'],
        ['turbomind', 'pytorch'],
        1,
    ),
    (
        ['Qwen/Qwen3-30B-A3B', 'Qwen/Qwen3-32B', 'Qwen/Qwen3.5-35B-A3B', 'Qwen/Qwen3.5-27B'],
        ['turbomind', 'pytorch'],
        2,
    ),
    (
        ['meta-llama/Meta-Llama-3-1-70B-Instruct'],
        ['turbomind', 'pytorch'],
        8,
    ),
)


def _build_passkey_params():
    rows = []
    for models, backends, tp in _PASSKEY_CASES:
        marks = [layout_mark({'tp': tp})]
        for model in models:
            for backend in backends:
                rows.append(
                    pytest.param(
                        model,
                        backend,
                        tp,
                        marks=marks,
                        id=f'{backend}-{model.replace("/", "_")}-tp{tp}',
                    ))
    return rows


_PASSKEY_PARAMS = _build_passkey_params()


@pytest.mark.parametrize('model, backend, tp', _PASSKEY_PARAMS)
def test_long_test_passkey(config, model, backend, tp, worker_id):
    log_name = ''.join(['pipeline_longtext_passkey_', worker_id, '.log'])
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=tp if tp > 1 else None)
        if tp > 1:
            os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_case_in_spawn(
        passkey_retrival_worker,
        (config, model, backend, log_name, tp, SESSION_LEN_CONFIG.get(model, SESSION_LEN_128K)),
    )
    if 'gw' in worker_id:
        unset_device_env_variable()


YARN_CONFIG = {'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}}

NTK_CONFIG = {
    'rope_scaling': {
        'type': 'dynamic',
        'factor': 2.0
    },
}


def passkey_retrival_worker(config, model, backend, log_name, tp_num, session_len: int = SESSION_LEN_128K):
    model_path = get_model_path_from_config(config, model)
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
            backend_config = PytorchEngineConfig(session_len=session_len,
                                                 tp=tp_num,
                                                 max_batch_size=1,
                                                 hf_overrides={'text_config': NTK_CONFIG})
        else:
            backend_config = PytorchEngineConfig(session_len=session_len, tp=tp_num, max_batch_size=1)

    pipe = pipeline(model_path, backend_config=backend_config)

    gen_config = GenerationConfig(top_k=40)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    pass_key1, prompt = get_passkey_prompt(pipe, session_len, tokenizer)
    response1 = pipe(prompt, gen_config=gen_config)

    pass_key2, prompt = get_passkey_prompt(pipe, session_len, tokenizer)
    response2 = pipe([prompt] * 2, gen_config=gen_config)

    pipe.close()

    assert isinstance(response1, Response), type(response1)
    assert response1.finish_reason in ('stop', 'length'), response1
    assert response1.generate_token_len > 0, response1

    assert isinstance(response2, list) and len(response2) == 2, response2
    for i, r in enumerate(response2):
        assert isinstance(r, Response), (i, type(r))
        assert r.finish_reason in ('stop', 'length'), r
        assert r.generate_token_len > 0, r

    assert str(pass_key1) in response1.text, str(response1)
    assert str(pass_key2) in response2[0].text and str(pass_key2) in response2[1].text, str(response2)


def get_passkey_prompt(pipe, session_len, tokenizer):
    task_description = 'There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.'  # noqa: E501
    garbage = 'The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.'  # noqa: E501

    n_times = (session_len - 1000) // len(tokenizer.encode(garbage))
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

    prompt = ' '.join(lines)
    return pass_key, prompt

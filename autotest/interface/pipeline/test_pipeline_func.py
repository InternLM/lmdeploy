from multiprocessing import Process

import pydantic
import pytest
from utils.config_utils import _is_bf16_supported_by_device, set_device_env_variable, unset_device_env_variable
from utils.get_run_config import _clear_device_cache
from utils.pipeline_chat import (assert_pipeline_batch_return, assert_pipeline_batch_stream_return,
                                 assert_pipeline_common_log, assert_pipeline_single_return,
                                 assert_pipeline_single_stream_return, save_pipeline_common_log)
from utils.restful_return_check import get_repeat_times

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline


def init_pipeline(model_path, backend_config):
    if not _is_bf16_supported_by_device() and isinstance(backend_config, PytorchEngineConfig):
        backend_config.dtype = 'float16'
    return pipeline(model_path, backend_config=backend_config)


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_return_with_prompt(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        response = pipe('Hi, pls intro yourself')
        result, msg = assert_pipeline_single_return(response)
        save_pipeline_common_log(config, file_name, result, response, msg)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_return_with_prompt_stream(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        response = []
        for item in pipe.stream_infer('Hi, pls intro yourself'):
            response.append(item)
        result, msg = assert_pipeline_single_stream_return(response)
        save_pipeline_common_log(config, file_name, result, response, msg)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_return_with_multi_prompt(config, model, backend, worker_id):

    def run_pipeline_testcase_with_prompt(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
        result, msg = assert_pipeline_batch_return(response, 2)
        save_pipeline_common_log(config, file_name, result, response, msg)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase_with_prompt, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_return_with_multi_prompt_stream(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        response = []
        for item in pipe.stream_infer(['Pls intro yourself', 'Shanghai is']):
            response.append(item)
        result, msg = assert_pipeline_batch_stream_return(response, 2)
        save_pipeline_common_log(config, file_name, result, response, msg)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_return_with_message(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):
        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        prompts = [[{'role': 'user', 'content': 'Hi, pls intro yourself'}]]
        response = pipe(prompts)
        print(response)
        result, msg = assert_pipeline_batch_return(response)
        save_pipeline_common_log(config, file_name, result, response, msg)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_return_with_message_stream(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):
        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        prompts = [[{'role': 'user', 'content': 'Hi, pls intro yourself'}]]
        response = []
        for item in pipe.stream_infer(prompts):
            response.append(item)
        result, msg = assert_pipeline_single_stream_return(response)
        save_pipeline_common_log(config, file_name, result, response, msg)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_return_with_message_batch(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):
        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        prompts = [[{
            'role': 'user',
            'content': 'Hi, pls intro yourself'
        }], [{
            'role': 'user',
            'content': 'Shanghai is'
        }]]
        response = pipe(prompts)
        print(response)
        result, msg = assert_pipeline_batch_return(response, 2)
        save_pipeline_common_log(config, file_name, result, response, msg)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_return_with_message_batch_stream(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):
        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        prompts = [[{
            'role': 'user',
            'content': 'Hi, pls intro yourself'
        }], [{
            'role': 'user',
            'content': 'Shanghai is'
        }]]
        response = []
        for item in pipe.stream_infer(prompts):
            response.append(item)
        result, msg = assert_pipeline_batch_stream_return(response, 2)
        save_pipeline_common_log(config, file_name, result, response, msg)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig])
def test_return_check_logprobs(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        gen_config = GenerationConfig(logprobs=10, max_new_tokens=5, top_k=40, do_sample=True)
        response = pipe('Hi, pls intro yourself', gen_config=gen_config)
        result, msg = assert_pipeline_single_return(response, logprobs_num=10)
        save_pipeline_common_log(config, file_name, result, response, msg)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig])
def test_return_check_logprobs_stream(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        gen_config = GenerationConfig(logprobs=10, max_new_tokens=5, top_k=40, do_sample=True)
        response = []
        for item in pipe.stream_infer('Hi, pls intro yourself', gen_config=gen_config):
            response.append(item)
        result, msg = assert_pipeline_single_stream_return(response, logprobs_num=10)
        save_pipeline_common_log(config, file_name, result, response, msg)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_backend_config_session_len(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(session_len=10, tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        response = pipe(['Hi, pls intro yourself', 'Shanghai is'])

        result = True
        for i in range(2):
            result &= response[i].finish_reason == 'error'
            result &= response[i].generate_token_len == 0
            result &= response[i].text == 'internal error happened, status code ResponseType.INPUT_LENGTH_ERROR'
        save_pipeline_common_log(config, file_name, result, response)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_min_new_tokens(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        # test min_new_tokens
        gen_config = GenerationConfig(min_new_tokens=200, ignore_eos=True)
        response = pipe(['Hi, pls intro yourself', 'Shanghai is'], gen_config=gen_config)
        result = True
        for i in range(2):
            result &= response[i].finish_reason == 'length'
            result &= response[i].index == i
        save_pipeline_common_log(config, file_name, result, response)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_min_new_tokens_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_stop_words(config, model, backend, worker_id):

    def run_pipeline_testcase_stop_words(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        # test stop_words
        gen_config = GenerationConfig(stop_words=[' and', '浦', ' to'])
        response = pipe(['Hi, pls intro yourself', 'Shanghai is'], gen_config=gen_config)
        result = True
        for i in range(2):
            result &= '浦' not in response[i].text
            result &= ' and' not in response[i].text and ' to ' not in response[i].text
            result &= response[i].finish_reason == 'stop' and response[i].generate_token_len < 50
        save_pipeline_common_log(config, file_name, result, response)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_stop_words_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase_stop_words, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_bad_words(config, model, backend, worker_id):

    def run_pipeline_testcase_bad_words(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        # test bad_words
        gen_config = GenerationConfig(bad_words=[' and', '浦', ' to'])
        response = pipe(['Hi, pls intro yourself', 'Shanghai is'], gen_config=gen_config)
        result = True
        for i in range(2):
            result &= '浦' not in response[i].text and ' and' not in response[i].text and ' to ' not in response[i].text
        save_pipeline_common_log(config, file_name, result, response)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_bad_words_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase_bad_words, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_special_words_false(config, model, backend, worker_id):

    def run_pipeline_testcase_special_words(config, model, backend, file_name):
        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        # test special_words
        prompt = '<|im_start|>system\n当开启工具以及代码时，根据需求选择合适的工具进行调用\n' + \
            '<|im_end|><|im_start|>system name=<|interpreter|>\n你现在已经' + \
            '能够在一个有状态的 Jupyter 笔记本环境中运行 Python 代码。当你向 python ' + \
            '发送含有 Python >代码的消息时，它将在该环境中执行。这个工具适用于多种场景，' + \
            '如数据分析或处理（包括数据操作、统计分析、图表绘制），复杂的计算问题（解决数学和物理' + \
            '难题），编程示例（理解编程概念或特性），文本处理和分析（比如文本解析和自然语言处理），机器学习和数据科学（用于' + \
            '展示模型训练和数据可视化），以及文件操作和数据导入（处理CSV、JSON等格式的文件）。<|im_end|>\n' + \
            '<|im_start|>user\n设 $L$ 为圆周$x^2+y^2=2x$，计算曲线积分：$I=\\int_L' + \
            '{x\\mathrm{d}s}=$<|im_end|>\n<|im_start|>assistant'

        gen_config = GenerationConfig(skip_special_tokens=False)
        response = pipe(prompt, gen_config=gen_config)
        result = '<|action_start|><|interpreter|>' in response.text
        save_pipeline_common_log(config, file_name, result, response)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_special_words_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase_special_words, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_special_words_true(config, model, backend, worker_id):

    def run_pipeline_testcase_special_words(config, model, backend, file_name):
        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        # test special_words
        prompt = '<|im_start|>system\n当开启工具以及代码时，根据需求选择合适的工具进行调用\n' + \
            '<|im_end|><|im_start|>system name=<|interpreter|>\n你现在已经' + \
            '能够在一个有状态的 Jupyter 笔记本环境中运行 Python 代码。当你向 python ' + \
            '发送含有 Python >代码的消息时，它将在该环境中执行。这个工具适用于多种场景，' + \
            '如数据分析或处理（包括数据操作、统计分析、图表绘制），复杂的计算问题（解决数学和物理' + \
            '难题），编程示例（理解编程概念或特性），文本处理和分析（比如文本解析和自然语言处理），机器学习和数据科学（用于' + \
            '展示模型训练和数据可视化），以及文件操作和数据导入（处理CSV、JSON等格式的文件）。<|im_end|>\n' + \
            '<|im_start|>user\n设 $L$ 为圆周$x^2+y^2=2x$，计算曲线积分：$I=\\int_L' + \
            '{x\\mathrm{d}s}=$<|im_end|>\n<|im_start|>assistant'

        gen_config = GenerationConfig(skip_special_tokens=True)
        response = pipe(prompt, gen_config=gen_config)
        result = '<|action_start|><|interpreter|>' not in response.text
        save_pipeline_common_log(config, file_name, result, response)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_special_words_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase_special_words, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_minimum_repetition_penalty(config, model, backend, worker_id):

    def run_pipeline_testcase_repetition_penalty(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        # test repetition_penalty
        gen_config = GenerationConfig(repetition_penalty=0.01, random_seed=1, do_sample=True)
        response = pipe('Shanghai is', gen_config=gen_config)

        result = get_repeat_times(response.text, 'is a name') > 5 or get_repeat_times(response.text, 'Shanghai is') > 5
        save_pipeline_common_log(config, file_name, result, response)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_repetition_penalty_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase_repetition_penalty, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_repetition_penalty_bigger_than_1(config, model, backend, worker_id):

    def run_pipeline_testcase_repetition_penalty(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        # test repetition_penalty
        gen_config = GenerationConfig(repetition_penalty=1.2, random_seed=1)
        response = pipe('Shanghai is', gen_config=gen_config)
        result, msg = assert_pipeline_single_return(response)
        save_pipeline_common_log(config, file_name, result, response, msg)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_repetition_penalty_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase_repetition_penalty, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_minimun_topp(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        # test repetition_penalty
        gen_config = GenerationConfig(top_p=0.1, random_seed=1)
        response = pipe('Shanghai is', gen_config=gen_config)
        result, msg = assert_pipeline_single_return(response)
        save_pipeline_common_log(config, file_name, result, response, msg)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_minimun_topk(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        # test repetition_penalty
        gen_config = GenerationConfig(top_k=1, max_new_tokens=20, do_sample=True)
        response_list = []
        for i in range(3):
            response_list.append(pipe('Shanghai is', gen_config=gen_config))
        result = response_list[0].text == response_list[1].text and response_list[1].text == response_list[2].text
        save_pipeline_common_log(config, file_name, result, response_list)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_diff_random_seed(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        response_list = []
        for i in range(3):
            gen_config = GenerationConfig(random_seed=i, temperature=1.0, top_k=40, do_sample=True)
            response_list.append(pipe('Shanghai is', gen_config=gen_config))
        result = response_list[0].text != response_list[1].text and response_list[1].text != response_list[2].text
        save_pipeline_common_log(config, file_name, result, response_list)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_same_random_seed(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        gen_config = GenerationConfig(random_seed=1, top_k=40, do_sample=True)
        response_list = []
        for i in range(3):
            response_list.append(pipe('Shanghai is', gen_config=gen_config))
        result = response_list[0].text == response_list[1].text and response_list[1].text == response_list[2].text
        save_pipeline_common_log(config, file_name, result, response_list)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_do_sample_batch(config, model, backend, worker_id):

    def run_pipeline_testcase(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        gen_config = GenerationConfig(temperature=1.0, top_k=40, do_sample=True)
        response = pipe(['Shanghai is'] * 3, gen_config=gen_config)
        result = response[0].text != response[1].text and response[1].text != response[2].text
        save_pipeline_common_log(config, file_name, result, response)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_max_new_tokens(config, model, backend, worker_id):

    def run_pipeline_testcase_max_new_tokens(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        # test max_new_tokens
        gen_config = GenerationConfig(max_new_tokens=5)
        response = pipe(['Hi, pls intro yourself', 'Shanghai is'], gen_config=gen_config)
        result = True
        for i in range(2):
            result &= response[i].finish_reason == 'length'
            result &= response[i].generate_token_len == 6 or response[i].generate_token_len == 5
        save_pipeline_common_log(config, file_name, result, response)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_max_new_tokens_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase_max_new_tokens, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_gen_config_ignore_eos(config, model, backend, worker_id):

    def run_pipeline_testcase_ignore_eos(config, model, backend, file_name):

        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=2)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        # test max_new_tokens with ignore_eos
        gen_config = GenerationConfig(ignore_eos=True, max_new_tokens=256)
        response = pipe(['Hi, pls intro yourself', 'Shanghai is'], gen_config=gen_config)
        result = True
        for i in range(2):
            result &= response[i].finish_reason == 'length'
            result &= response[i].generate_token_len == 257 or response[i].generate_token_len == 256
        save_pipeline_common_log(config, file_name, result, response)
        del pipe
        _clear_device_cache()

    file_name = f'pipeline_log_ignore_eos_{worker_id}.txt'
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    p = Process(target=run_pipeline_testcase_ignore_eos, args=(config, model, backend, file_name))

    p.start()
    p.join()
    assert_pipeline_common_log(config, file_name)
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig, PytorchEngineConfig])
def test_backend_config_input_validation(config, model, backend, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    model_path = '/'.join([config.get('model_path'), model])
    backend_config = backend(tp=2)
    pipe = init_pipeline(model_path, backend_config=backend_config)
    with pytest.raises(AssertionError):
        gen_config = GenerationConfig(top_p=-0.01)
        pipe('Shanghai is', gen_config=gen_config)

    with pytest.raises(AssertionError):
        gen_config = GenerationConfig(top_p=1.01)
        pipe('Shanghai is', gen_config=gen_config)

    with pytest.raises(AssertionError):
        gen_config = GenerationConfig(temperature=-1)
        pipe('Shanghai is', gen_config=gen_config)

    with pytest.raises(AssertionError):
        gen_config = GenerationConfig(temperature=2.01)
        pipe('Shanghai is', gen_config=gen_config)

    with pytest.raises(AssertionError):
        gen_config = GenerationConfig(top_k=-1)
        pipe('Shanghai is', gen_config=gen_config)

    with pytest.raises(AssertionError):
        gen_config = GenerationConfig(n=-1)
        pipe('Shanghai is', gen_config=gen_config)

    del pipe
    _clear_device_cache()
    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig])
def test_backend_config_validate_turbomind(config, model, backend, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    model_path = '/'.join([config.get('model_path'), model])
    with pytest.raises(pydantic.ValidationError, match='tp must be a positive integer'):
        backend_config = backend(tp=0)
        pipeline(model_path, backend_config=backend_config)

    with pytest.raises(AssertionError, match='max_batch_size should be greater than 0, but got 0'):
        backend_config = backend(max_batch_size=0)
        pipeline(model_path, backend_config=backend_config)

    with pytest.raises(pydantic.ValidationError):
        backend_config = backend(cache_max_entry_count=0)
        pipeline(model_path, backend_config=backend_config)

    with pytest.raises(pydantic.ValidationError):
        backend_config = backend(quant_policy=1)
        pipeline(model_path, backend_config=backend_config)

    with pytest.raises(pydantic.ValidationError):
        backend_config = backend(rope_scaling_factor=-1)
        pipeline(model_path, backend_config=backend_config)

    with pytest.raises(pydantic.ValidationError):
        backend_config = backend(max_prefill_token_num=-1)
        pipeline(model_path, backend_config=backend_config)

    with pytest.raises(pydantic.ValidationError):
        backend_config = backend(num_tokens_per_iter=-1)
        pipeline(model_path, backend_config=backend_config)

    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'OpenGVLab/InternVL2_5-26B'])
@pytest.mark.parametrize('backend', [PytorchEngineConfig])
def test_backend_config_validate_pytorch(config, model, backend, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
    model_path = '/'.join([config.get('model_path'), model])
    with pytest.raises(AssertionError):
        backend_config = backend(tp=0)
        init_pipeline(model_path, backend_config=backend_config)

    with pytest.raises(SystemExit):
        backend_config = backend(max_batch_size=0)
        init_pipeline(model_path, backend_config=backend_config)

    with pytest.raises(AssertionError):
        backend_config = backend(cache_max_entry_count=0)
        init_pipeline(model_path, backend_config=backend_config)

    with pytest.raises(AssertionError):
        backend_config = backend(num_cpu_blocks=-1)
        init_pipeline(model_path, backend_config=backend_config)

    with pytest.raises(AssertionError):
        backend_config = backend(num_gpu_blocks=-1)
        init_pipeline(model_path, backend_config=backend_config)

    if 'gw' in worker_id:
        unset_device_env_variable()


@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat'])
@pytest.mark.parametrize('backend', [TurbomindEngineConfig])
def test_backend_config_tp(config, model, backend, worker_id):
    with pytest.raises(AssertionError):
        if 'gw' in worker_id:
            set_device_env_variable(worker_id, tp_num=2)
        model_path = '/'.join([config.get('model_path'), model])
        backend_config = backend(tp=100)
        pipe = init_pipeline(model_path, backend_config=backend_config)
        del pipe
        _clear_device_cache()
        if 'gw' in worker_id:
            unset_device_env_variable()

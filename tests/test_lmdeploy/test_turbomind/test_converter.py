from lmdeploy.turbomind.deploy.converter import (
    get_input_model_registered_name,
    get_output_model_registered_name_and_config)
from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS
from lmdeploy.turbomind.deploy.target_model.base import OUTPUT_MODELS


def test_registered_models():
    for model in [
            'baichuan-inc/Baichuan-7B', 'baichuan-inc/Baichuan2-7B-Chat',
            'baichuan-inc/Baichuan-13B-Chat',
            'baichuan-inc/Baichuan2-13B-Chat', 'internlm/internlm-chat-7b',
            'internlm/internlm2-chat-7b',
            'internlm/internlm-xcomposer2-4khd-7b',
            'internlm/internlm-xcomposer2-vl-7b',
            'internlm/internlm-xcomposer2-7b', 'lmsys/vicuna-7b-v1.5',
            '01-ai/Yi-1.5-9B', 'deepseek-ai/deepseek-coder-6.7b-instruct',
            'deepseek-ai/deepseek-llm-7b-chat', 'Qwen/Qwen-7B-Chat',
            'Qwen/Qwen1.5-7B-Chat', 'Qwen/Qwen-VL-Chat',
            'liuhaotian/llava-v1.6-34b', 'liuhaotian/llava-v1.6-mistral-7b',
            'liuhaotian/llava-v1.6-vicuna-13b', 'OpenGVLab/InternVL-Chat-V1-5',
            'deepseek-ai/deepseek-vl-7b-chat', 'YanweiLi/MGM-7B'
    ]:
        input_name = get_input_model_registered_name(model, model_format='hf')
        assert input_name in list(INPUT_MODELS.module_dict.keys())

        output_name, config = get_output_model_registered_name_and_config(
            model, model_format='hf', group_size=0)
        assert output_name in list(OUTPUT_MODELS.module_dict.keys())
        assert config.group_size == 0
        assert config.session_len > 0
        assert config.model_arch is not None

    for model in [
            'Qwen/Qwen1.5-4B-Chat-AWQ',
            'solidrust/Meta-Llama-3-8B-Instruct-hf-AWQ'
    ]:
        name = get_input_model_registered_name(model, model_format='awq')
        assert name in list(INPUT_MODELS.module_dict.keys())
        output_name, config = get_output_model_registered_name_and_config(
            model, model_format='awq', group_size=0)
        assert output_name in list(OUTPUT_MODELS.module_dict.keys())
        assert config.group_size == 128
        assert config.session_len > 0
        assert config.model_arch is not None

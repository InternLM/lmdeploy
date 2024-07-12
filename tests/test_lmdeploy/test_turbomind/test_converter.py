from lmdeploy import TurbomindEngineConfig
from lmdeploy.turbomind.deploy.converter import (
    get_input_model_registered_name,
    get_output_model_registered_name_and_config)
from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS


def test_registered_models():
    for model, model_format, group_size, weight_type, register_name in [
        ('internlm/internlm2-7b', 'hf', 0, 'bf16', 'bf16'),
        ('baichuan-inc/Baichuan-7B', 'hf', 0, 'fp16', 'fp16'),
        ('baichuan-inc/Baichuan2-7B-Chat', 'hf', 0, 'bf16', 'bf16'),
        ('baichuan-inc/Baichuan-13B-Chat', 'hf', 0, 'bf16', 'bf16'),
        ('baichuan-inc/Baichuan2-13B-Chat', 'hf', 0, 'bf16', 'bf16'),
        ('internlm/internlm-chat-7b', 'hf', 0, 'fp16', 'fp16'),
        ('internlm/internlm2-chat-7b', 'hf', 0, 'bf16', 'bf16'),
        ('internlm/internlm-xcomposer2-4khd-7b', 'hf', 0, 'bf16', 'plora'),
        ('internlm/internlm-xcomposer2-vl-7b', 'hf', 0, 'bf16', 'plora'),
        ('internlm/internlm-xcomposer2-7b', 'hf', 0, 'bf16', 'plora'),
        ('lmsys/vicuna-7b-v1.5', 'hf', 0, 'fp16', 'fp16'),
        ('01-ai/Yi-1.5-9B', 'hf', 0, 'bf16', 'bf16'),
        ('deepseek-ai/deepseek-coder-6.7b-instruct', 'hf', 0, 'bf16', 'bf16'),
        ('deepseek-ai/deepseek-llm-7b-chat', 'hf', 0, 'bf16', 'bf16'),
        ('Qwen/Qwen-7B-Chat', 'hf', 0, 'bf16', 'bf16'),
        ('Qwen/Qwen1.5-7B-Chat', 'hf', 0, 'bf16', 'bf16'),
        ('Qwen/Qwen2-7B-Instruct', 'hf', 0, 'bf16', 'bf16'),
        ('Qwen/Qwen-VL-Chat', 'hf', 0, 'bf16', 'bf16'),
        ('liuhaotian/llava-v1.6-34b', 'hf', 0, 'bf16', 'bf16'),
        ('liuhaotian/llava-v1.6-mistral-7b', 'hf', 0, 'bf16', 'bf16'),
        ('liuhaotian/llava-v1.6-vicuna-13b', 'hf', 0, 'bf16', 'bf16'),
        ('OpenGVLab/InternVL-Chat-V1-5', 'hf', 0, 'bf16', 'bf16'),
        ('deepseek-ai/deepseek-vl-7b-chat', 'hf', 0, 'fp16', 'fp16'),
        ('YanweiLi/MGM-7B', 'hf', 0, 'bf16', 'bf16'),
        ('Qwen/Qwen1.5-4B-Chat-AWQ', 'awq', 128, 'int4', 'w4'),
        ('solidrust/Meta-Llama-3-8B-Instruct-hf-AWQ', 'awq', 128,
         'int4', 'w4'),
        ('internlm/internlm2-chat-20b-4bits', 'awq', 128, 'int4', 'w4'),
        ('internlm/internlm-xcomposer2-vl-7b-4bit', 'awq', 128, 'int4',
         'plora-w4')
    ]:
        input_name = get_input_model_registered_name(model,
                                                     model_format=model_format)
        assert input_name in list(INPUT_MODELS.module_dict.keys())

        output_name, config = get_output_model_registered_name_and_config(
            model, model_format=model_format, group_size=0)
        assert output_name == register_name
        assert config.group_size == group_size
        assert config.weight_type == weight_type
        assert config.session_len > 0
        assert config.model_arch is not None


def test_update_from_engine_config():
    import copy
    _, _config = get_output_model_registered_name_and_config(
        'internlm/internlm2-chat-7b', model_format='hf', group_size=0)
    config = copy.deepcopy(_config)
    config.update_from_engine_config(None)
    assert (config == _config)

    config = copy.deepcopy(_config)
    config.update_from_engine_config(TurbomindEngineConfig())
    assert config.tensor_para_size == 1
    assert config.session_len == 32776
    assert config.max_batch_size == 128
    assert config.cache_max_entry_count == 0.8
    assert config.quant_policy == 0
    assert config.max_prefill_iters == 5
    assert config.num_tokens_per_iter == 8192

    config = copy.deepcopy(_config)
    config.update_from_engine_config(
        TurbomindEngineConfig(max_prefill_token_num=2048,
                              num_tokens_per_iter=0))
    assert config.max_prefill_iters == 17
    assert config.num_tokens_per_iter == 2048

    config = copy.deepcopy(_config)
    config.update_from_engine_config(
        TurbomindEngineConfig(max_prefill_token_num=2048,
                              num_tokens_per_iter=256))
    assert config.max_prefill_iters == 1
    assert config.num_tokens_per_iter == 256

    config = copy.deepcopy(_config)
    engine_config = TurbomindEngineConfig(model_format='hf',
                                          tp=2,
                                          session_len=4000,
                                          max_batch_size=100,
                                          cache_max_entry_count=0.5,
                                          quant_policy=8,
                                          rope_scaling_factor=3.0,
                                          use_logn_attn=True,
                                          max_prefill_iters=64,
                                          num_tokens_per_iter=256)

    config.update_from_engine_config(engine_config)

    assert (config.tensor_para_size == engine_config.tp)
    assert (config.session_len == engine_config.session_len)
    assert (config.max_batch_size == engine_config.max_batch_size)
    assert (
        config.cache_max_entry_count == engine_config.cache_max_entry_count)
    assert (config.quant_policy == engine_config.quant_policy)
    assert (config.rope_scaling_factor == engine_config.rope_scaling_factor)
    assert (config.use_logn_attn == engine_config.use_logn_attn)
    assert (config.max_prefill_iters == engine_config.max_prefill_iters)
    assert (config.num_tokens_per_iter == engine_config.num_tokens_per_iter)

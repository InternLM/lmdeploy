from lmdeploy import TurbomindEngineConfig
from lmdeploy.turbomind.deploy.converter import (
    get_input_model_registered_name,
    get_output_model_registered_name_and_config)
from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS
from lmdeploy.turbomind.deploy.target_model.base import (OUTPUT_MODELS,
                                                         TurbomindModelConfig)


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


def test_turbomind_model_config_udpate():
    this = TurbomindModelConfig.from_dict({}, allow_none=True)
    this.head_num = 100
    this.weight_type = 'fp16'

    engine_config = TurbomindEngineConfig(model_format='awq',
                                          tp=2,
                                          session_len=4000,
                                          max_batch_size=100,
                                          cache_max_entry_count=0.5,
                                          quant_policy=8,
                                          rope_scaling_factor=3.0,
                                          use_logn_attn=True,
                                          max_prefill_iters=64,
                                          num_tokens_per_iter=256)
    other = TurbomindModelConfig.from_engine_config(engine_config)
    this.update(other)

    assert (this.head_num == 100)
    assert (this.weight_type == 'fp16')
    assert (this.tensor_para_size == other.tensor_para_size)
    assert (this.session_len == other.session_len)
    assert (this.max_batch_size == other.max_batch_size)
    assert (this.cache_max_entry_count == other.cache_max_entry_count)
    assert (this.max_prefill_iters == other.max_prefill_iters)
    assert (this.num_tokens_per_iter == other.num_tokens_per_iter)
    assert (this.quant_policy == other.quant_policy)
    assert (this.rope_scaling_factor == other.rope_scaling_factor)
    assert (this.use_logn_attn == other.use_logn_attn)

    engine_config = TurbomindEngineConfig(max_prefill_iters=512,
                                          num_tokens_per_iter=1024)
    other = TurbomindModelConfig.from_engine_config(engine_config)
    this.update(other)
    assert (this.max_prefill_iters == other.max_prefill_iters)
    assert (this.num_tokens_per_iter == other.num_tokens_per_iter)

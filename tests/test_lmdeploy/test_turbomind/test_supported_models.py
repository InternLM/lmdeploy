from transformers import PretrainedConfig

from lmdeploy.turbomind.supported_models import get_model_arch


def test_get_model_arch():
    models = [
        # model_path, model_arch
        ('baichuan-inc/Baichuan-7B', 'BaiChuanForCausalLM'),
        ('baichuan-inc/Baichuan2-7B-Chat', 'BaichuanForCausalLM'),
        ('baichuan-inc/Baichuan-13B-Chat', 'BaichuanForCausalLM'),
        ('baichuan-inc/Baichuan2-13B-Chat', 'BaichuanForCausalLM'),
        ('internlm/internlm-chat-7b', 'InternLMForCausalLM'),
        ('internlm/internlm2-chat-7b', 'InternLM2ForCausalLM'),
        ('internlm/internlm-xcomposer2-4khd-7b',
         'InternLMXComposer2ForCausalLM'),
        ('internlm/internlm-xcomposer2-vl-7b',
         'InternLMXComposer2ForCausalLM'),
        ('internlm/internlm-xcomposer2-7b', 'InternLMXComposer2ForCausalLM'),
        ('lmsys/vicuna-7b-v1.5', 'LlamaForCausalLM'),
        ('01-ai/Yi-1.5-9B', 'LlamaForCausalLM'),
        ('deepseek-ai/deepseek-coder-6.7b-instruct', 'LlamaForCausalLM'),
        ('deepseek-ai/deepseek-llm-7b-chat', 'LlamaForCausalLM'),
        ('Qwen/Qwen-7B-Chat', 'QWenLMHeadModel'),
        ('Qwen/Qwen1.5-7B-Chat', 'Qwen2ForCausalLM'),
        ('Qwen/Qwen-VL-Chat', 'QWenLMHeadModel'),
        ('liuhaotian/llava-v1.6-34b', 'LlavaLlamaForCausalLM'),
        ('liuhaotian/llava-v1.6-mistral-7b', 'LlavaMistralForCausalLM'),
        ('liuhaotian/llava-v1.6-vicuna-13b', 'LlavaLlamaForCausalLM'),
        ('OpenGVLab/InternVL-Chat-V1-5', 'InternVLChatModel'),
        ('deepseek-ai/deepseek-vl-7b-chat', 'MultiModalityCausalLM'),
        ('YanweiLi/MGM-2B', 'MiniGeminiGemmaForCausalLM'),
    ]
    for model_path, model_arch in models:
        arch, config = get_model_arch(model_path)
        assert arch == model_arch
        assert isinstance(config, PretrainedConfig)

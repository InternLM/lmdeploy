import numpy as np
import pytest


class TestAutoBackend:

    @pytest.fixture
    def models(self):
        # example models to test
        # format (model_path, is_turbomind_supported)
        models = [
            ('baichuan-inc/Baichuan-7B', True),
            ('baichuan-inc/Baichuan2-7B-Chat', True),
            ('baichuan-inc/Baichuan-13B-Chat', False),
            ('baichuan-inc/Baichuan2-13B-Chat', False),
            ('internlm/internlm-chat-7b', True),
            ('internlm/internlm2-chat-7b', True),
            ('internlm/internlm-xcomposer2-7b', True),
            ('internlm/internlm-xcomposer-7b', False),
            ('THUDM/chatglm2-6b', False),
            ('THUDM/chatglm3-6b', False),
            ('deepseek-ai/deepseek-moe-16b-chat', False),
            ('01-ai/Yi-34B-Chat', True),
            ('codellama/CodeLlama-7b-Instruct-hf', True),
            ('Qwen/Qwen-7B-Chat', True),
            ('Qwen/Qwen-VL-Chat', True),
            ('Qwen/Qwen1.5-4B-Chat', True),
            ('Qwen/Qwen1.5-0.5B-Chat', True),
        ]
        return models

    def test_autoget_backend(self, models):
        from lmdeploy.archs import autoget_backend
        n = len(models)
        choices = np.random.choice(n, n // 2, replace=False)
        for i in choices:
            model, is_support_turbomind = models[i]
            target = 'turbomind' if is_support_turbomind else 'pytorch'
            backend = autoget_backend(model, trust_remote_code=True)
            assert backend == target

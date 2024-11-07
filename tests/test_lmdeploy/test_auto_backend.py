import os
import tempfile

import numpy as np
import pytest


class TestAutoBackend:

    @pytest.fixture
    def turbomind_workspace(self):
        workspace = tempfile.TemporaryDirectory(
            'internlm-chat-7b-turbomind').name
        os.makedirs(os.path.join(workspace, 'triton_models'), exist_ok=True)
        return workspace

    @pytest.fixture
    def models(self):
        # example models to test
        # format (model_path, is_pytorch_supported, is_turbomind_supported)
        models = [
            ('baichuan-inc/Baichuan-7B', False, True),
            ('baichuan-inc/Baichuan2-7B-Chat', True, True),
            ('baichuan-inc/Baichuan-13B-Chat', False, False),
            ('baichuan-inc/Baichuan2-13B-Chat', True, False),
            ('internlm/internlm-chat-7b', True, True),
            ('internlm/internlm2-chat-7b', True, True),
            ('internlm/internlm-xcomposer2-7b', False, True),
            ('internlm/internlm-xcomposer-7b', False, False),
            ('THUDM/chatglm2-6b', True, False),
            ('THUDM/chatglm3-6b', True, False),
            ('deepseek-ai/deepseek-moe-16b-chat', True, False),
            ('tiiuae/falcon-7b-instruct', True, False),
            ('01-ai/Yi-34B-Chat', True, True),
            ('codellama/CodeLlama-7b-Instruct-hf', True, True),
            ('mistralai/Mistral-7B-Instruct-v0.1', True, True),
            ('mistralai/Mixtral-8x7B-Instruct-v0.1', True, True),
            ('Qwen/Qwen-7B-Chat', True, True),
            ('Qwen/Qwen-VL-Chat', False, True),
            ('Qwen/Qwen1.5-4B-Chat', True, True),
            ('Qwen/Qwen1.5-0.5B-Chat', True, True),
        ]
        return models

    def test_pytorch_is_supported(self, turbomind_workspace, models):
        from lmdeploy.pytorch.supported_models import is_supported
        assert is_supported(turbomind_workspace) is False
        for m, flag, _ in models:
            assert is_supported(m) is flag

    def test_turbomind_is_supported(self, turbomind_workspace, models):
        from lmdeploy.turbomind.supported_models import is_supported
        assert is_supported(turbomind_workspace) is True
        for m, _, flag in models:
            assert is_supported(m) is flag

    def test_autoget_backend(self, turbomind_workspace, models):
        from lmdeploy.archs import autoget_backend
        assert autoget_backend(turbomind_workspace) == 'turbomind'
        n = len(models)
        choices = np.random.choice(n, n // 2, replace=False)
        for i in choices:
            model, is_support_pytorch, is_support_turbomind = models[i]
            target = 'turbomind' if is_support_turbomind else 'pytorch'
            backend = autoget_backend(model)
            assert backend == target

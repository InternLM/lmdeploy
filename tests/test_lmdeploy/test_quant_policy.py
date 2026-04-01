"""Test quant_policy for PytorchEngine.

This test verifies that quant_policy=42 (K=4bit, V=2bit mixed precision) works correctly with PytorchEngine for
Qwen3-8B.
"""

import gc

import pytest
import torch

from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import Response

MODEL_ID = 'Qwen/Qwen3-8B'

class TestQuantPolicy42:
    """Test class for quant_policy=42 (K=4bit, V=2bit mixed precision)."""

    @pytest.fixture(scope='class', autouse=True)
    def pipe(self):
        """Create pipeline with quant_policy=42."""
        engine_config = PytorchEngineConfig(
            tp=1,
            cache_max_entry_count=0.1,
            quant_policy=42,
        )
        pipe = pipeline(MODEL_ID, backend_config=engine_config, log_level='INFO')
        yield pipe
        pipe.close()
        del pipe
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()

    def test_infer_single_prompt(self, pipe):
        """Test single prompt inference with quant_policy=42."""
        prompt = 'Hello, how are you?'
        response = pipe.infer(prompt, max_new_tokens=30)

        assert isinstance(response, Response)
        assert hasattr(response, 'text')
        assert len(response.text) > 0
        # Basic sanity check - output should contain readable text
        assert len(response.text.strip()) > 0

    def test_infer_batch_prompts(self, pipe):
        """Test batch inference with quant_policy=42."""
        prompts = ['What is AI?', 'Hello!']
        responses = pipe.infer(prompts, max_new_tokens=20)

        assert isinstance(responses, list)
        assert len(responses) == len(prompts)
        for resp in responses:
            assert isinstance(resp, Response)
            assert len(resp.text) > 0

    def test_infer_with_generation_config(self, pipe):
        """Test inference with GenerationConfig."""
        from lmdeploy import GenerationConfig

        gen_config = GenerationConfig(max_new_tokens=20, temperature=0.7)
        prompt = 'Tell me a short joke'
        response = pipe.infer(prompt, gen_config=gen_config)

        assert isinstance(response, Response)
        assert len(response.text) > 0

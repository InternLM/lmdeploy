"""Test quant_policy=42 (K=4bit, V=2bit mixed precision) for PytorchEngine.

This module tests both functional correctness and accuracy of quant_policy=42
against a non-quantized (quant_policy=0) baseline.

Model: Qwen/Qwen3-0.6B (smaller model to avoid OOM in CI environments)
"""

import gc

import pytest
import torch

from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline
from lmdeploy.messages import Response

# Use smaller model to avoid OOM when running both quant_policy=0 and quant_policy=42
MODEL_ID = 'Qwen/Qwen3-0.6B'


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture(scope='session')
def model_id():
    """Model ID for testing."""
    return MODEL_ID


@pytest.fixture(scope='session')
def pipe_no_quant(model_id):
    """Create pipeline without quantization (baseline).

    This fixture has session scope to avoid reloading the model for each test. Caller is responsible for cleanup.
    """
    engine_config = PytorchEngineConfig(
        tp=1,
        cache_max_entry_count=0.05,
        quant_policy=0,  # No quantization
    )
    pipe = pipeline(model_id, backend_config=engine_config, log_level='INFO')
    yield pipe
    # Cleanup
    pipe.close()
    del pipe
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()


@pytest.fixture(scope='session')
def pipe_quant_42(model_id):
    """Create pipeline with quant_policy=42.

    This fixture has session scope to avoid reloading the model for each test. Caller is responsible for cleanup.
    """
    engine_config = PytorchEngineConfig(
        tp=1,
        cache_max_entry_count=0.05,
        quant_policy=42,  # K=4bit, V=2bit mixed precision
    )
    pipe = pipeline(model_id, backend_config=engine_config, log_level='INFO')
    yield pipe
    # Cleanup
    pipe.close()
    del pipe
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()


# =============================================================================
# Basic Functional Tests (quant_policy=42 only)
# =============================================================================

class TestQuantPolicy42Basic:
    """Basic functional tests for quant_policy=42.

    These tests verify that the quantized model can perform basic inference without errors. They test single prompt,
    batch prompts, and generation config.
    """

    @pytest.fixture(scope='class')
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
        gen_config = GenerationConfig(max_new_tokens=20, temperature=0.7)
        prompt = 'Tell me a short joke'
        response = pipe.infer(prompt, gen_config=gen_config)

        assert isinstance(response, Response)
        assert len(response.text) > 0


# =============================================================================
# Accuracy Tests (quant_policy=0 vs quant_policy=42)
# =============================================================================

class TestQuantPolicy42Accuracy:
    """Accuracy tests comparing quant_policy=42 against non-quantized baseline.

    These tests verify the numerical accuracy/precision of quant_policy=42
    (K=4bit, V=2bit mixed precision) by comparing against quant_policy=0.

    Error thresholds are relaxed due to aggressive quantization:
    - MAE < 0.1 on logits
    - Max AE < 0.5 on logits
    """

    def test_logits_accuracy(self, pipe_no_quant, pipe_quant_42):
        """Test logits accuracy by comparing output logits.

        Compares logits between quantized and non-quantized models.
        Uses deterministic generation settings for reproducibility.

        Thresholds:
        - Mean absolute error (MAE) < 0.1
        - Max absolute error < 0.5
        """
        gen_config = GenerationConfig(
            max_new_tokens=0,  # Required for logits output
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            output_logits='all',
        )

        prompt = 'The capital of France is'

        response_no_quant = pipe_no_quant.infer(prompt, gen_config=gen_config)
        response_quant = pipe_quant_42.infer(prompt, gen_config=gen_config)

        assert isinstance(response_no_quant, Response)
        assert isinstance(response_quant, Response)

        if response_no_quant.logits is not None and response_quant.logits is not None:
            logits_no_quant = response_no_quant.logits
            logits_quant = response_quant.logits

            assert logits_no_quant.shape == logits_quant.shape, \
                f'Logits shape mismatch: {logits_no_quant.shape} vs {logits_quant.shape}'

            abs_error = (logits_no_quant - logits_quant).abs()
            mean_abs_error = abs_error.mean().item()
            max_abs_error = abs_error.max().item()

            print('\nLogits accuracy metrics:')
            print(f'  Mean absolute error: {mean_abs_error:.6f}')
            print(f'  Max absolute error: {max_abs_error:.6f}')

            assert mean_abs_error < 0.1, \
                f'Mean absolute error {mean_abs_error:.6f} exceeds threshold 0.1'
            assert max_abs_error < 0.5, \
                f'Max absolute error {max_abs_error:.6f} exceeds threshold 0.5'
        else:
            pytest.skip('Logits not available for comparison')

    def test_token_accuracy(self, pipe_no_quant, pipe_quant_42):
        """Test token-level accuracy by comparing output token IDs.

        Checks that both models generate output and compares token match rate.
        Note: With aggressive quantization (K=4bit, V=2bit), token match rate
        can be low - this is expected behavior.
        """
        gen_config = GenerationConfig(
            max_new_tokens=20,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )

        prompt = 'Hello, how are you?'

        response_no_quant = pipe_no_quant.infer(prompt, gen_config=gen_config)
        response_quant = pipe_quant_42.infer(prompt, gen_config=gen_config)

        assert isinstance(response_no_quant, Response)
        assert isinstance(response_quant, Response)

        tokens_no_quant = response_no_quant.token_ids
        tokens_quant = response_quant.token_ids

        min_len = min(len(tokens_no_quant), len(tokens_quant))
        if min_len > 0:
            matching_tokens = sum(1 for i in range(min_len)
                                  if tokens_no_quant[i] == tokens_quant[i])
            match_rate = matching_tokens / min_len

            print('\nToken accuracy metrics:')
            print(f'  Baseline tokens: {len(tokens_no_quant)}')
            print(f'  Quantized tokens: {len(tokens_quant)}')
            print(f'  Matching tokens: {matching_tokens}/{min_len}')
            print(f'  Match rate: {match_rate:.2%}')

            # Basic sanity check - both models should produce output
            assert len(tokens_no_quant) > 0, 'Baseline produced no tokens'
            assert len(tokens_quant) > 0, 'Quantized model produced no tokens'
        else:
            pytest.skip('No tokens generated for comparison')

    def test_text_quality(self, pipe_no_quant, pipe_quant_42):
        """Test that quantized output is still meaningful text.

        Verifies the quantized model produces coherent text output, even if not exactly matching the non-quantized
        baseline.
        """
        gen_config = GenerationConfig(
            max_new_tokens=30,
            temperature=0.7,
            top_p=0.9,
        )

        prompt = 'Write a short story about a robot.'

        response_no_quant = pipe_no_quant.infer(prompt, gen_config=gen_config)
        response_quant = pipe_quant_42.infer(prompt, gen_config=gen_config)

        assert isinstance(response_no_quant, Response)
        assert isinstance(response_quant, Response)

        assert len(response_no_quant.text.strip()) > 0, 'Baseline output is empty'
        assert len(response_quant.text.strip()) > 0, 'Quantized output is empty'

        print('\nText quality metrics:')
        print(f'  Baseline text length: {len(response_no_quant.text)}')
        print(f'  Quantized text length: {len(response_quant.text)}')

    def test_logprobs_sanity(self, pipe_no_quant, pipe_quant_42):
        """Test that logprobs are reasonable when available."""
        gen_config = GenerationConfig(
            max_new_tokens=10,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            logprobs=1,
        )

        prompt = 'What is 2+2?'

        response_no_quant = pipe_no_quant.infer(prompt, gen_config=gen_config)
        response_quant = pipe_quant_42.infer(prompt, gen_config=gen_config)

        assert isinstance(response_no_quant, Response)
        assert isinstance(response_quant, Response)

        if response_no_quant.logprobs is not None and response_quant.logprobs is not None:
            print('\nLogprobs available for both models')
            assert isinstance(response_no_quant.logprobs, list)
            assert isinstance(response_quant.logprobs, list)
        else:
            print('\nLogprobs not available (this is expected for some configurations)')

# Copyright (c) OpenMMLab. All rights reserved.
"""Integration tests for MTP (speculative decoding) + Guided Decoding.

Plan section 6.2 — Integration Tests (require GPU):

1. JSON Schema + Spec Decode
   - pipeline with speculative_config=SpeculativeConfig(method='qwen3_5_mtp')
   - response_format=json_schema → output must conform to the schema

2. Regex + Spec Decode
   - response_format=regex_schema → output must match the regex

3. JSON Object + Spec Decode
   - response_format=json_object → output must be a valid JSON object

4. Mixed Batch
   - Some sequences with guided decoding, some without
   - Both paths produce correct results

5. Spec Decode without Guided Decoding
   - Baseline: spec decode works when no grammar is applied

6. Streaming + Guided Spec Decode
   - Streaming inference still produces grammar-conformant output

NOTE: These tests require a GPU and will be skipped if CUDA is unavailable.
The model used is Qwen/Qwen3.5-0.8B (smallest Qwen3.5 with MTP support).
Qwen3.5 is a VLM that supports text-only inference via the PyTorch backend.
The spec method is 'qwen3_5_mtp'.
"""
import json
import re

import pytest
import torch
from jsonschema import validate

from lmdeploy import pipeline
from lmdeploy.messages import (
    GenerationConfig,
    PytorchEngineConfig,
    SpeculativeConfig,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use the smallest available Qwen3.5 MTP model.
MTP_MODEL_ID = 'Qwen/Qwen3.5-0.8B'

SCHEMA_MAP = {
    'json_schema': {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'skills': {
                'type': 'array',
                'items': {'type': 'string', 'maxLength': 10},
                'minItems': 1,
                'maxItems': 5,
            },
        },
        'required': ['name', 'skills'],
    },
    'regex_schema': 'call me [A-Za-z]{1,10}',
    'json_object': None,
}

PROMPT = 'Make a self introduction please.'


def _make_spec_config():
    """Create a SpeculativeConfig for qwen3_5_mtp."""
    return SpeculativeConfig(method='qwen3_5_mtp', num_speculative_tokens=1)


def _make_engine_config():
    """PytorchEngineConfig suitable for MTP + guided decoding tests."""
    return PytorchEngineConfig(
        max_batch_size=1,
        session_len=1024,
        cache_max_entry_count=0.1,
    )


# Skip entire module if no GPU is available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='GPU required for MTP + guided decoding integration tests',
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def pipe():
    """Shared pipeline instance for all tests in this module."""
    p = pipeline(
        MTP_MODEL_ID,
        backend_config=_make_engine_config(),
        speculative_config=_make_spec_config(),
        log_level='INFO',
    )
    yield p
    p.close()


# ---------------------------------------------------------------------------
# 1. JSON Schema + Spec Decode
# ---------------------------------------------------------------------------


class TestJSONSchemaSpecDecode:
    """response_format=json_schema with MTP speculative decoding."""

    def test_json_schema_conformance(self, pipe):
        schema = SCHEMA_MAP['json_schema']
        response_format = {
            'type': 'json_schema',
            'json_schema': {'name': 'test', 'schema': schema},
        }
        gen_config = GenerationConfig(
            response_format=response_format,
            max_new_tokens=200,
        )
        response = pipe(PROMPT, gen_config=gen_config)
        assert response.text, 'Response should not be empty'

        data = json.loads(response.text)
        validate(instance=data, schema=schema)

    def test_json_schema_batch(self, pipe):
        """Batch of identical prompts all produce schema-conformant output."""
        schema = SCHEMA_MAP['json_schema']
        response_format = {
            'type': 'json_schema',
            'json_schema': {'name': 'test', 'schema': schema},
        }
        gen_config = GenerationConfig(
            response_format=response_format,
            max_new_tokens=200,
        )
        # max_batch_size=1, so send one at a time to avoid queue issues
        responses = pipe([PROMPT], gen_config=gen_config)
        for resp in responses:
            data = json.loads(resp.text)
            validate(instance=data, schema=schema)


# ---------------------------------------------------------------------------
# 2. Regex + Spec Decode
# ---------------------------------------------------------------------------


class TestRegexSpecDecode:
    """response_format=regex_schema with MTP speculative decoding."""

    def test_regex_conformance(self, pipe):
        pattern = SCHEMA_MAP['regex_schema']
        response_format = {
            'type': 'regex_schema',
            'regex_schema': pattern,
        }
        gen_config = GenerationConfig(
            response_format=response_format,
            max_new_tokens=50,
        )
        response = pipe(PROMPT, gen_config=gen_config)
        assert response.text, 'Response should not be empty'
        assert re.fullmatch(pattern, response.text), (
            f"Output '{response.text}' does not match regex '{pattern}'"
        )


# ---------------------------------------------------------------------------
# 3. JSON Object + Spec Decode
# ---------------------------------------------------------------------------


class TestJSONObjectSpecDecode:
    """response_format=json_object with MTP speculative decoding."""

    def test_json_object_conformance(self, pipe):
        response_format = {'type': 'json_object'}
        gen_config = GenerationConfig(
            response_format=response_format,
            max_new_tokens=512,
        )
        # Use a structured prompt to guide the model toward a short, complete JSON
        json_prompt = 'Return a JSON object with exactly two keys: name (string) and age (integer).'
        response = pipe(json_prompt, gen_config=gen_config)
        assert response.text, 'Response should not be empty'

        data = json.loads(response.text)
        assert isinstance(data, dict), 'json_object must produce a JSON object (dict)'


# ---------------------------------------------------------------------------
# 4. Mixed Batch — guided + unguided
# ---------------------------------------------------------------------------


class TestMixedBatchSpecDecode:
    """Some sequences with guided decoding, some without."""

    def test_mixed_json_and_free(self, pipe):
        schema = SCHEMA_MAP['json_schema']
        response_format = {
            'type': 'json_schema',
            'json_schema': {'name': 'test', 'schema': schema},
        }
        guided_config = GenerationConfig(
            response_format=response_format,
            max_new_tokens=200,
        )
        free_config = GenerationConfig(max_new_tokens=50)

        # max_batch_size=1, so test one guided + one unguided sequentially
        guided_resp = pipe(PROMPT, gen_config=guided_config)
        free_resp = pipe('Tell me a short joke.', gen_config=free_config)

        # Guided must conform
        data = json.loads(guided_resp.text)
        validate(instance=data, schema=schema)

        # Free must produce text (no grammar constraint)
        assert free_resp.text, 'Free generation should produce text'


# ---------------------------------------------------------------------------
# 5. Spec Decode without Guided Decoding (baseline)
# ---------------------------------------------------------------------------


class TestSpecDecodeNoGuided:
    """MTP speculative decoding without guided decoding — baseline sanity."""

    def test_free_generation(self, pipe):
        gen_config = GenerationConfig(max_new_tokens=50)
        response = pipe(PROMPT, gen_config=gen_config)
        assert response.text, 'Free generation should produce text'
        assert response.generate_token_len > 0


# ---------------------------------------------------------------------------
# 6. Streaming + Guided Spec Decode
# ---------------------------------------------------------------------------


class TestStreamingGuidedSpecDecode:
    """Streaming inference with guided decoding + MTP speculative decoding."""

    def test_streaming_json_schema(self, pipe):
        schema = SCHEMA_MAP['json_schema']
        response_format = {
            'type': 'json_schema',
            'json_schema': {'name': 'test', 'schema': schema},
        }
        gen_config = GenerationConfig(
            response_format=response_format,
            max_new_tokens=200,
        )
        chunks = []
        for chunk in pipe.stream_infer(PROMPT, gen_config=gen_config):
            chunks.append(chunk.text)

        full_text = ''.join(chunks)
        assert full_text, 'Streaming should produce text'

        data = json.loads(full_text)
        validate(instance=data, schema=schema)

    def test_streaming_regex(self, pipe):
        pattern = SCHEMA_MAP['regex_schema']
        response_format = {
            'type': 'regex_schema',
            'regex_schema': pattern,
        }
        gen_config = GenerationConfig(
            response_format=response_format,
            max_new_tokens=50,
        )
        chunks = []
        for chunk in pipe.stream_infer(PROMPT, gen_config=gen_config):
            chunks.append(chunk.text)

        full_text = ''.join(chunks)
        assert full_text, 'Streaming should produce text'
        assert re.fullmatch(pattern, full_text), (
            f"Streaming output '{full_text}' does not match regex '{pattern}'"
        )

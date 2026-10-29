# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Adapted from Intern-Decision (0a8a2388), verify_lmdeploy_decisions.py.
# Modified for LMDeploy: compare shared native/OpenAI image preprocessing.
"""Real-checkpoint checks of shared preprocessing and read-only metadata."""
import asyncio
import base64
import os
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from PIL import Image


@pytest.fixture(scope='module')
def checkpoint():
    path = os.getenv('DECISION_TEST_CHECKPOINT')
    if not path:
        pytest.skip('Set DECISION_TEST_CHECKPOINT to run the Intern-Decision preprocessing tests')
    return path


@pytest.fixture(scope='module')
def preprocessing_engine(checkpoint):
    from transformers import AutoConfig

    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.model import get_chat_template
    from lmdeploy.serve.processors import MultimodalProcessor
    from lmdeploy.tokenizer import Tokenizer
    from lmdeploy.vl.engine import ImageEncoder
    # Only CPU preprocessing is exercised here; avoid a CUDA BF16 capability probe.
    backend_config = PytorchEngineConfig(dtype='float16')
    tokenizer = Tokenizer(checkpoint)
    vision = ImageEncoder(checkpoint, 'pytorch', backend_config=backend_config)
    engine = SimpleNamespace(backend='pytorch', backend_config=backend_config, speculative_config=None,
                             session_len=8192, hf_cfg=AutoConfig.from_pretrained(checkpoint),
                             model_name=checkpoint, tokenizer=tokenizer, vl_encoder=vision)
    engine.prompt_processor = MultimodalProcessor(tokenizer, get_chat_template(checkpoint), vision, backend='pytorch')
    yield engine
    vision.executor.shutdown(wait=True)


def test_context_is_read_only(checkpoint, preprocessing_engine):
    from lmdeploy.serve.parsers.decision_parser import create_decision_context

    context = create_decision_context(preprocessing_engine, checkpoint)
    with pytest.raises(FrozenInstanceError):
        context.temperature = 1.
    with pytest.raises(TypeError):
        context.symbol_ids['A'] = -1


@pytest.mark.parametrize('image_count', [1, 2])
def test_openai_images_match_native_preprocessing(checkpoint, tmp_path, preprocessing_engine, image_count):
    from lmdeploy.serve.parsers.decision_parser import DecisionParser, create_decision_context

    for name, size, color in (('red', (256, 256), (240, 20, 20)), ('blue', (320, 384), (20, 20, 240))):
        with Image.new('RGB', size, color) as image:
            image.save(tmp_path / f'{name}.png')
    images = ['data:image/png;base64,' + base64.b64encode((tmp_path / f'{color}.png').read_bytes()).decode()
              for color in ('red', 'blue')[:image_count]]
    config = create_decision_context(preprocessing_engine, checkpoint)
    body = dict(model=preprocessing_engine.model_name, input=[dict(role='user', content=[
        dict(type='input_text', text='Inspect these images.'),
        *[dict(type='input_image', image_url=image) for image in images],
    ])], questions=[dict(type='predicate', name='red', instructions='Is the first image red?')])
    native = dict(state=[dict(role='user', content=[
        dict(type='text', text='Inspect these images.'),
        *[dict(type='image', image_index=i + 1) for i in range(image_count)],
    ])], images=images, questions={'red': dict(type='noul', instructions='Is the first image red?')})

    async def run():
        results = []
        for request in (body, native):
            parser = DecisionParser(request, config)
            inputs = parser.prepare_input()
            result = await preprocessing_engine.prompt_processor.get_prompt_input(
                inputs.messages, **inputs.preprocess_kwargs)
            ids = result['input_ids']
            prepared = SimpleNamespace(inputs=result, gen_config=inputs.gen_config)
            parser.finalize_input(prepared)
            assert config.image_token_id in ids
            rendered = preprocessing_engine.tokenizer.decode(ids)
            for i in range(image_count):
                assert f'Picture {i + 1}:' in rendered
            results.append((ids, prepared.gen_config.logprob_start_len, parser.score_row_indices))
        assert results[0] == results[1]

    asyncio.run(run())

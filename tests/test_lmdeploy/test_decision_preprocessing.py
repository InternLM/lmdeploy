# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Adapted from Intern-Decision (0a8a2388), verify_lmdeploy_decisions.py.
# Modified for LMDeploy: compare shared native/OpenAI image preprocessing.
"""Real-checkpoint checks of shared preprocessing and read-only metadata."""
import asyncio
import base64
import os
from dataclasses import FrozenInstanceError
from io import BytesIO
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


@pytest.mark.parametrize('systemone', [False, True])
def test_wire_usage_counts_skeleton_with_checkpoint_tokenizer(checkpoint, preprocessing_engine, systemone):
    from lmdeploy.serve.core.async_engine import GenOut
    from lmdeploy.serve.openai.decision_protocol import SystemoneRequest
    from lmdeploy.serve.parsers.decision_parser import DecisionParser, create_decision_context

    request = (SystemoneRequest(state='text', questions={'q': dict(type='noul', instructions='Check')})
               if systemone else dict(model='local', input='text', questions=[
                   dict(type='predicate', name='q', instructions='Check')]))
    context = create_decision_context(preprocessing_engine, checkpoint)
    parser = DecisionParser(request, context)
    inputs = parser.prepare_input(tokenizer=preprocessing_engine.tokenizer)
    result = asyncio.run(preprocessing_engine.prompt_processor.get_prompt_input(
        inputs.messages, **inputs.preprocess_kwargs))
    parser.finalize_input(SimpleNamespace(inputs=result, gen_config=inputs.gen_config))
    rows = [{token: 0. for token in context.symbol_ids.values()} for _ in range(parser.expected_score_rows)]
    output = GenOut(response='', input_token_len=len(result['input_ids']), generate_token_len=0,
                    logprobs=rows, finish_reason='length')
    usage = parser.parse_output(output, model_name='local')['usage']
    # An independent literal fixes the expected content and excludes chat wrappers.
    expected = len(preprocessing_engine.tokenizer.model.model.encode(
        '{\n    "q": "<decision>"\n}', add_special_tokens=False))
    assert usage['output_tokens'] == expected > 0
    assert usage['input_tokens'] == len(result['input_ids'])
    if not systemone:
        assert usage['total_tokens'] == len(result['input_ids']) + expected


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
        from lmdeploy.intern_decision_encoding import compile_row

        parser = DecisionParser(body, config)
        inputs = parser.prepare_input(tokenizer=preprocessing_engine.tokenizer)
        for messages in (inputs.messages, compile_row(native).messages):
            result = await preprocessing_engine.prompt_processor.get_prompt_input(
                messages, **inputs.preprocess_kwargs)
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


@pytest.mark.parametrize('image_count', [1, 6])
def test_systemone_assets_keep_positions_and_visual_budget(checkpoint, preprocessing_engine, image_count):
    from lmdeploy.serve.openai.decision_protocol import SystemoneRequest
    from lmdeploy.serve.parsers.decision_parser import DecisionParser, create_decision_context

    def asset(color):
        stream = BytesIO()
        Image.new('RGB', (1568, 1568), color).save(stream, format='PNG')
        return dict(format='png', data_base64=base64.b64encode(stream.getvalue()).decode())

    row = dict(state='before red-ref after', assets={'red-ref': asset('red')}, questions={
        'q': dict(type='noul', instructions='Red?')})
    if image_count == 6:
        row['assets']['blue-ref'] = asset('blue')
        row['state'] = ['before red-ref after', 'blue-ref']
        row['questions']['q']['instructions'] = 'Compare red-ref with blue-ref'
        row['questions']['level'] = dict(type='score', criteria=['red-ref', 'blue-ref'])
    context = create_decision_context(preprocessing_engine, checkpoint)
    parser = DecisionParser(SystemoneRequest.model_validate(row), context)
    inputs = parser.prepare_input(tokenizer=preprocessing_engine.tokenizer)
    content = inputs.messages[1]['content']
    images = [part['image_data']['data'] for part in content if part['type'] == 'image_data']
    assert len(images) == image_count
    assert [image.size for image in images] == ([(1024, 1024)] if image_count == 1 else [(832, 832)] * 6)
    before = [(image.mode, image.size, image.tobytes()) for image in images]
    if image_count == 6:
        assert images[0] is images[2] is images[4]
        assert images[1] is images[3] is images[5]

    async def run():
        result = await preprocessing_engine.prompt_processor.get_prompt_input(
            inputs.messages, **inputs.preprocess_kwargs)
        parser.finalize_input(SimpleNamespace(inputs=result, gen_config=inputs.gen_config))
        ids = result['input_ids']
        visual_tokens = sum(token == context.image_token_id for token in ids)
        assert visual_tokens == (1024 if image_count == 1 else 676 * 6)
        rendered = preprocessing_engine.tokenizer.decode(ids)
        assert rendered.index('before ') < rendered.index('Picture 1:') < rendered.index(' after')
        assert 'ASSET_' not in rendered
        assert len(parser.score_row_indices) == len(row['questions'])
        assert [(image.mode, image.size, image.tobytes()) for image in images] == before

    asyncio.run(run())


@pytest.mark.parametrize('with_input_image', [False, True])
def test_openai_instruction_images_match_explicit_multimodal_prompt(checkpoint, preprocessing_engine, with_input_image):
    from lmdeploy.intern_decision_encoding import compile_row
    from lmdeploy.serve.parsers.decision_parser import DecisionParser, create_decision_context

    with BytesIO() as buffer, Image.new('RGB', (320, 320), 'blue') as image:
        image.save(buffer, format='PNG')
        url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    input_value = ([dict(role='user', content=[dict(type='input_image', image_url=url)])]
                   if with_input_image else 'Evidence')
    body = dict(model='local', input=input_value, questions=[dict(type='predicate', name='q', instructions=[
        dict(type='input_text', text='Before '), dict(type='input_image', image_url=url),
        dict(type='input_text', text=' after')])])
    context = create_decision_context(preprocessing_engine, checkpoint)
    parser = DecisionParser(body, context)
    inputs = parser.prepare_input(tokenizer=preprocessing_engine.tokenizer)
    assert parser.has_images
    state = ([dict(role='user', content=[dict(type='image', image_index=1)])]
             if with_input_image else 'Evidence')
    reference = compile_row(dict(state=state, images=[url] if with_input_image else [], questions={
        'q': dict(type='noul', instructions='Before <question-image> after')})).messages
    content = reference[1]['content']
    content = content if isinstance(content, list) else [dict(type='text', text=content)]
    expanded = []
    for part in content:
        if part['type'] == 'text':
            before, after = part['text'].split('<question-image>')
            expanded.extend([dict(type='text', text=before), dict(type='image_url', image_url=dict(url=url)),
                             dict(type='text', text=after)])
        else:
            expanded.append(part)
    reference[1]['content'] = expanded

    async def run():
        actual = await preprocessing_engine.prompt_processor.get_prompt_input(
            inputs.messages, **inputs.preprocess_kwargs)
        expected = await preprocessing_engine.prompt_processor.get_prompt_input(reference, **inputs.preprocess_kwargs)
        assert actual['input_ids'] == expected['input_ids']
        parser.finalize_input(SimpleNamespace(inputs=actual, gen_config=inputs.gen_config))
        assert actual['input_ids'].count(context.image_token_id) == 100 * (1 + with_input_image)
        assert len(parser.score_row_indices) == 1
        assert inputs.gen_config.logprob_start_len == actual['input_ids'].index(context.marker_id) - 1
        assert 'ASSET_' not in preprocessing_engine.tokenizer.decode(actual['input_ids'])

    asyncio.run(run())

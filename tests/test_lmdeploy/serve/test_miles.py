import asyncio
from types import SimpleNamespace

import httpx
import pytest
import torch
from fastapi import FastAPI
from pydantic import ValidationError

from lmdeploy.pytorch.engine.logits_process import FusedLogitsProcessor, SamplingInputs
from lmdeploy.serve.openai.endpoints.miles import install_miles_protocol
from lmdeploy.serve.openai.protocol import GenerateReqInput


def test_miles_sampling_translation():
    request = GenerateReqInput(input_ids=[1, 2], sampling_params={
        'max_new_tokens': 8192, 'temperature': 0.8, 'top_p': 1.0,
        'top_k': -1, 'no_stop_trim': True})
    assert (request.max_tokens, request.temperature, request.top_k) == (8192, 0.8, 0)
    assert request.include_stop_str_in_output
    with pytest.raises(ValidationError, match='Conflicting'):
        GenerateReqInput(input_ids=[1], temperature=1.0, sampling_params={'temperature': 0.8})
    with pytest.raises(ValidationError, match='Unsupported'):
        GenerateReqInput(input_ids=[1], sampling_params={'sampling_seed': 1})
    with pytest.raises(ValidationError, match='must be an object'):
        GenerateReqInput(input_ids=[1], sampling_params=None)


@pytest.mark.asyncio
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
async def test_temperature_logprobs_preserve_model_scores(dtype):
    scores = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, -1.0]], dtype=dtype)
    temperature = torch.tensor([0.8, 1.2])
    inputs = SamplingInputs(temperature=temperature, max_num_logprobs=1,
                            logits_processors=[[], []], batch_size=2)
    processor = FusedLogitsProcessor(inputs, logprobs_mode='temperature_logprobs')
    processed, logprobs = await processor(scores.clone())
    expected = scores.float() / temperature[:, None]
    torch.testing.assert_close(processed, expected)
    torch.testing.assert_close(logprobs, expected.log_softmax(-1))
    assert logprobs.dtype == torch.float32
    assert not torch.allclose(logprobs, scores.float().log_softmax(-1))


class FakeEngine:
    backend = 'pytorch'

    def __init__(self):
        self.backend_config = SimpleNamespace(tp=1, logprobs_mode='temperature_logprobs')
        self.sleeping_tags = set()
        self.updates = []
        self.engine = self

    @property
    def is_sleeping(self):
        return bool(self.sleeping_tags)

    async def sleep(self, level):
        self.sleeping_tags = {'weights', 'kv_cache'}

    def wakeup(self, tags=None):
        self.sleeping_tags -= set(tags or self.sleeping_tags)

    def update_params(self, request):
        self.updates.append(request)

    def complete_weights_update(self):
        self.sleeping_tags.discard('weights')


@pytest.mark.asyncio
async def test_update_transaction_blocks_generation_until_weights_and_kv_are_ready():
    engine = FakeEngine()
    context = SimpleNamespace(async_engine=engine, engine_config=engine.backend_config, health_monitor=None)
    app = FastAPI()

    @app.post('/generate')
    async def generate():
        return {'version': context.miles_state.weight_version}

    install_miles_protocol(app, context, model_path='test-model')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        assert (await client.post('/update_weight_version', json={'new_version': '9'})).status_code == 409
        assert (await client.post('/begin_weight_update', json={})).status_code == 200
        assert (await client.post('/generate')).status_code == 503
        assert (await client.post('/end_weight_update', json={})).status_code == 409
        assert (await client.post('/update_weights_from_tensor', json={
            'serialized_named_tensors': ['bucket'], 'load_format': None})).status_code == 200
        assert (await client.post('/end_weight_update', json={})).status_code == 200
        assert engine.updates[-1].finished
        assert engine.sleeping_tags == {'kv_cache'}
        assert (await client.post('/continue_generation', json={})).status_code == 409
        assert (await client.post('/update_weight_version', json={'new_version': '2'})).status_code == 200
        assert (await client.post('/continue_generation', json={})).status_code == 200
        assert (await client.post('/generate')).status_code == 503
        await client.post('/resume_memory_occupation', json={'tags': ['kv_cache', 'cuda_graph']})
        assert (await client.post('/generate')).json() == {'version': '2'}
        assert (await client.get('/model_info')).json()['weight_version'] == '2'


@pytest.mark.asyncio
async def test_pause_drains_existing_generations_and_rejects_new_ones():
    engine = FakeEngine()
    context = SimpleNamespace(async_engine=engine, engine_config=engine.backend_config, health_monitor=None)
    app = FastAPI()
    entered, finish = asyncio.Event(), asyncio.Event()

    @app.post('/generate')
    async def generate():
        entered.set()
        await finish.wait()
        return {'ok': True}

    install_miles_protocol(app, context, model_path='test-model')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        running = asyncio.create_task(client.post('/generate'))
        await entered.wait()
        pause = asyncio.create_task(client.post('/pause_generation', json={'mode': 'retract'}))
        while not context.miles_state.paused:
            await asyncio.sleep(0)
        assert not pause.done()
        assert (await client.post('/generate')).status_code == 503
        finish.set()
        assert (await running).status_code == 200
        assert (await pause).status_code == 200

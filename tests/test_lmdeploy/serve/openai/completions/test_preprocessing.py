import asyncio
import json
from types import SimpleNamespace

from fastapi import APIRouter

from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.endpoints.completions import register
from lmdeploy.serve.openai.protocol import CompletionRequest


class _Session:

    async def async_abort(self):
        pass


class _SessionManager:

    def __init__(self):
        self.removed = []

    def has(self, session_id):
        return False

    def remove(self, session):
        self.removed.append(session)


class _RawRequest:

    async def json(self):
        return {}

    async def is_disconnected(self):
        return False


def test_multi_prompt_preprocessing_is_concurrent_and_fails_as_one_request():

    class _AsyncEngine:
        model_name = 'fake-model'

        def __init__(self):
            self.started = 0
            self.all_started = asyncio.Event()
            self.generate_calls = 0

        async def preprocess(self, prompt, session, **kwargs):
            self.started += 1
            if self.started == 2:
                self.all_started.set()
            await asyncio.wait_for(self.all_started.wait(), timeout=1)
            if prompt == 'bad':
                raise RequestError(ErrorCode.PREPROCESS_FAILED)
            return SimpleNamespace(inputs={'input_ids': [1]}, input_token_len=1)

        def generate(self, request, **kwargs):
            self.generate_calls += 1
            raise AssertionError('generation must not start after a preprocessing failure')

    class _ServerContext:

        def __init__(self):
            self.async_engine = _AsyncEngine()
            self.engine_config = SimpleNamespace(logprobs_mode=None)
            self.session_manager = _SessionManager()
            self.default_gen_config = {}
            self.sessions = []

        def create_session(self, session_id=None):
            session = _Session()
            self.sessions.append(session)
            return session

    async def _run():
        context = _ServerContext()
        router = APIRouter()
        register(router, context)
        endpoint = router.routes[0].endpoint
        response = await endpoint(
            CompletionRequest(
                model='fake-model',
                prompt=['good', 'bad'],
                max_tokens=1,
            ),
            _RawRequest(),
        )
        return response, context

    response, context = asyncio.run(_run())
    body = json.loads(response.body)
    assert response.status_code == 400
    assert body['code'] == 400
    assert context.async_engine.generate_calls == 0
    assert context.session_manager.removed == context.sessions

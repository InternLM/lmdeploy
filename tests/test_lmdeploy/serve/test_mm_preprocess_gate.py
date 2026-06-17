import asyncio
from collections import defaultdict

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.pytorch.engine.mp_engine.base import MPEngineInstance, SessionState
from lmdeploy.serve.processors import MultimodalProcessor
from lmdeploy.serve.utils.mm_preprocess import MultimodalPreprocessGate


async def _run_gate_blocks_until_release():
    gate = MultimodalPreprocessGate(1)
    first = await gate.acquire()

    acquire_second = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0)
    assert not acquire_second.done()

    first.release()
    second = await asyncio.wait_for(acquire_second, timeout=1)
    second.release()


def test_mm_preprocess_gate_blocks_until_release():
    asyncio.run(_run_gate_blocks_until_release())


def test_has_multimodal_input_detects_openai_media():
    assert MultimodalProcessor._has_multimodal_input([{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'describe'},
            {'type': 'image_url', 'image_url': {'url': 'file:///tmp/a.png'}},
        ],
    }])
    assert not MultimodalProcessor._has_multimodal_input([{
        'role': 'user',
        'content': [{'type': 'text', 'text': 'hello'}],
    }])
    assert not MultimodalProcessor._has_multimodal_input('hello')


async def _run_generate_releases_gate_on_prompt_error():
    from lmdeploy.metrics.metrics_processor import metrics_processor
    from lmdeploy.metrics.stats import SchedulerStats
    from lmdeploy.serve.core.async_engine import AsyncEngine
    from lmdeploy.serve.managers import SessionManager

    class _FakePromptProcessor:

        async def get_prompt_input(self, **kwargs):
            raise RuntimeError('boom')

    class _FakeRequestLogger:

        def log_prompt(self, *args, **kwargs):
            pass

    old_stats = metrics_processor.scheduler_stats
    metrics_processor.scheduler_stats = SchedulerStats()
    try:
        engine = AsyncEngine.__new__(AsyncEngine)
        engine.session_mgr = SessionManager()
        engine.prompt_processor = _FakePromptProcessor()
        engine.request_logger = _FakeRequestLogger()

        gate = MultimodalPreprocessGate(1)
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'describe'},
                {'type': 'image_url', 'image_url': {'url': 'file:///tmp/a.png'}},
            ],
        }]

        generator = engine.generate(messages, 260606, mm_preprocess_gate=gate)
        out = await generator.__anext__()
        assert out.finish_reason == 'error'

        lease = await asyncio.wait_for(gate.acquire(), timeout=1)
        lease.release()
    finally:
        metrics_processor.scheduler_stats = old_stats


def test_generate_releases_mm_gate_on_prompt_error():
    asyncio.run(_run_generate_releases_gate_on_prompt_error())


class _FakeMPEngine:

    def __init__(self):
        self.session_states = defaultdict(SessionState)
        self.pending_cancel_sessions = set()
        self.forwarded_kwargs = None
        self.local_callback = None

    async def _collective_rpc_streaming_async(self,
                                              func,
                                              init_done,
                                              *args,
                                              local_request_accepted_callback=None,
                                              **kwargs):
        self.forwarded_kwargs = kwargs
        self.local_callback = local_request_accepted_callback
        if local_request_accepted_callback is not None:
            local_request_accepted_callback()
        yield EngineOutput(ResponseType.FINISH, [])


async def _run_mp_engine_keeps_request_accepted_callback_local():
    engine = _FakeMPEngine()
    instance = MPEngineInstance(engine)
    called = False

    def on_handoff():
        nonlocal called
        called = True

    async for _ in instance.async_stream_infer(260606, [1, 2], local_request_accepted_callback=on_handoff):
        break

    assert called
    assert engine.local_callback is on_handoff
    assert 'local_request_accepted_callback' not in engine.forwarded_kwargs


def test_mp_engine_keeps_request_accepted_callback_local():
    asyncio.run(_run_mp_engine_keeps_request_accepted_callback_local())

# tests/test_lmdeploy/serve/test_empty_prompt_guard.py
# Regression tests for the empty-prompt guard. Empty strings, lists, tuples,
# dicts and (after the __len__ generalisation) empty tensors are rejected
# before they reach the engine body. The _determine_gen_config stub reproduces
# the pre-guard ``len(input_ids)`` crash on None so the guard is what prevents
# it, not the engine itself.
import asyncio

import pytest

from lmdeploy.messages import GenerationConfig
from lmdeploy.metrics.metrics_processor import metrics_processor
from lmdeploy.metrics.stats import SchedulerStats
from lmdeploy.serve.core.async_engine import AsyncEngine
from lmdeploy.serve.managers import SessionManager
from lmdeploy.serve.processors import MultimodalProcessor


async def _collect(gen):
    """Drain an async generator so its body (and the guard) actually runs."""
    async for _ in gen:
        pass


def _make_engine():
    """Build an AsyncEngine without a loaded model.

    ``_determine_gen_config`` is stubbed to call ``len(input_ids)`` so the
    pre-guard crash (input_ids left as None) is reproduced — the guard is what
    prevents the TypeError, not the engine.
    """
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.session_mgr = SessionManager()

    def _determine_gen_config(session, input_ids, gen_config=None):
        len(input_ids)  # reproduces the pre-guard TypeError when input_ids is None
        return gen_config or GenerationConfig()

    engine._determine_gen_config = _determine_gen_config
    return engine


def _run_with_metrics(coro):
    old = metrics_processor.scheduler_stats
    metrics_processor.scheduler_stats = SchedulerStats()
    try:
        return asyncio.run(coro)
    finally:
        metrics_processor.scheduler_stats = old


# --- root-cause layer (pure function, no engine) ---

def test_format_prompts_rejects_empty_string():
    with pytest.raises(ValueError):
        MultimodalProcessor.format_prompts('')


def test_format_prompts_rejects_empty_list():
    with pytest.raises(ValueError):
        MultimodalProcessor.format_prompts([])


# --- direct AsyncEngine.generate() path (validator-proven offline SDK path) ---

def test_generate_rejects_empty_string_messages():
    engine = _make_engine()
    with pytest.raises(ValueError):
        _run_with_metrics(_collect(engine.generate(messages='', session_id=0)))


def test_generate_rejects_empty_list_messages():
    engine = _make_engine()
    with pytest.raises(ValueError):
        _run_with_metrics(_collect(engine.generate(messages=[], session_id=0)))


def test_generate_rejects_empty_dict_messages():
    engine = _make_engine()
    with pytest.raises(ValueError):
        _run_with_metrics(_collect(engine.generate(messages={}, session_id=0)))


def test_generate_rejects_empty_input_ids():
    engine = _make_engine()
    with pytest.raises(ValueError):
        _run_with_metrics(
            _collect(engine.generate(messages=None, session_id=0, input_ids=[])))


def test_validate_prompt_rejects_empty_tensor():
    np = pytest.importorskip('numpy')
    with pytest.raises(ValueError):
        MultimodalProcessor.validate_prompt(np.array([], dtype=int), name='input_ids')


# --- sanity: valid input is NOT rejected by the guard ---

def test_validate_prompt_accepts_non_empty_shapes():
    MultimodalProcessor.validate_prompt('hi')
    MultimodalProcessor.validate_prompt([{'role': 'user', 'content': 'x'}])
    MultimodalProcessor.validate_prompt({'role': 'user', 'content': 'x'})
    MultimodalProcessor.validate_prompt(('describe this', 'https://x/y.png'))

# tests/test_lmdeploy/serve/test_empty_prompt_guard.py
# Reproduction for P8-002: empty/falsy prompt slips past the
# `messages is not None` XOR guard in AsyncEngine.generate and crashes in
# `len(input_ids)` (input_ids left as default None) with a TypeError.
# RED on main, GREEN on branch.
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
    """AsyncEngine without a model (pattern from test_session_cleanup.py:207).

    _determine_gen_config is stubbed to reproduce master's `len(input_ids)`
    crash on None (async_engine.py:424 / :566)."""
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.session_mgr = SessionManager()

    def _determine_gen_config(session, input_ids, gen_config=None):
        len(input_ids)  # reproduces master TypeError when input_ids is None
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
    # main: returns [''] (no exception) -> RED ; branch: raises -> GREEN
    with pytest.raises(ValueError):
        MultimodalProcessor.format_prompts('')


def test_format_prompts_rejects_empty_list():
    # main: returns [] (no exception) -> RED ; branch: raises -> GREEN
    with pytest.raises(ValueError):
        MultimodalProcessor.format_prompts([])


# --- direct AsyncEngine.generate() path (validator-proven offline SDK path) ---

def test_generate_rejects_empty_string_messages():
    engine = _make_engine()
    # main: `if messages:` False -> else -> input_ids=None -> len(None) TypeError -> RED
    # branch: guard raises ValueError before session handling -> GREEN
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


# --- sanity: valid input is NOT rejected by the guard ---

def test_validate_prompt_accepts_non_empty_shapes():
    MultimodalProcessor.validate_prompt('hi')
    MultimodalProcessor.validate_prompt([{'role': 'user', 'content': 'x'}])
    MultimodalProcessor.validate_prompt({'role': 'user', 'content': 'x'})
    MultimodalProcessor.validate_prompt(('describe this', 'https://x/y.png'))

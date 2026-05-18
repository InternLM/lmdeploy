import asyncio

import pytest

from lmdeploy.serve.processors import MultimodalProcessor
from lmdeploy.vl.engine import ImageEncoder


def test_prompt_lock_waits_for_executor_job_after_cancellation(monkeypatch):
    """Test cancelled prompt prep keeps the lock until executor work ends."""

    async def run_case():
        loop = asyncio.get_event_loop()
        pending = loop.create_future()

        class FakeChatTemplate:

            def messages2prompt(self, *args, **kwargs):
                return 'hello'

        class FakeTokenizer:

            def encode(self, *args, **kwargs):
                return [1, 2, 3]

        def fake_run_in_executor(*args, **kwargs):
            return pending

        monkeypatch.setattr(loop, 'run_in_executor', fake_run_in_executor)
        processor = MultimodalProcessor(tokenizer=FakeTokenizer(), chat_template=FakeChatTemplate())

        task = asyncio.create_task(
            processor._get_text_prompt_input('hello',
                                             do_preprocess=True,
                                             sequence_start=True,
                                             adapter_name=None))
        await asyncio.sleep(0)
        assert processor.prompt_lock.locked()

        task.cancel()
        await asyncio.sleep(0)
        assert processor.prompt_lock.locked()

        pending.set_result({'prompt': 'hello', 'input_ids': [1, 2, 3]})
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not processor.prompt_lock.locked()

    asyncio.run(run_case())


def test_image_encoder_lock_waits_for_executor_job_after_cancellation(monkeypatch):
    """Test cancelled VL preprocess keeps the lock until executor work ends."""

    async def run_case():
        loop = asyncio.get_event_loop()
        pending = loop.create_future()

        class FakeModel:

            def preprocess(self, messages):
                return messages

        def fake_run_in_executor(*args, **kwargs):
            return pending

        monkeypatch.setattr(loop, 'run_in_executor', fake_run_in_executor)
        encoder = ImageEncoder.__new__(ImageEncoder)
        encoder.model = FakeModel()
        encoder.executor = None
        encoder.executor_lock = asyncio.Lock()
        encoder._uses_new_preprocess = False

        task = asyncio.create_task(encoder.preprocess([{'content': 'hello'}]))
        await asyncio.sleep(0)
        assert encoder.executor_lock.locked()

        task.cancel()
        await asyncio.sleep(0)
        assert encoder.executor_lock.locked()

        pending.set_result([{'content': 'hello'}])
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not encoder.executor_lock.locked()

    asyncio.run(run_case())

# Copyright (c) OpenMMLab. All rights reserved.
import asyncio


class MultimodalPreprocessLease:
    """Lease for one multimodal preprocessing slot."""

    def __init__(self, semaphore: asyncio.Semaphore | None = None):
        self._semaphore = semaphore
        # Multiple cleanup paths can reach release(); only the first one should
        # return the slot to the gate.
        self._released = False

    def release(self):
        """Release the slot once."""
        if self._released:
            return
        self._released = True
        if self._semaphore is not None:
            self._semaphore.release()


class MultimodalPreprocessGate:
    """Admission gate for large multimodal request preparation."""

    def __init__(self, max_concurrency: int = 0):
        self.max_concurrency = max(0, max_concurrency)
        self._semaphore = asyncio.Semaphore(self.max_concurrency) if self.max_concurrency > 0 else None

    @property
    def enabled(self) -> bool:
        """Return whether gating is enabled."""
        return self._semaphore is not None

    async def acquire(self) -> MultimodalPreprocessLease:
        """Acquire a multimodal preprocessing slot."""
        if self._semaphore is None:
            # Return a no-op lease so callers can keep one cleanup path.
            return MultimodalPreprocessLease()
        await self._semaphore.acquire()
        return MultimodalPreprocessLease(self._semaphore)

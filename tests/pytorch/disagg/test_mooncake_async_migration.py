"""Regression: Mooncake async migration must not hang on an unused Future (#4971)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lmdeploy.pytorch.disagg.backend.mooncake import MooncakeMigrationManagement


@pytest.mark.asyncio
async def test_p2p_migrate_async_awaits_executor_not_orphan_future(monkeypatch):
    """With LMDEPLOY_USE_ASYNC_MIGRATION, p2p_migrate must complete when _migrate returns.

    Previously an unresolved asyncio.Future was awaited after run_in_executor,
    so Decode hung forever even though the RDMA transfer finished.
    """
    monkeypatch.setattr(
        'lmdeploy.pytorch.disagg.backend.mooncake.LMDEPLOY_USE_ASYNC_MIGRATION',
        True,
    )

    backend = object.__new__(MooncakeMigrationManagement)
    called = {'n': 0}

    def fake_migrate(_assignment):
        called['n'] += 1

    backend._migrate = fake_migrate  # type: ignore[method-assign]
    assignment = SimpleNamespace(batch=[])

    await asyncio.wait_for(backend.p2p_migrate(assignment), timeout=2.0)
    assert called['n'] == 1


@pytest.mark.asyncio
async def test_p2p_migrate_sync_path_calls_migrate_directly(monkeypatch):
    monkeypatch.setattr(
        'lmdeploy.pytorch.disagg.backend.mooncake.LMDEPLOY_USE_ASYNC_MIGRATION',
        False,
    )

    backend = object.__new__(MooncakeMigrationManagement)
    backend._migrate = MagicMock()
    assignment = SimpleNamespace(batch=[])

    await backend.p2p_migrate(assignment)
    backend._migrate.assert_called_once_with(assignment)

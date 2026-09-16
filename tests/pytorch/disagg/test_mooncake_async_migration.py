# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from unittest.mock import Mock

import pytest

from lmdeploy.pytorch.disagg.backend import mooncake
from lmdeploy.pytorch.disagg.backend.mooncake import MooncakeMigrationManagement


def _make_management():
    mgmt = MooncakeMigrationManagement.__new__(MooncakeMigrationManagement)
    mgmt._migrate = Mock()
    return mgmt


def test_async_p2p_migrate_returns_after_executor_completes(monkeypatch):
    monkeypatch.setattr(mooncake, 'LMDEPLOY_USE_ASYNC_MIGRATION', '1')
    mgmt = _make_management()
    assignment = Mock()

    asyncio.run(asyncio.wait_for(mgmt.p2p_migrate(assignment), timeout=2))

    mgmt._migrate.assert_called_once_with(assignment)


def test_async_p2p_migrate_propagates_migrate_errors(monkeypatch):
    monkeypatch.setattr(mooncake, 'LMDEPLOY_USE_ASYNC_MIGRATION', '1')
    mgmt = _make_management()
    mgmt._migrate.side_effect = RuntimeError('Failed to perform sync transfer: 1')

    with pytest.raises(RuntimeError, match='Failed to perform sync transfer'):
        asyncio.run(asyncio.wait_for(mgmt.p2p_migrate(Mock()), timeout=2))

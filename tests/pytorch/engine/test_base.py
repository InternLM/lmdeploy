# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for lmdeploy.pytorch.engine.base module."""

import pytest

from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeConnectionRequest,
    DistServeDropConnectionRequest,
    DistServeInitRequest,
)
from lmdeploy.pytorch.engine.base import EngineBase, EngineInstanceBase


class TestEngineBase:
    """Test EngineBase class."""

    def test_close_not_implemented(self):
        """Test that close raises NotImplementedError."""
        engine = EngineBase()
        with pytest.raises(NotImplementedError):
            engine.close()

    def test_start_loop_exists(self):
        """Test that start_loop method exists."""
        engine = EngineBase()
        # Should not raise, just a no-op base implementation
        engine.start_loop()

    def test_end_session_not_implemented(self):
        """Test that end_session raises NotImplementedError."""
        engine = EngineBase()
        with pytest.raises(NotImplementedError):
            engine.end_session(session_id=1)

    def test_p2p_initialize_not_implemented(self):
        """Test that p2p_initialize raises NotImplementedError."""
        engine = EngineBase()
        # Use a mock request to avoid Pydantic validation
        mock_request = type('MockRequest', (), {})()
        with pytest.raises(NotImplementedError):
            engine.p2p_initialize(conn_request=mock_request)

    def test_p2p_connect_not_implemented(self):
        """Test that p2p_connect raises NotImplementedError."""
        engine = EngineBase()
        mock_request = type('MockRequest', (), {})()
        with pytest.raises(NotImplementedError):
            engine.p2p_connect(conn_request=mock_request)

    def test_p2p_drop_connect_not_implemented(self):
        """Test that p2p_drop_connect raises NotImplementedError."""
        engine = EngineBase()
        mock_request = type('MockRequest', (), {})()
        with pytest.raises(NotImplementedError):
            engine.p2p_drop_connect(drop_conn_request=mock_request)

    def test_create_instance_not_implemented(self):
        """Test that create_instance raises NotImplementedError."""
        engine = EngineBase()
        with pytest.raises(NotImplementedError):
            engine.create_instance(cuda_stream_id=0)

    def test_get_health_status_not_implemented(self):
        """Test that get_health_status raises NotImplementedError."""
        engine = EngineBase()
        with pytest.raises(NotImplementedError):
            import asyncio
            asyncio.run(engine.get_health_status())

    def test_subclass_can_override_methods(self):
        """Test that subclasses can override methods."""
        class ConcreteEngine(EngineBase):
            def close(self):
                return "closed"

            def end_session(self, session_id):
                return f"session {session_id} ended"

        engine = ConcreteEngine()
        assert engine.close() == "closed"
        assert engine.end_session(1) == "session 1 ended"


class TestEngineInstanceBase:
    """Test EngineInstanceBase class."""

    def test_async_end_not_implemented(self):
        """Test that async_end raises NotImplementedError."""
        instance = EngineInstanceBase()
        with pytest.raises(NotImplementedError):
            import asyncio
            asyncio.run(instance.async_end(session_id=1))

    def test_async_cancel_not_implemented(self):
        """Test that async_cancel raises NotImplementedError."""
        instance = EngineInstanceBase()
        with pytest.raises(NotImplementedError):
            import asyncio
            asyncio.run(instance.async_cancel(session_id=1))

    def test_async_stream_infer_not_implemented(self):
        """Test that async_stream_infer raises NotImplementedError."""
        instance = EngineInstanceBase()
        with pytest.raises(NotImplementedError):
            import asyncio
            asyncio.run(instance.async_stream_infer())

    def test_subclass_can_override_async_methods(self):
        """Test that subclasses can override async methods."""
        class ConcreteInstance(EngineInstanceBase):
            async def async_end(self, session_id):
                return f"session {session_id} ended"

            async def async_cancel(self, session_id):
                return f"session {session_id} cancelled"

            async def async_stream_infer(self, *args, **kwargs):
                return "streaming"

        instance = ConcreteInstance()
        import asyncio
        assert asyncio.run(instance.async_end(1)) == "session 1 ended"
        assert asyncio.run(instance.async_cancel(1)) == "session 1 cancelled"
        assert asyncio.run(instance.async_stream_infer()) == "streaming"


class TestEngineBaseIntegration:
    """Integration tests for EngineBase."""

    def test_engine_base_is_base_class(self):
        """Test that EngineBase is designed as a base class."""
        engine = EngineBase()
        # Should have all the expected methods
        assert hasattr(engine, 'close')
        assert hasattr(engine, 'start_loop')
        assert hasattr(engine, 'end_session')
        assert hasattr(engine, 'p2p_initialize')
        assert hasattr(engine, 'p2p_connect')
        assert hasattr(engine, 'p2p_drop_connect')
        assert hasattr(engine, 'create_instance')
        assert hasattr(engine, 'get_health_status')

    def test_engine_instance_base_is_base_class(self):
        """Test that EngineInstanceBase is designed as a base class."""
        instance = EngineInstanceBase()
        # Should have all the expected async methods
        assert hasattr(instance, 'async_end')
        assert hasattr(instance, 'async_cancel')
        assert hasattr(instance, 'async_stream_infer')

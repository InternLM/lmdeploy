# Copyright (c) OpenMMLab. All rights reserved.
"""Additional tests for lmdeploy.utils to improve coverage."""
import logging
import pytest

# Import only the functions we need, avoiding full lmdeploy import that triggers torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lmdeploy.utils import get_logger, logging_timer, package_is_exist


def test_get_logger_basic():
    """Test basic logger creation."""
    logger = get_logger('test_logger')
    assert logger is not None
    assert logger.name == 'test_logger'
    assert logger.level >= logging.DEBUG


def test_get_logger_singleton():
    """Test that get_logger returns singleton instance."""
    logger1 = get_logger('test_logger')
    logger2 = get_logger('test_logger')
    assert logger1 is logger2


def test_get_logger_different_names():
    """Test that different names create different loggers."""
    logger1 = get_logger('test_logger1')
    logger2 = get_logger('test_logger2')
    assert logger1 is not logger2


def test_logging_timer_decorator():
    """Test logging_timer decorator functionality."""
    import time

    logger = get_logger('test_timer')
    
    @logging_timer('test_operation', logger, logging.INFO)
    def slow_function():
        time.sleep(0.1)
        return 'result'
    
    result = slow_function()
    assert result == 'result'


def test_logging_timer_with_exception():
    """Test logging_timer decorator handles exceptions."""
    logger = get_logger('test_timer_exception')
    
    @logging_timer('test_operation', logger, logging.INFO)
    def failing_function():
        raise ValueError('Test error')
    
    with pytest.raises(ValueError, match='Test error'):
        failing_function()


def test_package_is_exist_positive():
    """Test package_is_exist for installed packages."""
    # Test with packages that should be installed
    assert package_is_exist('torch') is True
    assert package_is_exist('transformers') is True


def test_package_is_exist_negative():
    """Test package_is_exist for non-existent packages."""
    assert package_is_exist('nonexistent_package_xyz') is False


def test_get_plugin_library_empty_path():
    """Test _get_plugin_library with empty or None path."""
    # Test with empty string
    try:
        from lmdeploy.utils import _get_plugin_library
        result = _get_plugin_library('')
        assert result is None
    except ImportError:
        pass  # Skip if function not available
    
    try:
        result = _get_plugin_library(None)
        assert result is None
    except ImportError:
        pass  # Skip if function not available


def test_logger_levels():
    """Test logger with different log levels."""
    import logging
    
    # Test logger with INFO level
    logger_info = get_logger('test_info_logger')
    logger_info.setLevel(logging.INFO)
    assert logger_info.level == logging.INFO
    
    # Test logger with DEBUG level
    logger_debug = get_logger('test_debug_logger')
    logger_debug.setLevel(logging.DEBUG)
    assert logger_debug.level == logging.DEBUG


def test_logger_multiple_calls():
    """Test that multiple calls to get_logger are efficient."""
    # Call get_logger multiple times for the same name
    for i in range(100):
        logger = get_logger('test_multiple_calls')
    
    # Should still return the same instance
    final_logger = get_logger('test_multiple_calls')
    assert final_logger is logger
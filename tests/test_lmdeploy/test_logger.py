# Copyright (c) OpenMMLab. All rights reserved.
import logging

import pytest

import lmdeploy.logger as request_logger_module
from lmdeploy.cli.utils import ArgumentHelper
from lmdeploy.logger import RequestLogger
from lmdeploy.utils import REQUEST_LOG_LEVEL


@pytest.fixture(autouse=True)
def restore_request_logger_level():
    logger = request_logger_module.logger
    old_level = logger.level
    yield
    logger.setLevel(old_level)


def _capture_request_logs(monkeypatch, log_level):
    records = []

    class CaptureHandler(logging.Handler):

        def emit(self, record):
            records.append(record)

    logger = request_logger_module.logger
    handler = CaptureHandler()
    handler.setLevel(logging.DEBUG)
    monkeypatch.setattr(logger, 'handlers', [handler])
    logger.setLevel(log_level)
    return records


def test_request_logger_log_inputs_hidden_at_info(monkeypatch):
    records = _capture_request_logs(monkeypatch, logging.INFO)

    RequestLogger(max_log_len=None).log_inputs(1, 'hello', [1, 2], 'gen_config', 'default')

    assert not records


def test_request_logger_log_inputs_skips_formatting_at_info(monkeypatch):

    class UnformattableConfig:

        def __str__(self):
            raise AssertionError('gen_config should not be formatted')

    _capture_request_logs(monkeypatch, logging.INFO)

    RequestLogger(max_log_len=None).log_inputs(1, 'hello', [1, 2], UnformattableConfig(), 'default')


def test_request_logger_log_inputs_visible_at_request(monkeypatch):
    records = _capture_request_logs(monkeypatch, REQUEST_LOG_LEVEL)

    RequestLogger(max_log_len=None).log_inputs(1, 'hello', [1, 2], 'gen_config', 'default')

    assert len(records) == 1
    assert records[0].levelno == REQUEST_LOG_LEVEL
    assert 'input_tokens=2' in records[0].getMessage()


def test_request_logger_log_inputs_handles_none_token_ids(monkeypatch):
    records = _capture_request_logs(monkeypatch, REQUEST_LOG_LEVEL)

    RequestLogger(max_log_len=None).log_inputs(1, 'hello', None, 'gen_config', 'default')

    assert len(records) == 1
    assert 'input_tokens=0' in records[0].getMessage()
    assert 'prompt_token_id=None' in records[0].getMessage()


def test_request_log_level_includes_info_logs(monkeypatch):
    records = _capture_request_logs(monkeypatch, REQUEST_LOG_LEVEL)

    request_logger_module.logger.info('normal info')

    assert len(records) == 1
    assert records[0].levelno == logging.INFO


def test_log_level_argument_accepts_request():
    import argparse

    parser = argparse.ArgumentParser()
    action = ArgumentHelper.log_level(parser)

    assert 'REQUEST' in action.choices

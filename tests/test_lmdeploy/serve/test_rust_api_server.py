# Copyright (c) OpenMMLab. All rights reserved.

import json
import logging
import os

import pytest

from lmdeploy.serve.rust_api_server import _configure_logging, logger, serve, validate_model_files


def _write_model_files(path, *, model_type='qwen2', with_tokenizer=True, with_template=True):
    if with_tokenizer:
        (path / 'tokenizer.json').write_text('{}', encoding='utf-8')
    tokenizer_config = {'chat_template': '{{ messages }}'} if with_template else {}
    (path / 'tokenizer_config.json').write_text(json.dumps(tokenizer_config), encoding='utf-8')
    (path / 'config.json').write_text(json.dumps({'model_type': model_type}), encoding='utf-8')


def test_validate_requires_hf_tokenizer_json(tmp_path):
    _write_model_files(tmp_path, with_tokenizer=False)
    with pytest.raises(RuntimeError, match='tokenizer.json'):
        validate_model_files(str(tmp_path))


def test_validate_requires_hf_chat_template(tmp_path):
    _write_model_files(tmp_path, with_template=False)
    with pytest.raises(RuntimeError, match='chat_template'):
        validate_model_files(str(tmp_path))


def test_validate_rejects_gpt_oss_before_engine_load(tmp_path):
    _write_model_files(tmp_path, model_type='gpt_oss')
    with pytest.raises(RuntimeError, match='GPT-OSS'):
        validate_model_files(str(tmp_path))


def test_serve_rejects_unknown_parser_before_engine_import(tmp_path):
    _write_model_files(tmp_path)
    with pytest.raises(ValueError, match='tool-call parser'):
        serve(
            str(tmp_path),
            model_name='model',
            backend_config=object(),
            tool_call_parser='unknown',
        )


def test_configure_logging_sets_default_for_python_and_turbomind(monkeypatch):
    monkeypatch.delenv('TM_LOG_LEVEL', raising=False)
    monkeypatch.setattr(logger, 'level', logger.level)

    assert _configure_logging('warning') == 'WARNING'
    assert logger.level == logging.WARNING
    assert os.environ['TM_LOG_LEVEL'] == 'WARNING'


def test_configure_logging_preserves_explicit_turbomind_override(monkeypatch):
    monkeypatch.setenv('TM_LOG_LEVEL', 'DEBUG')
    monkeypatch.setattr(logger, 'level', logger.level)

    assert _configure_logging('ERROR') == 'ERROR'
    assert logger.level == logging.ERROR
    assert os.environ['TM_LOG_LEVEL'] == 'DEBUG'

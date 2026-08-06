# Copyright (c) OpenMMLab. All rights reserved.

"""Bootstrap the in-process Rust/Axum TurboMind API server."""

import json
import logging
import os
from pathlib import Path

from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _configure_logging(log_level: str) -> str:
    """Configure Python and TurboMind before the C++ logger is initialized."""
    normalized = log_level.upper()
    if normalized not in logging._nameToLevel:
        raise ValueError(f'Invalid log level: {log_level}')
    logger.setLevel(normalized)
    tm_level = {
        'CRITICAL': 'FATAL',
        'NOTSET': 'TRACE',
    }.get(normalized, normalized)
    os.environ.setdefault('TM_LOG_LEVEL', tm_level)
    return normalized


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding='utf-8') as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f'Failed to read required Hugging Face file {path}: {error}') from error
    if not isinstance(value, dict):
        raise RuntimeError(f'Expected a JSON object in {path}.')
    return value


def validate_model_files(model_path: str) -> None:
    """Fail before weight export when the Rust frontend cannot load a model."""
    model_dir = Path(model_path)
    tokenizer_json = model_dir / 'tokenizer.json'
    if not tokenizer_json.is_file():
        raise RuntimeError(
            f'rust_api_server requires a Hugging Face tokenizer.json; not found: {tokenizer_json}')

    tokenizer_config_path = model_dir / 'tokenizer_config.json'
    tokenizer_config = _read_json(tokenizer_config_path)
    if not tokenizer_config.get('chat_template'):
        raise RuntimeError(
            'rust_api_server requires chat_template in tokenizer_config.json for /v1/chat/completions.')

    config = _read_json(model_dir / 'config.json')
    model_type = str(config.get('model_type', '')).replace('-', '_').lower()
    architectures = [str(value).lower() for value in config.get('architectures', [])]
    if model_type == 'gpt_oss' or any('gptoss' in value or 'gpt_oss' in value for value in architectures):
        raise RuntimeError('rust_api_server does not support GPT-OSS because Harmony remains Python-only.')


def serve(model_path: str,
          model_name: str,
          backend_config: TurbomindEngineConfig,
          server_name: str = '0.0.0.0',
          server_port: int = 23333,
          log_level: str = 'WARNING',
          api_keys: list[str] | None = None,
          reasoning_parser: str | None = None,
          tool_call_parser: str | None = None,
          trust_remote_code: bool = False) -> None:
    """Load TurboMind in Python, then hand serving to Rust in the same process."""
    validate_model_files(model_path)
    supported_tool_parsers = {
        'qwen', 'qwen2d5', 'qwen3', 'qwen3coder', 'llama3', 'internlm', 'intern-s1',
        'interns2-preview', 'glm47', 'deepseek-v32', 'deepseek-v3.2', 'deepseek-v4'
    }
    if tool_call_parser == 'gpt-oss':
        raise RuntimeError('rust_api_server does not support GPT-OSS/Harmony parsing.')
    if tool_call_parser is not None and tool_call_parser not in supported_tool_parsers:
        raise ValueError(f'Unsupported Rust tool-call parser: {tool_call_parser}')
    supported_reasoning_parsers = {
        'default', 'deepseek-v3', 'deepseek-v32', 'deepseek-v3.2', 'deepseek-v4',
        'qwen-qwq', 'intern-s1', 'deepseek-r1'
    }
    if reasoning_parser is not None and reasoning_parser not in supported_reasoning_parsers:
        raise ValueError(f'Unsupported Rust reasoning parser: {reasoning_parser}')
    if not (0 < server_port < 65536):
        raise ValueError(f'Invalid server port: {server_port}')

    log_level = _configure_logging(log_level)

    from lmdeploy.turbomind.turbomind import TurboMind, _tm

    rust_entry = getattr(_tm, 'rust_api_server', None)
    if rust_entry is None:
        raise RuntimeError(
            'This LMDeploy build does not include rust_api_server. Rebuild with -DBUILD_RUST_API_SERVER=ON.')

    engine = TurboMind(
        model_path=model_path,
        model_name=model_name,
        engine_config=backend_config,
        trust_remote_code=trust_remote_code,
        load_tokenizer=False,
    )
    config = {
        'server_name': server_name,
        'server_port': server_port,
        'model_dir': os.path.abspath(model_path),
        'model_name': model_name,
        'log_level': log_level,
        'api_keys': api_keys or [],
        'reasoning_parser': reasoning_parser,
        'tool_call_parser': tool_call_parser,
    }
    try:
        rust_entry(engine.model_comm, json.dumps(config, ensure_ascii=False).encode('utf-8'))
    except KeyboardInterrupt:
        # Tokio also observes SIGINT and shuts Axum down gracefully. Python may
        # surface the same signal when pybind reacquires the GIL; keep normal
        # CLI shutdown free of a traceback.
        return

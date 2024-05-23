# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import functools
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from logging import Logger, LogRecord
from typing import List, Optional, TypeVar, Union

from huggingface_hub import hf_hub_download
from transformers import PretrainedConfig

logger_initialized = {}


class _ASNI_COLOR:
    BRIGHT_RED = '\033[91m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    WHITE = '\033[37m'
    GREEN = '\033[32m'


class ColorFormatter(logging.Formatter):

    _LEVELNAME_COLOR_MAP = dict(CRITICAL=_ASNI_COLOR.BRIGHT_RED,
                                ERROR=_ASNI_COLOR.RED,
                                WARN=_ASNI_COLOR.YELLOW,
                                WARNING=_ASNI_COLOR.YELLOW,
                                INFO=_ASNI_COLOR.WHITE,
                                DEBUG=_ASNI_COLOR.GREEN)

    _RESET_COLOR = '\033[0m'

    def format(self, record: LogRecord):
        """format."""
        if sys.platform == 'win32':
            # windows does not support ASNI color
            return super().format(record)
        levelname = record.levelname
        level_color = self._LEVELNAME_COLOR_MAP.get(levelname,
                                                    self._RESET_COLOR)
        levelname = f'{level_color}{levelname}{self._RESET_COLOR}'
        record.levelname = levelname
        return super().format(record)


class FilterDuplicateWarning(logging.Filter):
    """Filter the repeated warning message.

    Args:
        name (str): name of the filter.
    """

    def __init__(self, name: str = 'lmdeploy'):
        super().__init__(name)
        self.seen: set = set()

    def filter(self, record: LogRecord) -> bool:
        """Filter the repeated warning message.

        Args:
            record (LogRecord): The log record.

        Returns:
            bool: Whether to output the log record.
        """
        if record.levelno != logging.WARNING:
            return True

        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False


def get_logger(
    name: Optional[str] = None,
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    file_mode: str = 'w',
    log_formatter: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> Logger:
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified, a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
        log_formatter (str): The logger output format.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [stream_handler]

    if log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = ColorFormatter(log_formatter)
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        handler.addFilter(FilterDuplicateWarning(name))
        logger.addHandler(handler)

    logger.setLevel(log_level)
    logger.propagate = False
    logger_initialized[name] = True

    return logger


def filter_suffix(response: str, suffixes: Optional[List[str]] = None) -> str:
    """Filter response with suffixes.

    Args:
        response (str): generated response by LLMs.
        suffixes (str): a list of suffixes to be deleted.

    Return:
        str: a clean response.
    """
    if suffixes is None:
        return response
    for item in suffixes:
        if response.endswith(item):
            response = response[:len(response) - len(item)]
    return response


# TODO remove stop_word_offsets stuff and make it clean
def _stop_words(stop_words: List[str], tokenizer: object):
    """return list of stop-words to numpy.ndarray."""
    import numpy as np
    if stop_words is None:
        return None
    assert isinstance(stop_words, List) and \
        all(isinstance(elem, str) for elem in stop_words), \
        f'stop_words must be a list but got {type(stop_words)}'
    stop_indexes = []
    for stop_word in stop_words:
        stop_indexes += tokenizer.indexes_containing_token(stop_word)
    assert isinstance(stop_indexes, List) and all(
        isinstance(elem, int) for elem in stop_indexes), 'invalid stop_words'
    # each id in stop_indexes represents a stop word
    # refer to https://github.com/fauxpilot/fauxpilot/discussions/165 for
    # detailed explanation about fastertransformer's stop_indexes
    stop_word_offsets = range(1, len(stop_indexes) + 1)
    stop_words = np.array([[stop_indexes, stop_word_offsets]]).astype(np.int32)
    return stop_words


def get_hf_config_content(pretrained_model_name_or_path: str,
                          **kwargs) -> dict:
    """Get config content of a hf model."""
    if os.path.exists(pretrained_model_name_or_path):
        config_path = os.path.join(pretrained_model_name_or_path,
                                   'config.json')
    else:
        config_path = hf_hub_download(pretrained_model_name_or_path,
                                      'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_model(pretrained_model_name_or_path: str,
              download_dir: str = None,
              revision: str = None,
              token: str = None):
    """Get model from huggingface or modelscope."""
    import os
    if os.getenv('LMDEPLOY_USE_MODELSCOPE', 'False').lower() == 'true':
        from modelscope import snapshot_download
    else:
        from huggingface_hub import snapshot_download

    download_kwargs = {}
    if download_dir is not None:
        download_kwargs['cache_dir'] = download_dir
    if revision is not None:
        download_kwargs['revision'] = revision
    if token is not None:
        download_kwargs['token'] = token

    model_path = snapshot_download(pretrained_model_name_or_path,
                                   **download_kwargs)
    return model_path


def logging_timer(op_name: str, logger: Logger, level: int = logging.DEBUG):
    """logging timer."""

    @contextmanager
    def __timer():
        """timer."""
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        duration = (end - start) * 1000
        logger.log(level, f'<{op_name}> take time: {duration:.2f} ms')

    def __inner(func):
        """inner."""

        @functools.wraps(func)
        def __func_warpper(*args, **kwargs):
            """func warpper."""
            if logger.level > level:
                return func(*args, **kwargs)
            with __timer():
                return func(*args, **kwargs)

        @functools.wraps(func)
        def __async_warpper(*args, **kwargs):
            """async warpper."""

            async def __tmp():
                if logger.level > level:
                    return (await func(*args, **kwargs))
                with __timer():
                    return (await func(*args, **kwargs))

            return __tmp()

        if asyncio.iscoroutinefunction(func):
            return __async_warpper
        else:
            return __func_warpper

    return __inner


# copy from https://github.com/vllm-project/vllm/blob/0650e5935b0f6af35fb2acf71769982c47b804d7/vllm/config.py#L1082-L1150  # noqa
def _get_and_verify_max_len(
    hf_tm_config: Union[PretrainedConfig,
                        TypeVar('TurbomindModelConfig')],
    max_model_len: Optional[int],
) -> int:
    """Get and verify the model's maximum length."""
    logger = get_logger('lmdeploy')
    derived_max_model_len = float('inf')
    possible_keys = [
        # OPT
        'max_position_embeddings',
        # GPT-2
        'n_positions',
        # MPT
        'max_seq_len',
        # ChatGLM2
        'seq_length',
        # Command-R
        'model_max_length',
        # Others
        'max_sequence_length',
        'max_seq_length',
        'seq_len',
    ]
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(hf_tm_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len \
                else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)
    if derived_max_model_len == float('inf'):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            'keys to determine the original maximum length of the model: '
            f"{possible_keys}. Assuming the model's maximum length is "
            f'{default_max_len}.')
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_tm_config, 'rope_scaling', None)
    if rope_scaling is not None and rope_scaling['type'] != 'su':
        assert 'factor' in rope_scaling
        scaling_factor = rope_scaling['factor']
        if rope_scaling['type'] == 'yarn':
            derived_max_model_len = rope_scaling[
                'original_max_position_embeddings']
        derived_max_model_len *= scaling_factor

    if max_model_len is None:
        max_model_len = int(derived_max_model_len)
    elif max_model_len > derived_max_model_len:
        # Some models might have a separate key for specifying model_max_length
        # that will be bigger than derived_max_model_len. We compare user input
        # with model_max_length and allow this override when it's smaller.
        model_max_length = getattr(hf_tm_config, 'model_max_length', None)
        if model_max_length is not None and max_model_len <= model_max_length:
            pass
        else:
            logger.warning(
                f'User-specified max_model_len ({max_model_len}) is greater '
                'than the derived max_model_len '
                f'({max_len_key}={derived_max_model_len} or model_max_length='
                f"{model_max_length} in model's config.json).")
    return int(max_model_len)

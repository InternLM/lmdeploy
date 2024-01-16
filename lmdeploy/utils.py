# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

logger_initialized = {}


def get_logger(name: str,
               log_file: Optional[str] = None,
               log_level: int = logging.INFO,
               file_mode: str = 'w'):
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
    Returns:
        logging.Logger: The expected logger.
    """
    # use logger in mmengine if exists.
    try:
        from mmengine.logging import MMLogger
        if MMLogger.check_instance_created(name):
            logger = MMLogger.get_instance(name)
        else:
            logger = MMLogger.get_instance(name,
                                           logger_name=name,
                                           log_file=log_file,
                                           log_level=log_level,
                                           file_mode=file_mode)
        return logger

    except Exception:
        pass

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

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)
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

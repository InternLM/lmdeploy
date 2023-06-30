# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional, Union

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

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


def prepare_tensor(name, input_tensor):
    t = grpcclient.InferInput(name, list(input_tensor.shape),
                              np_to_triton_dtype(input_tensor.dtype))
    t.set_data_from_numpy(input_tensor)
    return t


class Preprocessor:

    def __init__(self, tritonserver_addr: str):
        self.tritonserver_addr = tritonserver_addr
        self.model_name = 'preprocessing'

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def infer(self, prompts: Union[str, List[str]]) -> tuple:
        """Tokenize the input prompts.

        Args:
            prompts(str | List[str]): user's prompt, or a batch prompts

        Returns:
            Tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray): prompt's token
            ids, ids' length and requested output length
        """
        if isinstance(prompts, str):
            input0 = [[prompts]]
        elif isinstance(prompts, List):
            input0 = [[prompt] for prompt in prompts]
        else:
            assert 0, f'str or List[str] prompts are expected but got ' \
                      f'{type(prompts)}'

        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32)
        inputs = [
            prepare_tensor('QUERY', input0_data),
            prepare_tensor('REQUEST_OUTPUT_LEN', output0_len)
        ]

        with grpcclient.InferenceServerClient(self.tritonserver_addr) as \
                client:
            result = client.infer(self.model_name, inputs)
            output0 = result.as_numpy('INPUT_ID')
            output1 = result.as_numpy('REQUEST_INPUT_LEN')
        return output0, output1


class Postprocessor:

    def __init__(self, tritonserver_addr: str):
        self.tritonserver_addr = tritonserver_addr

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def infer(self, output_ids: np.ndarray, seqlen: np.ndarray):
        """De-tokenize tokens for text.

        Args:
            output_ids(np.ndarray): tokens' id
            seqlen(np.ndarray): sequence length

        Returns:
            str: decoded tokens
        """
        inputs = [
            prepare_tensor('TOKENS_BATCH', output_ids),
            prepare_tensor('sequence_length', seqlen)
        ]
        inputs[0].set_data_from_numpy(output_ids)
        inputs[1].set_data_from_numpy(seqlen)
        model_name = 'postprocessing'
        with grpcclient.InferenceServerClient(self.tritonserver_addr) \
                as client:
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy('OUTPUT')
        return output0

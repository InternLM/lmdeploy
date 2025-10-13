# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import inspect
from inspect import Parameter, Signature
from typing import Dict, Sequence

import psutil

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def get_gpu_memory(device_id: int = None) -> int:
    """Returns the free and total physical memory of the GPU in bytes."""
    import torch
    if device_id is None:
        device_id = torch.cuda.current_device()
    return torch.cuda.mem_get_info(device_id)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def bind_sigature(input_names: str, args: Sequence, kwargs: Dict):
    """Bind args and kwargs to given input names."""
    kind = inspect._ParameterKind.POSITIONAL_OR_KEYWORD

    sig = Signature([Parameter(name, kind) for name in input_names])
    bind = sig.bind(*args, **kwargs)
    return bind.arguments


# from vllm
def maybe_register_config_serialize_by_value(trust_remote_code: bool) -> None:
    """Try to register HF model configuration class to serialize by value With
    trust_remote_code, the config class is typically an instance of a custom
    class imported from the HF modules cache.

    The class will not be
    importable in spawned workers by default (and won't exist at all on
    other nodes), which breaks serialization of the config.
    In this function we tell the cloudpickle serialization library to pass
    instances of these generated classes by value instead of by reference,
    i.e. the class definition is serialized along with its data so that the
    class module does not need to be importable on the receiving end. This
    registration only works if the modules cache has already been
    initialized.
    See: https://github.com/cloudpipe/cloudpickle?tab=readme-ov-file#overriding-pickles-serialization-mechanism-for-importable-constructs
    """  # noqa: E501
    if not trust_remote_code:
        return

    try:
        import transformers_modules
    except ImportError:
        logger.debug('Could not import transformers_modules used for remote'
                     ' code. If remote code is not needed remove'
                     ' `--trust-remote-code`.')
        return

    try:
        import cloudpickle
        cloudpickle.register_pickle_by_value(transformers_modules)

        # ray vendors its own version of cloudpickle
        try:
            import ray
        except ImportError:
            return

        ray.cloudpickle.register_pickle_by_value(transformers_modules)

        # multiprocessing uses pickle to serialize arguments when using spawn
        # Here we get pickle to use cloudpickle to serialize ModelConfig objects
        # that contain instances of the custom config class to avoid
        # serialization problems if the generated module (and model) has a `.`
        # in its name
        import multiprocessing
        import pickle

        from lmdeploy.pytorch.config import ModelConfig

        def _reduce_modelconfig(mc: ModelConfig):
            return (pickle.loads, (cloudpickle.dumps(mc), ))

        multiprocessing.reducer.register(ModelConfig, _reduce_modelconfig)

    except Exception as e:
        logger.warning(
            'Unable to register remote classes used by'
            ' trust_remote_code with by-value serialization. This may'
            ' lead to a later error. If remote code is not needed'
            ' remove `--trust-remote-code`',
            exc_info=e)

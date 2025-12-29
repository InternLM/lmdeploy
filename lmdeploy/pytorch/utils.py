# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import asyncio
import inspect
from contextlib import contextmanager
from inspect import Parameter, Signature
from typing import Dict, Generic, Optional, Sequence, TypeVar

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


def singleton(cls):
    """Singleton decorator."""
    import multiprocessing as mp

    from lmdeploy.utils import get_logger
    logger = get_logger('lmdeploy')
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            pid = mp.current_process().pid
            logger.debug(f'pid:{pid} - Creating instance of singleton class {cls.__name__}')
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


T = TypeVar('T')


class CtxMgrBase(Generic[T]):
    """Context manager base class."""

    def __init__(self, default: Optional[T] = None):
        self._context = default

    def current_context(self) -> Optional[T]:
        """Get current context."""
        return self._context

    def set_context(self, context: Optional[T]):
        """Set current context."""
        self._context = context

    @contextmanager
    def context(self, context: T):
        """Context manager."""
        origin_context = self.current_context()
        self.set_context(context)
        try:
            yield self
        finally:
            self.set_context(origin_context)


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


def monkey_patch_hf_modules_cache():
    """Monkey patch HF_MODULES_CACHE to a temporary directory per process. This
    is necessary to avoid conflicts when multiple processes try to read/write
    to the same HF_MODULES_CACHE directory, especially in multi-GPU setups.

    modified from: https://github.com/InternLM/xtuner/blob/main/xtuner/v1/utils/misc.py
    """
    import os

    import transformers
    from huggingface_hub import constants

    # When using `remote_code` in HF components like tokenizer or config
    # (e.g., `AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)`),
    # the hf_model_path is copied to HF_MODULES_CACHE.
    # On multi-GPU machines (e.g., 8 GPUs), simultaneous read/write operations
    # by multiple processes on this shared directory can cause conflicts.
    # Therefore, we set HF_MODULES_CACHE to a temporary directory per process.

    HF_PATCH_MODULES_CACHE_PREFIX = 'modules_pid_'
    modules_cache = os.path.join(constants.HF_HOME, f'{HF_PATCH_MODULES_CACHE_PREFIX}{os.getpid()}')
    os.environ['HF_MODULES_CACHE'] = modules_cache

    transformers.utils.hub.HF_MODULES_CACHE = modules_cache

    # During import, Python creates a new name HF_MODULES_CACHE in the namespace
    # of the dynamic_module_utils module, binding it to the object referenced by
    # transformers.utils.HF_MODULES_CACHE at that moment.
    # Hence, we also need to set transformers.dynamic_module_utils.HF_MODULES_CACHE
    # to the new modules_cache.

    transformers.dynamic_module_utils.HF_MODULES_CACHE = modules_cache
    transformers.utils.HF_MODULES_CACHE = modules_cache

    logger.info(f'Set HF_MODULES_CACHE to {modules_cache} for current process {os.getpid()}')


async def wait_for_async_tasks(tasks: Sequence[asyncio.Task],
                               cancel_pending: bool = True,
                               ignore_cancellederror: bool = True):
    """Wait for async tasks."""
    if len(tasks) == 0:
        return [], []

    for task in tasks:
        if not isinstance(task, asyncio.Task):
            raise ValueError('All inputs must be asyncio.Task instances.')

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

        if cancel_pending:
            # cancel all pending tasks
            for task in pending:
                task.cancel()

        # raise exception if any
        for task in done:
            if task.cancelled():
                continue
            if exc := task.exception():
                if isinstance(exc, asyncio.CancelledError) and ignore_cancellederror:
                    logger.debug(f'Task <{task.get_name()}> cancelled.')
                    continue
                raise exc from None
    except asyncio.CancelledError:
        for task in tasks:
            if not task.done():
                task.cancel()
        raise

    return done, pending


async def cancel_async_tasks(tasks: Sequence[asyncio.Task]):
    """Cancel async tasks."""
    if isinstance(tasks, asyncio.Task):
        tasks = [tasks]

    tasks = list(task for task in tasks if not task.done())
    for task in tasks:
        task.cancel()
    return await asyncio.gather(*tasks, return_exceptions=True)

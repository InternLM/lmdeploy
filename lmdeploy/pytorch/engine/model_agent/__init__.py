# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, MiscConfig, ModelConfig, SpecDecodeConfig
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.distributed import DistContext, get_dist_manager

from .agent import BaseModelAgent, BatchedOutputs  # noqa: F401


def build_model_agent(
    model_path: str,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    backend_config: BackendConfig,
    misc_config: MiscConfig,
    dist_ctx: DistContext = None,
    device_ctx: DeviceContext = None,
    adapters: Dict[str, str] = None,
    specdecode_config: SpecDecodeConfig = None,
):
    """Create model agent.

    Args:
        model_path (str): the path of the input model
        cache_config (CacheConfig): config of kv cache
        backend_config (BackendConfig): config of backend devices
        trust_remote_code (bool): To use the remote modeling code or not
        adapters (Dict): lora adapters
        tp (int): the number of devices to be used in tensor parallelism
        dtype (str): the data type of model weights and activations
        custom_module_map (str): customized nn module map
    """

    if device_ctx is None:
        device_mgr = get_device_manager()
        device_ctx = device_mgr.current_context()
    if dist_ctx is None:
        dist_mgr = get_dist_manager()
        dist_ctx = dist_mgr.current_context()

    model_agent = BaseModelAgent(
        model_path,
        model_config=model_config,
        cache_config=cache_config,
        backend_config=backend_config,
        misc_config=misc_config,
        adapters=adapters,
        dist_ctx=dist_ctx,
        device_ctx=device_ctx,
        specdecode_config=specdecode_config,
    )
    return model_agent

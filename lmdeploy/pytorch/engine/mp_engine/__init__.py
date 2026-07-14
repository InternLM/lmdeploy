# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.messages import PytorchEngineConfig


def build_mp_engine(backend: str, model_path: str, engine_config: PytorchEngineConfig = None,
                    trust_remote_code: bool = False, **kwargs):
    """Build mp engine."""
    if backend == 'mp':
        from .zmq_engine import ZMQMPEngine
        return ZMQMPEngine(model_path, engine_config=engine_config, trust_remote_code=trust_remote_code, **kwargs)
    elif backend == 'ray':
        from .ray_engine import RayMPEngine
        return RayMPEngine(model_path, engine_config=engine_config, trust_remote_code=trust_remote_code, **kwargs)
    else:
        raise ValueError(f'Unsupported backend: {backend}')

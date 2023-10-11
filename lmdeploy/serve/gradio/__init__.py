# Copyright (c) OpenMMLab. All rights reserved.
from .api_server_decoupled import run_api_server
from .triton_server_decoupled import run_triton_server
from .turbomind_coupled import run_local

__all__ = ['run_api_server', 'run_triton_server', 'run_local']

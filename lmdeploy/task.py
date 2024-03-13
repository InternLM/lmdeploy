# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from lmdeploy.utils import get_hf_config_content

SUPPORTED_TASKS = {'llm': AsyncEngine, 'vl-llm': VLAsyncEngine}


def check_vl_llm(config: dict):
    arch = config['architectures'][0]
    if arch == 'LlavaLlamaForCausalLM':
        return True
    elif arch == 'QWenLMHeadModel' and 'visual' in config:
        return True
    return False


def get_task(model_path: str):
    config = get_hf_config_content(model_path)
    if check_vl_llm(config):
        return 'vl-llm', VLAsyncEngine

    # default task, pipeline_class
    return 'llm', AsyncEngine

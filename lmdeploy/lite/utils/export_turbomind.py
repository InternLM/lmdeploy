# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import shutil

from huggingface_hub import snapshot_download

from lmdeploy.turbomind.utils import get_hf_config_content


def export_turbomind_config(model_name: str,
                            model_path: str,
                            work_dir: str,
                            model_format: str = 'awq',
                            group_size: int = 128,
                            tp: int = 1):
    """Export hf lmdeploy model and config.json."""
    import lmdeploy
    from lmdeploy.model import MODELS
    from lmdeploy.turbomind.deploy.converter import get_model_format
    from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS
    from lmdeploy.turbomind.deploy.target_model.base import (
        OUTPUT_MODELS, TurbomindModelConfig)

    assert model_name in MODELS.module_dict.keys(), \
        f"'{model_name}' is not supported. " \
        f'The supported models are: {MODELS.module_dict.keys()}'

    if not os.path.exists(model_path):
        model_path = snapshot_download(model_path, local_files_only=True)

    lmdeploy_dir = os.path.split(lmdeploy.__file__)[0]
    hf_repo = os.path.join(lmdeploy_dir, 'turbomind', 'hf_repo')
    files = os.listdir(hf_repo)
    for file in files:
        src = os.path.join(hf_repo, file)
        dst = os.path.join(work_dir, file)
        shutil.copy(src, dst)

    cfg = TurbomindModelConfig.from_dict({}, allow_none=True)
    cfg.model_name = model_name
    cfg.tensor_para_size = tp
    cfg.rotary_embedding = cfg.size_per_head
    cfg.group_size = group_size
    cfg.weight_type = 'int4'
    output_format = 'w4'

    inferred_model_format = get_model_format(model_name, model_format)
    input_model = INPUT_MODELS.get(inferred_model_format)(
        model_path=model_path, tokenizer_path=work_dir, ckpt_path=work_dir)
    output_model = OUTPUT_MODELS.get(output_format)(input_model=input_model,
                                                    cfg=cfg,
                                                    to_file=False,
                                                    out_dir='')

    old_data = get_hf_config_content(model_path)
    config = output_model.cfg.__dict__
    config_file = os.path.join(work_dir, 'config.json')
    with open(config_file) as f:
        data = json.load(f)
    for k, v in old_data.items():
        if k in data:
            data[f'__{k}'] = v
        else:
            data[k] = v
    data['turbomind'] = config
    from lmdeploy.version import __version__
    data['lmdeploy_version'] = __version__
    with open(config_file, 'w') as f:
        f.write(json.dumps(data, indent=2) + '\n')

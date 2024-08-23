# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
import subprocess


def get_llama_gemm():
    """get the executable binary llama_gemm."""
    import os.path as osp

    import lmdeploy
    lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
    bin_path = osp.join(lmdeploy_dir, 'bin', 'llama_gemm')
    assert osp.exists(bin_path), f'{bin_path} not exists'
    return bin_path


def read_config(config_file: str):
    """read turbomind config from turbomind.

    Args:
        config_file (str): the path of turbomind config file in turbomind model
    """

    import yaml

    from lmdeploy.turbomind.deploy.target_model.base import \
        TurbomindModelConfig
    with open(config_file, 'r') as f:
        _cfg = yaml.safe_load(f)
    cfg = TurbomindModelConfig.from_dict(_cfg)
    return cfg.head_num, cfg.size_per_head, cfg.inter_size, \
        cfg.vocab_size, cfg.tensor_para_size


def main(head_num: int = 32,
         size_per_head: int = 128,
         vocab_size: int = 32000,
         inter_size: int = 11008,
         tensor_para_size: int = 1,
         max_batch_size: int = 64,
         model_path: str = None):
    if model_path is not None:
        from lmdeploy.turbomind.turbomind import get_model_source
        from lmdeploy.turbomind.utils import ModelSource

        model_source = get_model_source(model_path)
        if model_source == ModelSource.WORKSPACE:
            head_num, size_per_head, inter_size, vocab_size, \
                tensor_para_size = read_config(
                    osp.join(model_path,
                             'triton_models', 'weights', 'config.yaml'))
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path,
                                                trust_remote_code=True)
            head_num = config.num_attention_heads
            size_per_head = config.hidden_size // head_num
            inter_size = config.intermediate_size
            vocab_size = config.vocab_size
    for bsz in range(1, max_batch_size + 1):
        subprocess.call(
            f'{get_llama_gemm()} {bsz} 1 1 {head_num} {size_per_head}'
            f' {inter_size} {vocab_size} 1 {tensor_para_size}'
            f' {0 if bsz == 1 else 1}',
            shell=True)


if __name__ == '__main__':
    import fire

    fire.Fire(main)

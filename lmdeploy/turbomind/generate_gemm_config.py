# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
import subprocess

from lmdeploy.turbomind.deploy.target_model.base import TurbomindModelConfig


def get_llama_gemm():
    """get the executable binary llama_gemm."""
    import os.path as osp

    import lmdeploy
    lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
    bin_path = osp.join(lmdeploy_dir, 'bin', 'llama_gemm')
    assert osp.exists(bin_path), f'{bin_path} not exists'
    return bin_path


def read_config(ini_path: str):
    """read turbomind config from turbomind.

    Args:
        ini_path (str): the path of `config.ini` file in turbomind model
    """
    from configparser import ConfigParser
    with open(ini_path, 'r') as f:
        parser = ConfigParser()
        parser.read_file(f)
    section_name = 'llama'
    _cfg = parser._sections[section_name]
    cfg = TurbomindModelConfig.from_dict(_cfg)
    return cfg['head_num'], cfg['size_per_head'], cfg['inter_size'], cfg[
        'vocab_size'], cfg['tensor_para_size']


def get_config(model_path: str, tp: int):
    """get turbomind config from transformers model.

    Args:
        model_path (str): the path or repo name of the transformers model
        tp (int): the number of GPUs in tensor parallelism
    """
    from huggingface_hub import snapshot_download

    from lmdeploy.model import best_match_model

    model_name = best_match_model(model_path)
    if model_name is None:
        print(f'failed to get chat template name from the path {model_path}')
        exit(-1)
    if not osp.exists(model_path):
        print(f'can\'t find model from local_path {model_path}, '
              'try to download from huggingface')
        model_path = snapshot_download(model_path)
        print(f'load model from {model_path}')

    from lmdeploy.turbomind.deploy.converter import get_model_format
    from lmdeploy.turbomind.deploy.source_model.base import INPUT_MODELS
    from lmdeploy.turbomind.deploy.target_model.base import OUTPUT_MODELS
    inferred_model_format = get_model_format(model_name, 'hf')
    input_model = INPUT_MODELS.get(inferred_model_format)(
        model_path=model_path, tokenizer_path=model_path, ckpt_path=None)

    cfg = TurbomindModelConfig(model_name=model_name, tensor_para_size=tp)
    output_model = OUTPUT_MODELS.get('fp16')(input_model=input_model,
                                             cfg=cfg,
                                             to_file=False,
                                             out_dir='')
    cfg = output_model.get_config(cfg)
    return cfg.head_num, cfg.size_per_head, cfg.inter_size, cfg.vocab_size


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
                             'triton_models', 'weights', 'config.ini'))
        else:
            head_num, size_per_head, inter_size, vocab_size \
                = get_config(model_path, tensor_para_size)
    for bsz in range(1, max_batch_size + 1):
        subprocess.call(
            f'{get_llama_gemm()} {bsz} 1 1 {head_num} {size_per_head}'
            f' {inter_size} {vocab_size} 1 {tensor_para_size}'
            f' {0 if bsz == 1 else 1}',
            shell=True)


if __name__ == '__main__':
    import fire

    fire.Fire(main)

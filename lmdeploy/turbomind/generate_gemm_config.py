# Copyright (c) OpenMMLab. All rights reserved.

import subprocess


def get_llama_gemm():
    import os.path as osp

    import lmdeploy
    lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
    bin_path = osp.join(lmdeploy_dir, 'bin', 'llama_gemm')
    assert osp.exists(bin_path), f'{bin_path} not exists'
    return bin_path


def main(head_num: int = 32,
         size_per_head: int = 128,
         vocab_size: int = 32000,
         inter_size: int = 11008,
         tensor_para_size: int = 1,
         max_batch_size: int = 64):
    for bsz in range(1, max_batch_size + 1):
        subprocess.call(
            f'{get_llama_gemm()} {bsz} 1 1 {head_num} {size_per_head}'
            f' {inter_size} {vocab_size} 1 {tensor_para_size}'
            f' {0 if bsz == 1 else 1}',
            shell=True)


if __name__ == '__main__':
    import fire

    fire.Fire(main)

# Copyright (c) OpenMMLab. All rights reserved.

import subprocess
import fire


def main(head_num: int = 32,
         size_per_head: int = 128,
         vocab_size: int = 32000,
         inter_size: int = 11008,
         tensor_para_size: int = 1,
         max_batch_size: int = 64):
    for bsz in range(1, max_batch_size + 1):
        subprocess.call(
            f'bin/llama_gemm {bsz} 1 1 {head_num} {size_per_head} {inter_size} {vocab_size} 1 {tensor_para_size} {0 if bsz == 1 else 1}',
            shell=True)


if __name__ == '__main__':
    fire.Fire(main)

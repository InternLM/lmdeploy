# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil

import fire
import torch
from tqdm import tqdm

from lmdeploy.vl.model.builder import vl_model_with_tokenizer


def main(src_path: str, dst_path: str, task: str):
    """merge internlm-xcomposer2d5-7b LoRA model weights.

    Args:
        src_path (str): the source model path of internlm-xcomposer2d5-7b
        dst_path (str): the target model path of merged model
        task (str): the task of source model, should choose from
            ['web', 'write']
    """
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)

    to_merged = dict(web=['lora_web'], write=['lora_sft', 'lora_dpo'])
    keys = to_merged[task]

    # load model
    model, _, tokenizer = vl_model_with_tokenizer(src_path)

    # merge lora weight to base model
    @torch.inference_mode
    def _merge(module: torch.nn.Module, lora_weights):
        # merge lora weight first to reduce precision loss
        mw = None
        for wa, wb in lora_weights:
            if mw is None:
                mw = (wb.float() @ wa.float())
            else:
                mw += (wb.float() @ wa.float())
        ow = module.weight
        mw += ow.float()
        module.weight.data = mw.half()

    def _extract_lora(module: torch.nn.Module, keys: str):
        lora_weights = []
        for key in keys:
            lora_a_key = f'{key}_A'
            lora_b_key = f'{key}_B'
            wa = getattr(module, lora_a_key).weight
            wb = getattr(module, lora_b_key).weight
            lora_weights.append((wa, wb))
        return lora_weights

    for _, module in tqdm(model.named_modules()):
        if type(module).__name__ == 'PLoRA':
            lora_weights = _extract_lora(module, keys)
            _merge(module, lora_weights)

    # save model
    model.save_pretrained(dst_path)
    tokenizer.save_pretrained(dst_path)


if __name__ == '__main__':
    fire.Fire(main)

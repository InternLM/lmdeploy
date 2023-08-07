# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List, Tuple, Optional, Union

import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from lmdeploy.lite.utils import collect_target_modules
from lmdeploy.lite.utils import get_calib_loaders
from lmdeploy.lite.quantization.calibration import Calibration

from mmengine import Config
from accelerate import init_empty_weights,load_checkpoint_in_model, infer_auto_device_map
def main(model: str,
         layer_type: Union[str, type]='LlamaDecoderLayer',
         norm_type: Union[str, type]='LlamaRMSNorm',
         smooth: bool = True,
         w_bits: int = 4,
         w_sym: bool = False,
         w_granularity: str = 'per_group',
         w_group_size: int = 128,
         calib_dataset: str = 'c4',
         calib_samples: int = 128,
         calib_seqlen: int = 2048,
         work_dir='./work_dir',
         device: str = 'cuda'):
    

    assert calib_dataset in ['c4', 'ptb', 'wikitext2', 'pileval'], \
        'Currently, only support `c4`, `ptb`, `wikitext2`, or `pileval`.'

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    hf_config = AutoConfig.from_pretrained(model)
    checkpoint = hf_config._name_or_path

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    
    model.config.use_cache = False
    decoder_layers = collect_target_modules(model, layer_type)
    device_map = infer_auto_device_map(model,no_split_module_classes=[layer_type])
    for name in device_map.keys():
        if name in decoder_layers:
            device_map[name] = 'cpu'
        else:
            device_map[name] = 0
    load_checkpoint_in_model(model, checkpoint,device_map)

    print('Loading calibrate dataset ...')
    calib_loader, _ = get_calib_loaders(calib_dataset,
                                        tokenizer,
                                        nsamples=calib_samples,
                                        seqlen=calib_seqlen)

    
    calibrator = Calibration(
                    model,
                    layer_type=layer_type,
                    norm_type=norm_type,
                    smooth=smooth,
                    w_qconfig=w_qconfig,
                    work_dir=work_dir,
                    device=device) 

    with calibrator:
        all_data = torch.cat([data if isinstance(data, torch.Tensor) else data[0] for data in calib_loader]).to(device)
        calibrator.step(all_data)
    
    

if __name__ == '__main__':

    fire.Fire(main)

# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict

import fire
import torch

from transformers import AutoModelForCausalLM
from lmdeploy.lite.utils import collect_target_modules
from pathlib import Path
import numpy as np


    

def export_turbomind_kv(key_stats, value_stats, bits, symmetry, mp, out_dir):

    out_dir = Path(out_dir)
    if symmetry:
            
        keys_absmax = key_stats['absmax']
        values_absmax = value_stats['absmax']
        for layer_idx, name in enumerate(keys_absmax.keys()):
            k_absmax = keys_absmax[name]
            v_absmax = values_absmax[name]

            heads, dims = k_absmax.shape
            assert heads % mp == 0

            mp_k_absmax = torch.chunk(k_absmax, mp)
            mp_v_absmax = torch.chunk(v_absmax, mp)
            for i in range(mp):
                # quant: q = f / scale
                # dequant: f = q * scale
                k_s = max(mp_k_absmax[i]) / (2**(bits - 1) - 1)
                v_s = max(mp_v_absmax[i]) / (2**(bits - 1) - 1)
                
                kv_qparams = np.array([k_s, v_s], dtype=np.float32)
                save_path = out_dir / f'layers.{layer_idx}.past_kv_scale.{i}.weight'
                kv_qparams.tofile(save_path)
                print(f'Layer {layer_idx} MP {i} KV scales done.')
    else:
        keys_min = key_stats['min']
        values_min = value_stats['min']

        keys_max = key_stats['max']
        values_max = value_stats['max']
        for layer_idx, name in enumerate(keys_min.keys()):
            k_max = keys_max[name]
            v_max = values_max[name]

            k_min = keys_min[name]
            v_min = values_min[name]

            heads, dims = keys_min.shape
            assert heads % mp == 0

            mp_k_min = torch.chunk(k_min, mp)
            mp_v_min = torch.chunk(v_min, mp)

            mp_k_max = torch.chunk(k_max, mp)
            mp_v_max = torch.chunk(v_max, mp)
            for i in range(mp):
                # quant: q = (f - zp) / scale
                # dequant: f = q * scale + zp
                k_min = min(mp_k_min[i]) 
                v_min = min(mp_v_min[i]) 

                k_max = max(mp_k_max[i])
                v_max = max(mp_v_max[i])

                k_scale = (k_max - k_min) / (2**bits - 1)
                v_scale = (v_max - v_min) / (2**bits - 1)
                
                kv_qparams = np.array([k_scale, k_min, v_scale, v_min],
                                  dtype=np.float32)
                save_path = out_dir / f'layers.{layer_idx}.past_kv_scale.{i}.weight'
                kv_qparams.tofile(save_path)
                print(f'Layer {layer_idx} MP {i} KV scales&zeros done.')

def main(model: str,
         qparams_dir: str,
         turbomind_dir: str,
         kv_bits: int = 8,
         kv_sym: bool = False,
         model_parallel: int = 1):
    
    qparams_dir =  Path(qparams_dir)
    
    key_stats_url = qparams_dir / 'kv_stats.pth'
    value_stats_url = qparams_dir / 'value_stats.pth'
    assert key_stats_url.exists()
    assert value_stats_url.exists()

    key_stats = torch.load(key_stats_url, map_location='cpu')
    value_stats = torch.load(value_stats_url, map_location='cpu')
    
    # TODO support per channel
    export_turbomind_kv(key_stats, value_stats, kv_bits, kv_sym, model_parallel, turbomind_dir)
        
    # TODO export turbomind weights
    

if __name__ == '__main__':

    fire.Fire(main)

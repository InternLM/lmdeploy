# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional, Callable
import torch
from torch import nn

from lmdeploy.lite.utils.global_avail import GlobalAvailMixin

import numpy as np
class KVCacheObserver(GlobalAvailMixin):
    
    def __init__(self, num_head, head_dim):
        
        self.num_head = num_head
        self.head_dim = head_dim
        self.max_val = torch.full((num_head, head_dim), -torch.inf, dtype=torch.float16)
        self.min_val = torch.full((num_head, head_dim), torch.inf, dtype=torch.float16)
        self.absmax_val = torch.full((num_head, head_dim), 0, dtype=torch.float16)
    
    @torch.inference_mode
    def observe(self, x: torch.Tensor):
        
        assert len(x.shape) == 4
        x = x.transpose(1,2)
        assert x.size(2) == self.num_head
        assert x.size(3) == self.head_dim
        
        cur_max = x.flatten(0,1).max(0)[0].cpu()
        cur_min = x.flatten(0,1).min(0)[0].cpu()
        cur_absmax = x.flatten(0,1).abs().max(0)[0].cpu()

        self.max_val = torch.maximum(self.max_val, cur_max)
        self.min_val = torch.minimum(self.min_val, cur_min)
        self.absmax_val = torch.maximum(self.absmax_val, cur_absmax)

    


class ActivationObserver(GlobalAvailMixin):
    
    def __init__(self, dim):
        
        self.dim = dim

        self.max_val = torch.full((self.dim,), -torch.inf, dtype=torch.float16)
        self.min_val = torch.full((self.dim,), torch.inf, dtype=torch.float16)
        self.absmax_val = torch.full((self.dim,), 0, dtype=torch.float16)
    
    @torch.inference_mode
    def observe(self, x: torch.Tensor):
        
        assert len(x.shape) == 3
        assert x.size(2) == self.dim
        
        
        
        cur_max = x.flatten(0,1).max(0)[0].cpu()
        cur_min = x.flatten(0,1).min(0)[0].cpu()
        cur_absmax = x.flatten(0,1).abs().max(0)[0].cpu()

        self.max_val = torch.maximum(self.max_val, cur_max)
        self.min_val = torch.minimum(self.min_val, cur_min)
        self.absmax_val = torch.maximum(self.absmax_val, cur_absmax)


    

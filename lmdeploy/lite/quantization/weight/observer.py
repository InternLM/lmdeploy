# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional, Callable
import torch
from torch import nn

from lmdeploy.lite.utils.global_avail import GlobalAvailMixin

import numpy as np

class WeightObserver(GlobalAvailMixin):
    
    def __init__(self, channels, observe_dim = 1):
        
        self.channels = channels
        self.observe_dim = observe_dim

        self.max_val = torch.full((self.channels,), -torch.inf, dtype=torch.float16)
        self.min_val = torch.full((self.channels,), torch.inf, dtype=torch.float16)
        self.absmax_val = torch.full((self.channels,), 0, dtype=torch.float16)
    
    @torch.inference_mode
    def observe(self, x: torch.Tensor):
        
        assert len(x.shape) == 2
        assert x.size(self.observe_dim) == self.channels
        
        
        cur_max = x.max(self.observe_dim)[0].cpu()
        cur_min = x.min(self.observe_dim)[0].cpu()
        cur_absmax = x.abs().max(self.observe_dim)[0].cpu()

        self.max_val = torch.maximum(self.max_val, cur_max)
        self.min_val = torch.maximum(self.min_val, cur_min)
        self.absmax_val = torch.maximum(self.absmax_val, cur_absmax)


    @classmethod
    def merge_stats(self, observers):

        max_val = None
        min_val = None
        absmax_val = None

        for i, obs in enumerate(observers):
            if i==0:
                max_val = obs.max_val
                min_val = obs.min_val
                absmax_val = obs.absmax_val
            else:
                max_val = torch.maximum(obs.max_val, max_val)
                min_val = torch.minimum (obs.min_val, min_val)
                absmax_val = torch.maximum(obs.absmax_val, absmax_val)


        for obs in observers:
            obs.max_val = max_val
            obs.min_val = min_val
            obs.absmax_val = absmax_val

    
        
        

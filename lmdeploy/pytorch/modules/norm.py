import torch
from torch import nn
import triton
from lmdeploy.pytorch.kernels import rms_norm
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class LMDNorm(nn.Module):

    supported = (LlamaRMSNorm,)
    def __init__(self, weight, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.variance_epsilon)
    
    
    @classmethod
    def from_hf(cls, hf_mod):
        if isinstance(hf_mod, LlamaRMSNorm):
            return cls(hf_mod.weight, hf_mod.variance_epsilon)

    


def norm_pass(model):
    for name, mod in model.named_modules():
        if not isinstance(mod, LMDNorm.supported):
            continue

        norm = LMDNorm.from_hf(mod)

        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name[len(parent_name) + 1:]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ''
            parent = model
            child_name = name

        print(f"Replacing {name} with LMDNorm; parent: {parent_name}, child's name: {child_name}")

        setattr(parent, child_name, norm)
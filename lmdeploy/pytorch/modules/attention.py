import torch
from torch import nn
from lmdeploy.pytorch.kernels import rotate_half#,attention

from transformers.models.llama.modeling_llama import LlamaAttention

class LMDAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    supported = (LlamaAttention,)
    def __init__(
        self,
        hidden_size,
        num_heads,
        qkv_proj,
        o_proj,
        with_rotary_emb=True,
        with_multi_query=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                             f" and `num_heads`: {num_heads}).")
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj

    def forward(self, hidden_states, past_key_value=None, attention_mask=None, position_ids=None, output_attentions=False, use_cache=False):
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.view(bsz, q_len, 3, self.num_heads, self.head_dim)

        # This updates the query and key states in-place, saving VRAM.
        rotate_half(qkv_states[:, :, :2], position_ids)

        query_states, key_states, value_states = torch.split(qkv_states, 1, dim=2)
        del qkv_states
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim)
        is_causal = past_key_value is None

        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)

        if use_cache:
            # Since qkv_proj is fused, query_states etc will hold a reference to the original qkv_states tensor
            # which can cause excessive memory usage by the cache. `contiguous` is a convenient way to workaround this.
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
            query_states = query_states.contiguous()

        past_key_value = (key_states, value_states) if use_cache else None

        import pdb;pdb.set_trace()
        from xformers.ops import memory_efficient_attention, LowerTriangularMask
        from flash_attn.flash_attn_interface import flash_attn_func
        sm_scale = 1 / (query_states.shape[-1] ** 0.5)
        attn_output = flash_attn_func(query_states, key_states,value_states, 0.0, sm_scale,is_causal)
        # attn_output = memory_efficient_attention(query_states, key_states, value_states,LowerTriangularMask() if is_causal else None )
        # attn_output = attention(query_states, key_states, value_states, is_causal)
        # print(attn_output.sum())
        # import pdb;pdb.set_trace()
        # attn_output = memory_efficient_attention(query_states, key_states, value_states,LowerTriangularMask() if is_causal else None )
        # del query_states, key_states, value_states

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
    
    @classmethod
    def from_hf(cls, hf_mod):

        
        if isinstance(hf_mod, LlamaAttention):
            hidden_size = hf_mod.hidden_size
            num_heads = hf_mod.num_heads
            head_dim = hf_mod.head_dim
            
            q_proj = hf_mod.q_proj
            k_proj = hf_mod.k_proj
            v_proj = hf_mod.v_proj
            o_proj = hf_mod.o_proj

            dtype = q_proj.weight.dtype
            with_bias = True if q_proj.bias else False

            qkv_proj = nn.Linear( num_heads * head_dim * 3,hidden_size, bias=with_bias)
            qkv_proj_weight = nn.Parameter(torch.cat(
                [q_proj.weight, k_proj.weight, v_proj.weight],  dim=0),requires_grad=False)
            print(q_proj.weight.shape, qkv_proj_weight.shape)
            
            if with_bias:
                qkv_proj_bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias])
            else:
                qkv_proj_bias = None
            qkv_proj.weight = qkv_proj_weight
            qkv_proj.bias = qkv_proj_bias

            return cls(hidden_size, num_heads, qkv_proj, o_proj)
        
def attention_pass(model):
    """
    Replace all LlamaAttention modules with QuantLlamaAttention modules, fusing the q, k, v projections.
    """

    for name, mod in model.named_modules():
        if not isinstance(mod, LMDAttention.supported):
            continue

        attn = LMDAttention.from_hf(mod)

        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name[len(parent_name) + 1:]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ''
            parent = model
            child_name = name

        print(f"Replacing {name} with LMDAttention; parent: {parent_name}, child's name: {child_name}")

        setattr(parent, child_name, attn)

# from transformers import AutoModel

# model = AutoModel.from_pretrained('/nvme/share_data/llama-7b/')
# # from .utils import attention_pass

# attention_pass(model)
# import pdb;pdb.set_trace()
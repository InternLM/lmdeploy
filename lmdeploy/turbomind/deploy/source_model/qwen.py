# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import re

import torch

from ..config import RopeParam
from ..loader import create_loader
from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class QwenReader(LlamaReader):
    """QwenReader."""

    attn_layer_patten = r'transformer\.h\.([0-9]+).'
    tok_embeddings_key = 'transformer.wte.weight'
    norm_weight_key = 'transformer.ln_f.weight'
    output_weight_key = 'lm_head.weight'

    attn_pattern = r'attn'
    ffn_pattern = r'mlp'

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o kind for layer i."""
        q, k, v, o = (None, ) * 4
        qkv = self.params[f'transformer.h.{i}.attn.c_attn.{kind}']
        qkv = self.transform(qkv, kind)
        if qkv is not None:
            q, k, v = torch.split(qkv, qkv.size(0) // 3, dim=0)
        o = self.params.get(f'transformer.h.{i}.attn.c_proj.{kind}')
        o = self.transform(o, kind)
        if o is None:
            o = torch.zeros_like(q)
        return q, k, v, o

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[f'transformer.h.{i}.ln_1.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        result = []
        for key in ['w2', 'c_proj', 'w1']:
            tensor = self.params[f'transformer.h.{i}.mlp.{key}.{kind}']
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[f'transformer.h.{i}.ln_2.weight']


@INPUT_MODELS.register_module(name='qwen')
class QwenModel(LlamaModel):
    """Qwen model in hf format."""

    Reader = QwenReader

    def model_info(self):
        """Read model info."""
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            config = json.load(f)
            hidden_units = config['hidden_size']
            num_layer = config['num_hidden_layers']
            norm_eps = config['layer_norm_epsilon']
            kv_channels = config['kv_channels']
            rope_theta = float(config.get('rotary_emb_base', 10000.0))
            if 'num_key_value_heads' in config:
                kv_head_num = config['num_key_value_heads']
            else:
                kv_head_num = config['num_attention_heads']
            attn_head_num = config['num_attention_heads']
            seq_length = config['seq_length']
            use_dynamic_ntk = int(config['use_dynamic_ntk'])
            use_logn_attn = int(config['use_logn_attn'])
            vocab_size = config['vocab_size']
            inter_size = config['intermediate_size']
            scaling_type = 'dynamic' if use_dynamic_ntk else 'default'
            # need setting rope_scaling_factor in TurbomindEngineConfig if scaling_type is dynamic
            rope_param = RopeParam(type=scaling_type,
                                   base=rope_theta,
                                   dim=kv_channels,
                                   max_position_embeddings=seq_length,
                                   factor=0)

        return dict(size_per_head=kv_channels,
                    num_layer=num_layer,
                    norm_eps=norm_eps,
                    hidden_units=hidden_units,
                    head_num=attn_head_num,
                    kv_head_num=kv_head_num,
                    vocab_size=vocab_size,
                    inter_size=inter_size,
                    attn_bias=1,
                    rope_param=rope_param,
                    max_position_embeddings=seq_length,
                    use_dynamic_ntk=int(use_dynamic_ntk),
                    use_logn_attn=use_logn_attn)


@INPUT_MODELS.register_module(name='qwen2')
class Qwen2Model(LlamaModel):
    """Qwen model in hf format.

    The weight of qwen2 model is similar to Llama, except its attention bias doesn't include o_proj bias.
    """

    Reader = LlamaReader

    def model_info(self):
        cfg = super().model_info()
        cfg['attn_bias'] = 1
        return cfg


class Qwen2MoeReader(LlamaReader):

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        if not kind:
            return self.filter(r'experts', i)
        result = []
        for key in ['gate', 'down', 'up']:
            name = f'{self.attn_layer_prefix}.{i}.mlp.experts.{e}.{key}_proj.{kind}'
            tensor = self.params.get(name)
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def moe_ffn_gate(self, i, kind):
        return self.transform(self.params.get(f'{self.attn_layer_prefix}.{i}.mlp.gate.{kind}'), kind)

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        if not kind:
            return self.filter(r'shared_expert\.', i)
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params[f'{self.attn_layer_prefix}.{i}.mlp.shared_expert.{key}_proj.{kind}']
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def ffn(self, i: int, kind: str):
        if not kind:
            return self.filter(r'shared_expert\.', i)
        return self._ffn(i, kind)

    def moe_ffn_shared_gate(self, i):
        return self.params.get(f'{self.attn_layer_prefix}.{i}.mlp.shared_expert_gate.weight')


@INPUT_MODELS.register_module(name='qwen2-moe')
class Qwen2MoeModel(LlamaModel):

    Reader = Qwen2MoeReader

    def model_info(self):
        cfg = self.model_config
        info = super().model_info()
        info['expert_num'] = cfg['num_experts']
        info['expert_inter_size'] = cfg['moe_intermediate_size']
        info['experts_per_token'] = cfg['num_experts_per_tok']
        info['inter_size'] = cfg['shared_expert_intermediate_size']
        info['moe_shared_gate'] = True
        info['norm_topk_prob'] = cfg['norm_topk_prob']
        info['attn_bias'] = cfg.get('qkv_bias', 1)
        return info


class Qwen3Reader(LlamaReader):

    def qk_norm(self, i: int):
        result = []
        for x in ['q', 'k']:
            name = f'{self.attn_layer_prefix}.{i}.self_attn.{x}_norm.weight'
            result.append(self.transform(self.params.get(name), 'weight'))
        return (*result, )


@INPUT_MODELS.register_module(name='qwen3')
class Qwen3Model(LlamaModel):
    Reader = Qwen3Reader

    def model_info(self):
        cfg = self.model_config
        info = super().model_info()
        info.update(qk_norm=True, attn_bias=cfg.get('attention_bias', 0))
        return info


class Qwen3MoeReader(Qwen2MoeReader):

    def qk_norm(self, i: int):
        result = []
        for x in ['q', 'k']:
            name = f'{self.attn_layer_prefix}.{i}.self_attn.{x}_norm.weight'
            result.append(self.transform(self.params.get(name), 'weight'))
        return (*result, )


@INPUT_MODELS.register_module(name='qwen3-moe')
class Qwen3MoeModel(LlamaModel):
    Reader = Qwen3MoeReader

    def model_info(self):
        cfg = self.model_config
        info = super().model_info()
        info.update(
            qk_norm=True,
            expert_num=cfg.get('num_experts', 128),
            experts_per_token=cfg.get('num_experts_per_tok', 8),
            expert_inter_size=cfg.get('moe_intermediate_size', 768),
            attn_bias=cfg.get('attention_bias', 0),
            inter_size=0,  # no shared expert
            norm_topk_prob=cfg.get('norm_topk_prob', False))
        return info


class Qwen3_5ReaderMixin:
    """Mixin providing linear attention weight reading for Qwen3.5 models.

    Qwen3.5 uses a zero-centered RMSNorm: ``output = norm(x) * (1 + weight)``
    where weight is initialized to zeros.  TurboMind's RMSNorm kernel computes
    ``norm(x) * weight`` (standard LLaMA style), so we add 1 to every
    RMSNorm weight during export.  The GDN-internal norm
    (``Qwen3_5MoeRMSNormGated``) uses standard weight and is NOT affected.
    """

    attn_layer_pattern = r'(?:model\.language_model\.|model\.)layers\.([0-9]+)\.'

    _LINEAR_ATTN_KEYS = ['conv1d', 'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj', 'A_log', 'dt_bias']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any(k.startswith('model.language_model.') for k in self.params.keys()):
            self.attn_layer_prefix = 'model.language_model.layers'
            self.tok_embeddings_key = 'model.language_model.embed_tokens.weight'
            self.norm_weight_key = 'model.language_model.norm.weight'

    # ---- zero-centered RMSNorm: add 1 to weights during export ----

    def attn_norm(self, i: int):
        w = super().attn_norm(i)
        if w is not None:
            w = w.float() + 1.0
        return w

    def ffn_norm(self, i: int):
        w = super().ffn_norm(i)
        if w is not None:
            w = w.float() + 1.0
        return w

    def norm_weight(self):
        w = super().norm_weight()
        if w is not None:
            w = w.float() + 1.0
        return w

    def qk_norm(self, i: int):
        result = super().qk_norm(i)
        return tuple(w.float() + 1.0 if w is not None else w for w in result)

    # ---- handle mixed QKV(fp16) + O(AWQ) attention layers -------

    def _attn(self, i: int, kind: str):
        """Override to handle mixed QKV(fp16) + O(AWQ) attention layers.

        Some AWQ-quantized Qwen3.5 models keep QKV in fp16 while quantizing only the O projection.  TurboMind requires
        uniform weight types per layer, so we dequantize O to fp16 at export time.
        """
        prefix = f'{self.attn_layer_prefix}.{i}.self_attn'
        q_is_fp16 = f'{prefix}.q_proj.weight' in self.params
        o_is_awq = f'{prefix}.o_proj.qweight' in self.params

        if not (q_is_fp16 and o_is_awq):
            # Not a mixed-format layer, use standard behaviour.
            return super()._attn(i, kind)

        # Mixed format detected: QKV are fp16 but O is AWQ.
        if kind == 'weight':
            # Get fp16 QKV the normal way, then dequantize O.
            q, k, v, _ = super()._attn(i, kind)
            o = self._awq_dequant(f'{prefix}.o_proj')
            o = self.transform(o, kind)
            return (q, k, v, o)

        # For any quant kind (qweight/scales/qzeros), return all None
        # so that the AWQ handler skips this layer entirely — the O
        # weight is already handled via dequantization above.
        return (None, None, None, None)

    def _awq_dequant(self, prefix: str):
        """Dequantize an AWQ-quantized linear layer to fp16.

        AWQ stores weights in transposed form relative to PyTorch's
        convention ([in, out] vs [out, in]), so we transpose here to
        match the fp16 ``.weight`` layout that downstream export
        expects.
        """
        from lmdeploy.pytorch.backends.default.awq_modules import dequantize_gemm
        qweight = self.params[f'{prefix}.qweight']
        scales = self.params[f'{prefix}.scales']
        qzeros = self.params[f'{prefix}.qzeros']
        group_size = qweight.shape[0] // scales.shape[0]
        w = dequantize_gemm(qweight, qzeros, scales, 4, group_size)
        return w.t()  # [in, out] → [out, in] (PyTorch convention)

    def linear_attn(self, i: int, kind: str):
        if not kind:
            return self.filter(r'linear_attn\.', i)
        # Always return a fixed-length tuple with None placeholders to
        # preserve positional alignment with the name list in module.py.
        result = []
        for key in self._LINEAR_ATTN_KEYS:
            prefix = f'{self.attn_layer_prefix}.{i}.linear_attn.{key}'
            tensor = self.params.get(f'{prefix}.{kind}')
            # A_log and dt_bias are bare nn.Parameter (no .weight suffix)
            if tensor is None:
                tensor = self.params.get(prefix)
            # If requesting weight but only AWQ qweight exists,
            # dequantize on the fly so LinearAttn gets fp16 tensors.
            if tensor is None and kind == 'weight':
                if f'{prefix}.qweight' in self.params:
                    tensor = self._awq_dequant(prefix)
            if tensor is not None:
                tensor = self.transform(tensor, kind)
            result.append(tensor)  # keep None to preserve alignment
        if all(t is None for t in result):
            return tuple()
        return tuple(result)

    def linear_norm(self, i: int, kind: str = 'weight'):
        tensor = self.params.get(f'{self.attn_layer_prefix}.{i}.linear_attn.norm.{kind}')
        if tensor is not None:
            return self.transform(tensor, kind)
        return None


class Qwen3_5Reader(Qwen3_5ReaderMixin, Qwen3Reader):
    pass


@INPUT_MODELS.register_module(name='qwen3_5')
class Qwen3_5Model(Qwen3Model):
    Reader = Qwen3_5Reader

    def model_info(self):
        if 'text_config' in self.model_config:
            self.model_config = self.model_config['text_config']
        cfg = self.model_config
        info = super().model_info()
        # MoE parameters (same as Qwen2MoeModel / Qwen3MoeModel)
        info['expert_num'] = cfg.get('num_experts', 0)
        info['expert_inter_size'] = cfg.get('moe_intermediate_size', 0)
        info['experts_per_token'] = cfg.get('num_experts_per_tok', 0)
        # For MoE models, inter_size is the shared expert intermediate size;
        # for dense models, keep the value from super() (intermediate_size).
        shared_expert_size = cfg.get('shared_expert_intermediate_size')
        if shared_expert_size is not None:
            info['inter_size'] = shared_expert_size
        info['moe_shared_gate'] = True
        # Qwen3.5 uses sigmoid MoE routing (not softmax)
        info['scoring_func'] = 'sigmoid'
        info['norm_topk_prob'] = True
        # Fix RoPE dim for partial_rotary_factor
        rope_params = cfg.get('rope_parameters', {})
        partial_rotary_factor = rope_params.get('partial_rotary_factor', cfg.get('partial_rotary_factor', 1.0))
        if partial_rotary_factor < 1.0:
            info['rope_param'].dim = int(info['size_per_head'] * partial_rotary_factor)
        # Linear attention parameters
        info['layer_types'] = cfg.get('layer_types', [])
        info['linear_key_head_dim'] = cfg.get('linear_key_head_dim', 0)
        info['linear_value_head_dim'] = cfg.get('linear_value_head_dim', 0)
        info['linear_conv_kernel_dim'] = cfg.get('linear_conv_kernel_dim', 0)
        info['linear_num_key_heads'] = cfg.get('linear_num_key_heads', 0)
        info['linear_num_value_heads'] = cfg.get('linear_num_value_heads', 0)
        # attn_output_gate doubles Q projection for full-attention layers
        info['attn_output_gate'] = cfg.get('attn_output_gate', False)
        return info


class Qwen3_5MoeReader(Qwen3_5ReaderMixin, Qwen3MoeReader):

    def _unpacked_moe_expert(self, e: int, i: int, kind: str):
        prefix = f'{self.attn_layer_prefix}.{i}.mlp.experts'
        gate_up = self.params.get(f'{prefix}.gate_up_proj.{kind}')
        down = self.params.get(f'{prefix}.down_proj.{kind}')
        if gate_up is None or down is None:
            return None

        # Packed Qwen3.5 MoE checkpoints store all experts in the first
        # dimension. Slice one expert before transform so quantized policies
        # still see a 2D tensor.
        gate_up = self.transform(gate_up[e], kind)
        down = self.transform(down[e], kind)
        gate, up = gate_up.chunk(2, dim=0)
        return (gate, down, up)

    def moe_ffn_expert(self, e=None, i=None, kind=None):
        if not kind:
            return self.filter(r'experts', i)
        unpacked = self._unpacked_moe_expert(e, i, kind)
        if unpacked is not None:
            return unpacked

        return super().moe_ffn_expert(e, i, kind)


@INPUT_MODELS.register_module(name='qwen3_5-moe')
class Qwen3_5MoeModel(Qwen3MoeModel):
    Reader = Qwen3_5MoeReader

    @staticmethod
    def map_packed_qwen35_experts(name: str):
        """Map packed expert names to weight names, i.e.,
        "mlp.experts.gate_up_proj" -> "mlp.experts.gate_up_proj.weight" so that
        class Weight in parameter.py can classify them."""
        s = re.sub(r'(mlp\.experts\.(?:gate_up|down)_proj)$', r'\1.weight', name)
        return s

    def readers(self):
        pattern = getattr(self.Reader, 'attn_layer_pattern', self.Reader.attn_layer_patten)
        loader = create_loader(self.model_path, pattern, [])

        has_packed_gate_up = any('mlp.experts.gate_up_proj' in k for k in loader.index.keys())
        has_packed_down = any('mlp.experts.down_proj' in k for k in loader.index.keys())
        if has_packed_gate_up and has_packed_down:
            loader.mappings = [self.map_packed_qwen35_experts]

        for i, param in loader.items():
            reader = self.Reader(param, {}, False, self.model_config, policy=self.policy, fp8_quant=self.fp8_quant)
            yield i, reader
        torch.cuda.empty_cache()

    def model_info(self):
        if 'text_config' in self.model_config:
            self.model_config = self.model_config['text_config']
        cfg = self.model_config
        info = super().model_info()
        # Shared expert params (missing from Qwen3MoeModel base)
        info['inter_size'] = cfg.get('shared_expert_intermediate_size', 0)
        info['moe_shared_gate'] = True
        # Qwen3.5 uses sigmoid MoE routing (not softmax)
        info['scoring_func'] = 'sigmoid'
        info['norm_topk_prob'] = True
        # Fix RoPE dim for partial_rotary_factor
        rope_params = cfg.get('rope_parameters', {})
        partial_rotary_factor = rope_params.get('partial_rotary_factor', cfg.get('partial_rotary_factor', 1.0))
        if partial_rotary_factor < 1.0:
            info['rope_param'].dim = int(info['size_per_head'] * partial_rotary_factor)
        # Linear attention parameters
        info['layer_types'] = cfg.get('layer_types', [])
        info['linear_key_head_dim'] = cfg.get('linear_key_head_dim', 0)
        info['linear_value_head_dim'] = cfg.get('linear_value_head_dim', 0)
        info['linear_conv_kernel_dim'] = cfg.get('linear_conv_kernel_dim', 0)
        info['linear_num_key_heads'] = cfg.get('linear_num_key_heads', 0)
        info['linear_num_value_heads'] = cfg.get('linear_num_value_heads', 0)
        # attn_output_gate doubles Q projection for full-attention layers
        info['attn_output_gate'] = cfg.get('attn_output_gate', False)
        return info

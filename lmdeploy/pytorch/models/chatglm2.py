# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.engine.input_process import (BaseModelInputProcessor,
                                                   PreprocessInputResult)
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RMSNorm, RopeType,
                                 SiluAndMul, build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_merged_colwise_linear,
                                        build_qkv_proj, build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h] and returns output of
    the same size.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)

        self.projection_size = config.kv_channels * config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = self.num_attention_heads
        self.head_size = (self.projection_size // config.num_attention_heads)
        self.multi_query_attention = config.multi_query_attention
        if self.multi_query_attention:
            self.num_kv_heads = config.multi_query_group_num
        self.query_key_value = build_qkv_proj(
            config.hidden_size,
            num_q_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            bias=config.add_bias_linear or config.add_qkv_bias,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # apply rotary
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            self.num_attention_heads,
            self.head_size,
            num_kv_heads=self.num_kv_heads,
        )

        # o_proj
        self.dense = build_rowwise_linear(self.projection_size,
                                          config.hidden_size,
                                          bias=config.add_bias_linear,
                                          quant_config=quantization_config,
                                          dtype=dtype,
                                          device=device,
                                          is_tp=True)

    @staticmethod
    def _extract_rope(states: torch.Tensor):
        """extract rope."""
        rope = states.chunk(2, -1)[0]
        rope = rope.unflatten(-1, (-1, 2))
        rope = rope.transpose(-2, -1).flatten(-2, -1).contiguous()
        return rope

    @staticmethod
    def _fill_rope(states: torch.Tensor, rope: torch.Tensor):
        """fill rope."""
        rope_part = states.chunk(2, -1)[0]
        rope = rope.unflatten(-1, (2, -1))
        rope = rope.transpose(-2, -1).flatten(-2, -1)
        rope_part.copy_(rope)
        return states

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        # qkv proj
        qkv_states = self.query_key_value(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        (query_states, key_states,
         value_states) = self.query_key_value.split_qkv(qkv_states)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        q_rope = self._extract_rope(query_states)
        k_rope = self._extract_rope(key_states)
        q_rope, k_rope = self.apply_rotary_pos_emb(
            q_rope,
            k_rope,
            cos,
            sin,
            inplace=True,
        )
        query_states = self._fill_rope(query_states, q_rope)
        key_states = self._fill_rope(key_states, k_rope)

        # attention
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            k_scales_zeros=None
            if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None
            if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.dense(attn_output)
        return attn_output


class MLP(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)

        self.add_bias = config.add_bias_linear
        # gate up
        self.dense_h_to_4h = build_merged_colwise_linear(
            config.hidden_size,
            [config.ffn_hidden_size, config.ffn_hidden_size],
            bias=self.add_bias,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.dense_4h_to_h = build_rowwise_linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=True)

    def forward(self, x):
        """forward."""
        gate_up = self.dense_h_to_4h(x)
        act = self.act_fn(gate_up)
        return self.dense_4h_to_h(act)


class GLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an output of
    the same size.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 layer_number: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_number = layer_number
        self.apply_residual_connection_post_layernorm = \
            config.apply_residual_connection_post_layernorm
        assert not self.apply_residual_connection_post_layernorm

        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attention = SelfAttention(config, dtype=dtype, device=device)

        # build MLP
        self.mlp = MLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.layernorm_epsilon,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.layernorm_epsilon,
            quant_config=quantization_config,
            dtype=dtype,
            device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):

        if residual is None:
            residual = hidden_states
            layernorm_output = self.input_layernorm(hidden_states)
        else:
            layernorm_output, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        layernorm_input = self.self_attention(
            hidden_states=layernorm_output,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        layernorm_output, residual = self.post_attention_layernorm(
            layernorm_input, residual)
        mlp_output = self.mlp(layernorm_output)

        outputs = (mlp_output, residual)
        return outputs


class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.num_layers = config.num_layers
        self.post_layer_norm = config.post_layer_norm

        def build_layer(layer_number):
            """build layer."""
            return GLMBlock(config, layer_number, dtype=dtype, device=device)

        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            assert config.rmsnorm
            self.final_layernorm = RMSNorm(config.hidden_size,
                                           config.layernorm_epsilon,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device)

    def _get_layer(self, layer_number: int):
        """get layer."""
        return self.layers[layer_number]

    def forward(
        self,
        hidden_states: torch.LongTensor,
        rotary_pos_emb: List[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        attn_metadata: Any,
    ):
        """forward."""
        residual = None
        for index in range(self.num_layers):
            layer = self._get_layer(index)
            hidden_states, residual = layer(
                hidden_states,
                rotary_pos_emb,
                past_key_value=past_key_values[index],
                residual=residual,
                attn_metadata=attn_metadata,
            )

        if self.post_layer_norm:
            hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class Embedding(nn.Module):
    """Language model embeddings."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(config.padded_vocab_size,
                                            self.hidden_size,
                                            dtype=dtype,
                                            device=device)
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        """Rewrite to not transpose hidden_statens for all models."""
        # Embeddings.
        embeddings = self.word_embeddings(input_ids)
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class ChatGLMModel(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.embedding = Embedding(config, dtype=dtype, device=device)

        # build rotary embedding
        emb_type = RopeType.LinearScaling
        rotary_dim = (config.hidden_size // config.num_attention_heads
                      if config.kv_channels is None else config.kv_channels)
        rope_max_pos_emb = 1 << 20
        rope_base = 10000 * getattr(config, 'rope_ratio', 1.0)
        self.rotary_pos_emb = build_rotary_embedding(
            rotary_dim // 2,
            rope_max_pos_emb,
            rope_base,
            emb_type=emb_type,
        )

        # build encoder
        self.encoder = GLMTransformer(config, dtype=dtype, device=device)

        # output_layers
        self.output_layer = build_rowwise_linear(config.hidden_size,
                                                 config.padded_vocab_size,
                                                 bias=False,
                                                 dtype=dtype,
                                                 device=device)

        self.vision = None
        if hasattr(config, 'vision_config'):
            from .cogvlm import EVA2CLIPModel
            self.vision = EVA2CLIPModel(config, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        images: torch.Tensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """forward."""

        # token embedding
        if inputs_embeds is None:
            images_features = None
            if images is not None:
                images_features = self.vision(images)
                images_features = images_features.flatten(0, 1)[None]
            inputs_embeds = self.embedding(input_ids)
            if images is not None:
                from lmdeploy.vl.constants import IMAGE_DUMMY_TOKEN_INDEX
                vis_mask = input_ids == IMAGE_DUMMY_TOKEN_INDEX
                inputs_embeds.masked_scatter_(vis_mask[..., None],
                                              images_features)

        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_pos_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        hidden_states = self.encoder(
            hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
        )

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.embedding


class ChatGLMForConditionalGeneration(nn.Module, DeployModelMixin,
                                      CudaGraphMixin):
    """rewrote model of LlamaForCausalLM."""

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # build Model
        self.transformer = ChatGLMModel(config, dtype=dtype, device=device)

        self.input_processor = ChatGLMInputProcessor(self.config, dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        images: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """model forward, return logits."""
        hidden_states = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            images=images,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.transformer.output_layer(hidden_states)

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.transformer.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        images = None
        if context.input_multimodals is not None:
            images = [
                input_mm.get('image', [])
                for input_mm in context.input_multimodals
            ]
            # flatten batch
            images = [data for im_data in images for data in im_data]
            if len(images) != 0:
                images = torch.stack([data.data for data in images])

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            images=images,
            inputs_embeds=inputs_embeds,
        )

    def update_model_metas(self,
                           past_key_values: List[List[torch.Tensor]],
                           inputs_embeds: Optional[torch.Tensor] = None,
                           context: StepContext = None):
        """update model meta."""
        model_metas = context.model_metas
        if not hasattr(self.config, 'vision_config'):
            return model_metas

        input_multimodals = context.input_multimodals
        if input_multimodals is None:
            input_imgs = [[] for _ in model_metas]
        else:
            input_imgs = []
            for mm in input_multimodals:
                if mm is None:
                    input_imgs.append([])
                else:
                    input_imgs.append(mm.get('image', []))

        config = self.config
        image_size: int = config.vision_config['image_size']
        patch_size: int = config.vision_config['patch_size']
        vision_token_num = ((image_size // patch_size // 2) *
                            (image_size // patch_size // 2) + 2)
        num_pad = vision_token_num - 3

        batched_num_img_tokens = []
        new_model_metas = []
        for meta, imgs in zip(model_metas, input_imgs):
            if meta is None:
                num_img_tokens = 0
            else:
                num_img_tokens = meta.get('num_img_tokens', 0)

            batched_num_img_tokens.append(num_img_tokens)

            num_img_tokens += num_pad * len(imgs)
            new_model_metas.append(dict(num_img_tokens=num_img_tokens))

        # prepare cogvlm position_ids
        q_seqlens = context.q_seqlens
        position_ids = context.position_ids

        if context.is_decoding or all(len(imgs) == 0 for imgs in input_imgs):
            num_img_tokens = torch.tensor(batched_num_img_tokens,
                                          device=position_ids.device)
            position_ids -= num_img_tokens[None]
        else:
            batched_position_ids = position_ids[0].split(q_seqlens)
            for pos_ids, num_img_tok, imgs in zip(batched_position_ids,
                                                  batched_num_img_tokens,
                                                  input_imgs):
                pos_ids -= num_img_tok
                if len(imgs) == 0:
                    continue

                seq_len = pos_ids.size(0)
                start = pos_ids[0].cpu().item()
                new_pos_ids = []

                imgs = sorted(imgs, key=lambda img: img.start)
                for img in imgs:
                    new_pos_ids += list(range(start, img.start + 1))
                    new_pos_ids += [img.start + 1] * (img.end - img.start - 2)
                    start = img.start + 2

                remain = seq_len - len(new_pos_ids)
                new_pos_ids += list(range(start, start + remain))

                new_pos_ids = pos_ids.new_tensor(new_pos_ids)
                pos_ids[:] = new_pos_ids

            position_ids = torch.cat(batched_position_ids)[None]
        context.position_ids = position_ids

        return new_model_metas

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        # modify from vllm

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'transformer.vision' in name:
                param = params_dict[name]
                load_weight(param, loaded_weight)

            if 'rotary_pos_emb.inv_freq' in name:
                continue
            if ('rotary_pos_emb.cos_cached' in name
                    or 'rotary_pos_emb.sin_cached' in name):
                continue
            if (self.config.tie_word_embeddings
                    and 'output_layer.weight' in name):
                continue
            if '.query_key_value' in name:
                param = params_dict[name]
                q, k, v = param.weight_spliter(loaded_weight)
                load_weight(param, q, shard_id='q')
                load_weight(param, k, shard_id='k')
                load_weight(param, v, shard_id='v')
            elif '.dense_h_to_4h' in name:
                param = params_dict[name]
                gate, up = param.weight_spliter(loaded_weight)
                load_weight(param, gate, shard_id=0)
                load_weight(param, up, shard_id=1)
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)

    def get_input_processor(self) -> BaseModelInputProcessor:
        """get input processor."""
        return self.input_processor


class ChatGLMInputProcessor(BaseModelInputProcessor):
    """input processor."""

    def __init__(self, config: PretrainedConfig, dtype) -> None:
        self.config = config
        self.dtype = dtype

        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
            self.image_size = vision_config.image_size
            self.patch_size = vision_config.patch_size
            self.num_patches = (self.image_size // self.patch_size)**2
            self.num_positions = self.num_patches + 1
            self.vision_token_num = self.num_patches // 4

    def preprocess_input(self,
                         input_ids: List[int],
                         input_multimodals: List[Dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        """prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values'].to(self.dtype)
            offset = input_mm['offset']

            mm_data = MultiModalTensor(data=pixel_values,
                                       start=offset,
                                       end=offset + self.vision_token_num)
            input_imgs.append(mm_data)

        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=dict(image=input_imgs),
        )
        return result

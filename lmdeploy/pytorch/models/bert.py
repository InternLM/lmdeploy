# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from HuggingFace Transformers' BERT modeling code for LMDeploy inference.

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from lmdeploy.pytorch.nn import LayerNorm
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_rowwise_linear


class BertConfig:

    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 intermediate_size: int,
                 max_position_embeddings: int,
                 add_cross_attention: bool = True,
                 is_decoder: bool = True,
                 cross_attention_freq: int = 2,
                 hidden_act: str = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 type_vocab_size: int = 2,
                 pad_token_id: int = 0,
                 initializer_range: float = 0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.add_cross_attention = add_cross_attention
        self.is_decoder = is_decoder
        self.cross_attention_freq = cross_attention_freq
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.type_vocab_size = type_vocab_size
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range


def _make_causal_mask(input_shape: torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    batch_size, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len)


def _init_bert_weights(module: nn.Module, config: BertConfig):
    with torch.no_grad():
        if isinstance(module, nn.Embedding):
            module.weight.normal_(mean=0.0, std=config.initializer_range)
            if module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.weight.fill_(1.0)
            if module.bias is not None:
                module.bias.zero_()
        elif hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter) and module.weight.dim() == 2:
            module.weight.normal_(mean=0.0, std=config.initializer_range)
            if getattr(module, 'bias', None) is not None:
                module.bias.zero_()


def _prepare_cross_attention_mask(attention_mask: torch.Tensor | None, inputs_embeds: torch.Tensor,
                                  encoder_hidden_states: torch.Tensor | None):
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        return attention_mask.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

    batch_size, query_length = inputs_embeds.shape[:2]
    key_length = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else attention_mask.shape[-1]
    expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, query_length, key_length)
    expanded_mask = expanded_mask.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)


class BertEmbeddings(nn.Module):

    def __init__(self, config: BertConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id, dtype=dtype, device=device)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, dtype=dtype, device=device)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size, dtype=dtype, device=device)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype, device=device)
        self.register_buffer(
            'position_ids', torch.arange(config.max_position_embeddings, device=device).expand((1, -1)),
            persistent=False)
        self.register_buffer('token_type_ids', torch.zeros(self.position_ids.size(), dtype=torch.long, device=device),
                             persistent=False)

    def forward(self,
                input_ids: torch.Tensor | None = None,
                token_type_ids: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :seq_length].expand(batch_size, seq_length)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        return self.LayerNorm(embeddings)


class BertSelfAttention(nn.Module):

    def __init__(self, config: BertConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f'hidden_size must be divisible by num_attention_heads, got {config.hidden_size} and '
                f'{config.num_attention_heads}.')

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        linear_kwargs = dict(dtype=dtype, device=device)
        self.query = build_colwise_linear(config.hidden_size, self.all_head_size, bias=True, **linear_kwargs)
        self.key = build_colwise_linear(config.hidden_size, self.all_head_size, bias=True, **linear_kwargs)
        self.value = build_colwise_linear(config.hidden_size, self.all_head_size, bias=True, **linear_kwargs)

    def _shape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tensor.shape[:2]
        tensor = tensor.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return tensor.transpose(1, 2)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                encoder_hidden_states: torch.Tensor | None = None):
        key_value_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        query_layer = self._shape(self.query(hidden_states))
        key_layer = self._shape(self.key(key_value_states))
        value_layer = self._shape(self.value(key_value_states))

        attn_output = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output.reshape(hidden_states.shape[0], hidden_states.shape[1], self.all_head_size)


class BertSelfOutput(nn.Module):

    def __init__(self, config: BertConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.dense = build_rowwise_linear(
            config.hidden_size, config.hidden_size, bias=True, dtype=dtype, device=device)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertAttention(nn.Module):

    def __init__(self, config: BertConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.self = BertSelfAttention(config, dtype=dtype, device=device)
        self.output = BertSelfOutput(config, dtype=dtype, device=device)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                encoder_hidden_states: torch.Tensor | None = None):
        attention_output = self.self(
            hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states)
        return self.output(attention_output, hidden_states)


class BertIntermediate(nn.Module):

    def __init__(self, config: BertConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.dense = build_colwise_linear(
            config.hidden_size, config.intermediate_size, bias=True, dtype=dtype, device=device)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class BertOutput(nn.Module):

    def __init__(self, config: BertConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.dense = build_rowwise_linear(
            config.intermediate_size, config.hidden_size, bias=True, dtype=dtype, device=device)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertLayer(nn.Module):

    def __init__(self, config: BertConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.attention = BertAttention(config, dtype=dtype, device=device)
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            self.crossattention = BertAttention(config, dtype=dtype, device=device)
        self.intermediate = BertIntermediate(config, dtype=dtype, device=device)
        self.output = BertOutput(config, dtype=dtype, device=device)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                encoder_hidden_states: torch.Tensor | None = None,
                encoder_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)

        if self.add_cross_attention and encoder_hidden_states is not None:
            attention_output = self.crossattention(
                attention_output, attention_mask=encoder_attention_mask, encoder_hidden_states=encoder_hidden_states)

        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class BertEncoder(nn.Module):

    def __init__(self, config: BertConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.layer = nn.ModuleList(
            [BertLayer(config, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                encoder_hidden_states: torch.Tensor | None = None,
                encoder_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
        return hidden_states


class BertPooler(nn.Module):

    def __init__(self, config: BertConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.dense = build_colwise_linear(
            config.hidden_size, config.hidden_size, bias=True, dtype=dtype, device=device)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(hidden_states[:, 0]))


class BertModel(nn.Module):

    def __init__(self, config: BertConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config, dtype=dtype, device=device)
        self.encoder = BertEncoder(config, dtype=dtype, device=device)
        self.pooler = BertPooler(config, dtype=dtype, device=device)
        self.apply(lambda module: _init_bert_weights(module, config))

    def forward(self,
                input_ids: torch.Tensor | None = None,
                token_type_ids: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                encoder_hidden_states: torch.Tensor | None = None,
                encoder_attention_mask: torch.Tensor | None = None):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds.')

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        self_attention_mask = _make_causal_mask(
            embedding_output.shape[:2], embedding_output.dtype, embedding_output.device)
        cross_attention_mask = _prepare_cross_attention_mask(
            encoder_attention_mask, embedding_output, encoder_hidden_states)

        sequence_output = self.encoder(
            embedding_output,
            attention_mask=self_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=cross_attention_mask,
        )
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output

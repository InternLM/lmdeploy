# Copyright (c) OpenMMLab. All rights reserved.

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.nn import LayerNorm
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_rowwise_linear

from .whisper import WhisperEncoderLayer


class InternS1ProTimeSeriesEncoder(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config

        self.embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, self.embed_dim, kernel_size=3, padding=1, dtype=dtype, device=device)
        self.conv2 = nn.Conv1d(self.embed_dim,
                               self.embed_dim,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               dtype=dtype,
                               device=device)
        self.embed_positions = nn.Embedding(self.max_source_positions, self.embed_dim, dtype=dtype, device=device)

        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config, dtype=dtype, device=device) for _ in range(config.encoder_layers)])
        self.layer_norm = LayerNorm(config.d_model, eps=1e-5, dtype=dtype, device=device)

        self.adapt_in = build_colwise_linear(
            in_features=config.ts_adapt_in_dim,
            out_features=80,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.adapt_out = build_rowwise_linear(
            in_features=self.embed_dim,
            out_features=config.ts_adapt_out_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )

    def _make_causal_mask(self,
                          input_ids_shape: torch.Size,
                          dtype: torch.dtype,
                          device: torch.device,
                          past_key_values_length: int = 0):
        """Make causal mask used for bi-directional self-attention."""
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _prepare_decoder_attention_mask(self, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        return combined_attention_mask

    def forward(self, input_features):
        # (N, T, C) -> (T, N, C) -> (N, C, T)
        input_features = input_features.permute(1, 0, 2)
        input_features = self.adapt_in(input_features)
        input_features = input_features.permute(1, 2, 0)

        # (N, C, T) -> (N, C, T//2)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        # (N, C, T) -> (N, T, C)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        if inputs_embeds.shape[1] > embed_pos.shape[0]:
            target_len = inputs_embeds.shape[1]
            padding = [0, 0, 0, target_len - embed_pos.shape[0]]

            embed_pos = nn.functional.pad(embed_pos, pad=padding, mode='constant', value=0)
            hidden_states = inputs_embeds[:, :embed_pos.shape[0], :] + embed_pos
        else:
            hidden_states = inputs_embeds + embed_pos[:inputs_embeds.shape[1], :]

        input_shape = inputs_embeds.size()[:-1]
        past_key_values_length = 0
        attention_mask = self._prepare_decoder_attention_mask(input_shape, inputs_embeds, past_key_values_length)

        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(hidden_states, attention_mask)
            hidden_states = layer_outputs

        # (N, T, C) -> (T, N, C)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.adapt_out(hidden_states)

        # (T, N, C) -> (N, T, C)
        hidden_states = hidden_states.permute(1, 0, 2)

        return hidden_states


class InternS1ProTimeSeriesConcatSubsampling(nn.Module):

    def __init__(self, in_channels: int, concat_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * concat_size

    def forward(self, ts_signals: torch.Tensor, ts_lens: torch.Tensor):
        if ts_signals.shape[1] % 2 != 0:
            ts_signals = ts_signals[:, :-1, :]
        even_frames = ts_signals[:, ::2, :]
        odd_frames = ts_signals[:, 1::2, :]
        ts_signals = torch.cat((even_frames, odd_frames), dim=2)
        ts_lens = ts_lens // 2
        return ts_signals, ts_lens


class InternS1ProTimeSeriesFixPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=20000, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # TODO: zhouxinyu, hf forces float32 during init, but becomes bf16 during forward
        pe = pe.unsqueeze(0).transpose(0, 1).to(dtype=dtype, device=device)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe, persistent=True)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x.clone()


class InternS1ProTimeSeriesMultiChannelAdaptiveSubsampling(nn.Module):

    def __init__(self,
                 hidden_dim=128,
                 nhead=8,
                 num_encoder_layers=1,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=hidden_dim,
                              kernel_size=5,
                              stride=1,
                              padding=2,
                              dtype=dtype,
                              device=device)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dtype=dtype, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.pos_encoder = InternS1ProTimeSeriesFixPositionalEncoding(d_model=hidden_dim, dtype=dtype, device=device)
        self.subsampling = InternS1ProTimeSeriesConcatSubsampling(128, 2)

    def forward(self, inputs, input_lens, sr):
        sr = torch.as_tensor(sr, dtype=torch.float32)
        strides = torch.floor(160 / ((1 + torch.exp(-sr / 100))**6))
        patch_sizes = strides * 2
        patched_outputs = []
        output_lens = []

        for i in range(len(inputs)):
            seq = inputs[i]  # [seq_len, num_channel]
            ps = patch_sizes[i].item()
            st = strides[i].item()
            le = input_lens[i]

            output_len = torch.ceil((le - ps) / st) + 1
            pad_len = ((output_len - 1) * st + ps - le).long().item()
            if seq.ndim == 1:
                seq = seq.unsqueeze(-1)
            seq = nn.functional.pad(seq, (0, 0, 0, pad_len), 'constant', 0)
            assert output_len > 0, (seq.shape, ps, st, le, output_len)
            output_lens.append(output_len)
            indices = (torch.arange(0, output_len * st, st).unsqueeze(1) + torch.arange(ps)).long()
            patched = seq[indices]

            output = self.forward_encoder(patched)  # [num_patch, D]
            patched_outputs.append(output)

        outputs = nn.utils.rnn.pad_sequence(patched_outputs, batch_first=True)
        output_lens = torch.tensor(output_lens).squeeze().to(outputs.device).long()
        if output_lens.ndim == 0:
            output_lens = output_lens.unsqueeze(0)

        outputs, output_lens = self.subsampling(outputs.clone(), output_lens.clone())
        return outputs, output_lens

    def forward_encoder(self, x):
        num_patch, patch_len, C = x.shape
        # conv1
        # treat each channel as an independent sample and feed it into conv1
        x = x.reshape(num_patch * C, 1, patch_len)
        x = nn.functional.relu((self.conv(x)))  # [B*C, D1, L]
        x = x.permute(2, 0, 1)  # [L, B*C, D1]

        x = self.pos_encoder(x)  # [L, B*C, D1]
        x = self.transformer_encoder(x)
        x = x.mean(0)

        x = x.reshape(num_patch, C, -1)

        return x.mean(1)


class InternS1ProTimeSeriesProjector(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.layer_norm = LayerNorm(config.ts_hidden_dim, eps=1e-5, dtype=dtype, device=device)
        self.linear_1 = build_colwise_linear(in_features=config.ts_hidden_dim,
                                             out_features=config.out_hidden_size,
                                             bias=True,
                                             dtype=dtype,
                                             device=device)
        self.act = ACT2FN[config.activation_function]
        self.linear_2 = build_rowwise_linear(in_features=config.out_hidden_size,
                                             out_features=config.out_hidden_size,
                                             bias=True,
                                             dtype=dtype,
                                             device=device)

    def forward(self, ts_features):
        hidden_states = self.layer_norm(ts_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class InternS1ProTimeSeriesModel(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.encoder_embed = InternS1ProTimeSeriesMultiChannelAdaptiveSubsampling(dtype=dtype, device=device)
        self.encoder = InternS1ProTimeSeriesEncoder(config, dtype=dtype, device=device)
        self.projector = InternS1ProTimeSeriesProjector(config, dtype=dtype, device=device)

    def forward(
        self,
        time_series_signals: Optional[torch.FloatTensor] = None,
        ts_lens: Optional[torch.Tensor] = None,
        sr: Optional[torch.Tensor] = None,
        time_series_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple]:
        if time_series_signals is None and time_series_embeds is None:
            raise ValueError('You have to specify time_series_signals or time_series_embeds')

        # embedded values can be passed in directly, but the dimensions must match
        if time_series_embeds is not None and len(
                time_series_embeds.shape) == 3 and time_series_embeds.shape[-1] == self.config.ts_adapt_in_dim:
            time_series_embeds = time_series_embeds
        else:
            if ((isinstance(time_series_signals, list) and len(time_series_signals[0].shape) == 2)
                    or (isinstance(time_series_signals, torch.Tensor) and len(time_series_signals.shape) == 3)):
                time_series_embeds, ts_lens = self.encoder_embed(time_series_signals, ts_lens, sr)
            else:
                raise ValueError(f'wrong time_series_signals size: {time_series_signals[0].shape}')

        # [B, 64000, 1] -> [B, 200, 256] -> [B, 100, 1024]
        last_hidden_state = self.encoder(input_features=time_series_embeds)

        # ts_lens after encoder
        ts_lens = (ts_lens + 1) // 2
        assert torch.all(ts_lens > 0), f'The length of time_series_embeds is so small. ts_lens: {ts_lens}'

        last_hidden_state = self.projector(last_hidden_state)
        return last_hidden_state

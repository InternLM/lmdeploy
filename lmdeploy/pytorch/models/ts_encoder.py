# Copyright (c) OpenMMLab. All rights reserved.

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperConfig, WhisperPreTrainedModel

from lmdeploy.pytorch.nn import LayerNorm
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_rowwise_linear

from .bert import BertConfig, BertModel
from .whisper import WhisperEncoderLayer


class FixPositionalEncoding(nn.Module):

    def __init__(self,
                 d_model: int,
                 max_len: int = 200000,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        pe = torch.zeros(max_len, 1, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.to(dtype=dtype, device=device)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :, :]
        return x


class CustomWhisperEncoder(WhisperPreTrainedModel):
    """Whisper encoder with time-series input/output adapters."""

    def __init__(self, config: WhisperConfig, dtype: torch.dtype | None = None, device: torch.device | None = None):
        super().__init__(config)

        self.embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(
            self.num_mel_bins, self.embed_dim, kernel_size=3, padding=1, dtype=dtype, device=device)
        self.conv2 = nn.Conv1d(
            self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1, dtype=dtype, device=device)
        self.embed_positions = nn.Embedding(self.max_source_positions, self.embed_dim, dtype=dtype, device=device)

        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config, dtype=dtype, device=device) for _ in range(config.encoder_layers)])
        self.layer_norm = LayerNorm(config.d_model, eps=1e-5, dtype=dtype, device=device)

        self.adapt_in = build_colwise_linear(
            config.ts_adapt_in_dim,
            80,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.adapt_out = build_rowwise_linear(
            self.embed_dim,
            config.ts_adapt_out_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )

        self.mask_type = None
        self.chunk_length = None

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def define_masktype(self, masktype: str, chunk_length: int | None = None):
        self.mask_type = masktype
        self.chunk_length = chunk_length

    def _make_causal_mask(self,
                          input_ids_shape: torch.Size,
                          dtype: torch.dtype,
                          device: torch.device,
                          past_key_values_length: int = 0) -> torch.Tensor:
        """Create a causal attention mask in HF Whisper's expected shape."""
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

    def prepare_chunk_attention_mask(self, input_shape, inputs_embeds):
        block_size = round(self.chunk_length / 4 * 2)
        matrix_size = input_shape[1]

        matrix = torch.ones(matrix_size, matrix_size)

        num_full_blocks = round(matrix_size // block_size)
        remainder = matrix_size % block_size
        for i in range(num_full_blocks):
            row_start = i * block_size
            col_start = i * block_size
            matrix[row_start:row_start + block_size, col_start:col_start + block_size] = torch.zeros(
                block_size, block_size)

        if remainder > 0:
            last_row_start = num_full_blocks * block_size
            last_col_start = num_full_blocks * block_size
            matrix[last_row_start:last_row_start + remainder,
                   last_col_start:last_col_start + remainder] = torch.zeros(remainder, remainder)

        matrix = matrix * -65504
        matrix = matrix.unsqueeze(0).unsqueeze(0).repeat(input_shape[0], 1, 1, 1)
        attention_mask = matrix.to(inputs_embeds.device)
        return attention_mask

    def prepare_padding_mask(self, input_shape, inputs_embeds, input_lens):
        matrix_size = input_shape[1]
        matrix_list = []

        for i in range(input_shape[0]):
            padding_matrix = torch.ones(matrix_size, matrix_size) * -65504
            padding_matrix[:input_lens[i], :input_lens[i]] = 0
            matrix_list.append(padding_matrix)

        attention_mask = torch.stack(matrix_list).unsqueeze(1).to(inputs_embeds.dtype).to(inputs_embeds.device)

        return attention_mask

    def forward(
        self,
        input_features,
        input_length=None,
        causal=True,
    ):
        input_features = self.adapt_in(input_features)
        input_features = input_features.permute(1, 2, 0)

        inputs_embeds = F.gelu(self.conv1(input_features))
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        if inputs_embeds.shape[1] > embed_pos.shape[0]:
            target_len = inputs_embeds.shape[1]
            padding = [0, 0, 0, target_len - embed_pos.shape[0]]

            embed_pos = F.pad(embed_pos, pad=padding, mode='constant', value=0)
            hidden_states = inputs_embeds[:, :embed_pos.shape[0], :] + embed_pos
        else:
            hidden_states = inputs_embeds + embed_pos[:inputs_embeds.shape[1], :]

        input_shape = inputs_embeds.size()[:-1]
        past_key_values_length = 0
        if causal:
            if self.mask_type == 'chunk':
                attention_mask = self.prepare_chunk_attention_mask(input_shape, inputs_embeds)
            else:
                attention_mask = self._prepare_decoder_attention_mask(input_shape, inputs_embeds,
                                                                      past_key_values_length)
        else:
            attention_mask = self.prepare_padding_mask(input_shape, inputs_embeds, (input_length + 1) // 2)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states, attention_mask)
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.adapt_out(hidden_states)

        lengths = (input_length + 1) // 2

        return hidden_states, lengths


class MRQFormer(nn.Module):

    def __init__(self,
                 num_query_list: list[int],
                 hidden_size: int = 1024,
                 num_layers: int = 6,
                 cross_attention_freq: int = 2,
                 encoder_hidden_size: int | None = None,
                 output_size: int | None = None,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()

        self.num_query_list = num_query_list
        self.hidden_size = hidden_size

        self.query_token_list = nn.ParameterList()
        for num_queries in self.num_query_list:
            self.query_token_list.append(
                nn.Parameter(torch.randn(1, num_queries, hidden_size, dtype=dtype, device=device)))

        bert_config = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=hidden_size // 16,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max(self.num_query_list),
            add_cross_attention=True,
            is_decoder=True,
            cross_attention_freq=cross_attention_freq,
        )
        self.transformer = BertModel(bert_config, dtype=dtype, device=device)

        if encoder_hidden_size is not None and encoder_hidden_size != hidden_size:
            self.encoder_proj = build_colwise_linear(
                encoder_hidden_size,
                hidden_size,
                bias=True,
                dtype=dtype,
                device=device,
            )
        else:
            self.encoder_proj = nn.Identity()

        if output_size is None:
            output_size = hidden_size

        self.proj_out = nn.Sequential(
            build_colwise_linear(
                in_features=hidden_size,
                out_features=4 * hidden_size,
                bias=True,
                dtype=dtype,
                device=device,
            ),
            nn.ReLU(),
            build_rowwise_linear(
                in_features=4 * hidden_size,
                out_features=output_size,
                bias=True,
                dtype=dtype,
                device=device,
            ),
        )

    def forward(self, encoder_features: torch.Tensor, attention_mask=None, res_idx: int = 0):
        batch_size = encoder_features.shape[0]

        encoder_features = self.encoder_proj(encoder_features)

        query_tokens = self.query_token_list[res_idx]
        query_tokens = query_tokens.expand(batch_size, -1, -1)

        encoder_attention_mask = None
        if attention_mask is not None:
            encoder_attention_mask = attention_mask

        outputs, _ = self.transformer(
            inputs_embeds=query_tokens,
            encoder_hidden_states=encoder_features,
            encoder_attention_mask=encoder_attention_mask
        )

        outputs = self.proj_out(outputs)

        return outputs


class SingleResChunkQformerSubsampling(nn.Module):

    def __init__(self,
                 hidden_dim: int = 128,
                 alpha: float = 0.5,
                 patch: int = 800,
                 num_query: int = 32,
                 num_conv_layers: int = 0,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()

        self.patch_list = [patch]
        self.num_query_list = [num_query]

        self.num_conv_layers = num_conv_layers
        if self.num_conv_layers == 0:
            self.conv = nn.Conv1d(
                in_channels=1,
                out_channels=hidden_dim - 2,
                kernel_size=5,
                stride=1,
                padding=2,
                dtype=dtype,
                device=device)
            self.mask_pool = None
        else:
            conv_layers = []
            pool_layers = []
            output_channels = hidden_dim - 2

            for i in range(self.num_conv_layers):
                in_channels = 1 if i == 0 else output_channels
                conv_layers.append(
                    nn.Conv1d(in_channels,
                              output_channels,
                              kernel_size=5,
                              stride=2,
                              padding=2,
                              dtype=dtype,
                              device=device))
                conv_layers.append(nn.ReLU(inplace=True))
                pool_layers.append(nn.MaxPool1d(kernel_size=5, stride=2, padding=2))

            self.conv = nn.Sequential(*conv_layers)
            self.mask_pool = nn.Sequential(*pool_layers)

        self.mrqformer = MRQFormer(
            num_query_list=self.num_query_list,
            hidden_size=hidden_dim,
            num_layers=6,
            cross_attention_freq=2,
            output_size=hidden_dim,
            dtype=dtype,
            device=device)

        self.pos_encoder = FixPositionalEncoding(d_model=hidden_dim, dtype=dtype, device=device)

        self.alpha = alpha
        channel_encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=32, dtype=dtype, device=device)
        self.channel_encoder = TransformerEncoder(channel_encoder_layers, num_layers=1)
        self.fuze_proj = build_rowwise_linear(
            in_features=hidden_dim,
            out_features=256,
            bias=True,
            dtype=dtype,
            device=device,
        )

    def forward(self,
                inputs: torch.Tensor,
                input_lens: torch.Tensor,
                sr,
                force_strides,
                mask: torch.Tensor | None = None):
        if mask is None:
            mask = torch.ones(inputs.shape).to(inputs.device)
        features, feature_lens, learnable_lens, strides, raw_outputs = self.forward_patch(
            inputs, input_lens, sr, force_strides, mask)

        return features, feature_lens, learnable_lens, strides, raw_outputs

    def forward_patch(self,
                      inputs: torch.Tensor,
                      input_lens: torch.Tensor,
                      sr,
                      force_strides,
                      mask: torch.Tensor | None = None):
        pred_strides = None

        output_list = []

        seq = inputs

        mean = seq.mean(dim=1, keepdim=True)
        std = seq.std(dim=1, keepdim=True)

        abs_mean_plus_1 = torch.abs(mean) + 1.0
        log_abs_plus_1 = torch.log(abs_mean_plus_1)
        sign_mean = torch.sign(mean)
        mean_feat = sign_mean * log_abs_plus_1

        std_feat = torch.log(std + 1e-7)

        seq = (seq - mean) / (std + 1e-7)

        max_patch = torch.tensor([max(self.patch_list)]).to(seq.device)
        max_num_query = torch.tensor([max(self.num_query_list)]).to(seq.device)
        patch_size = max_patch
        step = max_patch
        seq_len = torch.tensor(inputs.shape[1]).to(seq.device)

        output_len = torch.ceil((seq_len - patch_size) / step + 1)
        pad_len = (torch.ceil((output_len - 1) * step + patch_size - seq_len)).long().item()

        if seq.ndim == 2:
            seq = seq.unsqueeze(-1)
        seq = F.pad(seq, (0, 0, 0, pad_len, 0, 0), 'constant', 0)

        padded_mask = F.pad(mask, (0, 0, 0, pad_len, 0, 0), 'constant', 0)

        batch_size, seq_len, channels = seq.shape
        seq = seq.permute(0, 2, 1)
        seq = seq.reshape(batch_size * channels, 1, seq_len)
        if self.num_conv_layers == 0:
            seq = F.relu(self.conv(seq))
        else:
            seq = self.conv(seq)
        seq_len = seq.shape[-1]
        seq = seq.reshape(batch_size, channels, -1, seq_len)
        seq = seq.permute(0, 3, 1, 2)

        mean_feat = mean_feat.expand([batch_size, seq_len, channels]).unsqueeze(-1)
        std_feat = std_feat.expand([batch_size, seq_len, channels]).unsqueeze(-1)

        seq = torch.cat([seq, mean_feat, std_feat], dim=-1)
        hidden_dim = seq.shape[-1]

        if self.mask_pool is not None:
            padded_mask = padded_mask.permute(0, 2, 1)
            padded_mask = self.mask_pool(padded_mask)
            padded_mask = padded_mask.permute(0, 2, 1)

        for res_idx, patch in enumerate(self.patch_list):
            patch_output_len = output_len * max_patch / (patch * (2**self.num_conv_layers))

            indices = (torch.floor(torch.arange(0, patch_output_len.item()).to(seq.device) * patch).unsqueeze(1)
                       + torch.arange(patch).to(seq.device)).long()
            patched = seq[:, indices, :, :]
            _, num_patch, patch_len, _, _ = patched.shape
            patched = patched.permute(1, 2, 0, 3, 4)
            patched = patched.reshape(num_patch, patch_len, -1)
            patched = patched.reshape(num_patch, patch_len, batch_size, channels, hidden_dim)

            patched = patched.permute(2, 0, 1, 3, 4)
            patched = patched.reshape(batch_size * num_patch, patch_len, channels, hidden_dim)

            patched_mask = padded_mask[:, indices, :]
            patched_mask = patched_mask.reshape(batch_size * num_patch, patch_len, channels)

            raw_output = self.forward_encoder(patched, patched_mask, res_idx)
            output = raw_output.reshape(-1, hidden_dim)
            output_list.append(output)

        outputs = torch.cat(output_list, dim=1)
        outputs = self.fuze_proj(outputs)
        out_hidden_dim = outputs.shape[1]

        outputs = outputs.reshape(batch_size, -1, out_hidden_dim)

        output_lens = torch.ceil(padded_mask[:, :, 0].sum(dim=-1) / max_patch) * max_num_query

        learnable_lens = None
        raw_outputs = None

        return outputs, output_lens, learnable_lens, pred_strides, raw_outputs

    def forward_encoder(self, inputs: torch.Tensor, mask: torch.Tensor, res_idx: int = 0) -> torch.Tensor:
        x = inputs
        num_patch, patch_len, channels, hidden_dim = x.shape
        x = x.permute(1, 0, 2, 3)
        x = x.reshape(patch_len, num_patch * channels, -1)

        x = self.pos_encoder(x)

        mask = mask.permute(0, 2, 1)
        mask = mask.reshape(num_patch * channels, patch_len)

        chunk_size = 4096 * 8
        all_outputs = []

        for i in range(0, x.size(1), chunk_size):
            x_chunk = x[:, i:i + chunk_size, :].contiguous()
            chunk_mask = mask[i:i + chunk_size]
            x_chunk = x_chunk.permute(1, 0, 2)
            x_chunk = self.mrqformer(x_chunk, attention_mask=chunk_mask, res_idx=res_idx)
            x_chunk = x_chunk.permute(1, 0, 2)
            all_outputs.append(x_chunk)

        x = torch.cat(all_outputs, dim=1)

        x = x.reshape(-1, num_patch, channels, hidden_dim)
        x = x.permute(2, 1, 0, 3)
        x = x.reshape(channels, -1, hidden_dim)

        all_outputs = []

        for i in range(0, x.size(1), chunk_size):
            x_chunk = x[:, i:i + chunk_size, :].contiguous()
            x_chunk = self.channel_encoder(x_chunk)
            x_chunk = x_chunk.mean(0)
            all_outputs.append(x_chunk)

        x = torch.cat(all_outputs, dim=0)
        x = x.reshape(num_patch, -1, hidden_dim)

        return x


class ChunkModel(nn.Module):

    def __init__(self, encoder_embed: nn.Module, encoder: nn.Module, chunk_size: int = 6400, step: int = 6400):
        super().__init__()

        self.chunk_size = chunk_size
        self.step = step
        assert self.step <= self.chunk_size or self.chunk_size < 0

        self.encoder_embed = encoder_embed
        self.encoder = encoder

    def chunk_tensor(self, tensor: torch.Tensor):
        seq_len = tensor.shape[0]
        chunk_size = self.chunk_size
        step = self.step

        chunks = []
        masks = []

        if chunk_size > 0:
            start = 0
            while start < seq_len:
                end = min(start + chunk_size, seq_len)
                chunk = tensor[start:end, :]
                chunks.append(chunk)
                mask = torch.zeros(chunk.shape)
                mask[:end - start, :] = 1
                masks.append(mask)

                if end >= seq_len:
                    break

                start += step

            output = pad_sequence(chunks, batch_first=True)
            output_mask = pad_sequence(masks, batch_first=True).to(output.device)
        else:
            output = tensor.unsqueeze(0)
            output_mask = torch.ones(output.shape).to(tensor.device)

        return output, output_mask

    def concat_chunks_simple(self,
                             chunks: torch.Tensor,
                             chunk_lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
        """Concatenate chunks without overlap."""
        num_chunks, chunk_len, channels = chunks.shape

        if chunk_lengths is None:
            signal = chunks.reshape(-1, channels)
            signal_len = num_chunks * chunk_len
        else:
            signal_list = []
            signal_len = 0
            for i in range(num_chunks):
                valid_len = int(chunk_lengths[i].item())
                signal_list.append(chunks[i, :valid_len, :])
                signal_len += valid_len
            signal = torch.cat(signal_list, dim=0)

        return signal, signal_len

    def forward_encoder(self,
                        x: torch.Tensor,
                        x_lens: torch.Tensor,
                        x_channels: torch.Tensor,
                        sr=None,
                        force_strides=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs."""
        assert x.shape[0] == len(x_lens) and x.shape[0] == len(x_channels)
        batch_size = x.shape[0]

        outputs = []
        output_lens = []
        output_chunks = []
        for b in range(batch_size):
            seq = x[b, :x_lens[b], :x_channels[b]]
            chunks, mask = self.chunk_tensor(seq)

            chunk_output, chunk_lens, learnable_lens, _, _ = self.encoder_embed(
                chunks, x_lens[b], sr, force_strides, mask=mask)

            outputs.append(chunk_output)
            output_lens.append(chunk_lens)
            output_chunks.append(chunk_output.shape[0])

        x_lens = torch.cat(output_lens, dim=0)
        max_len = x_lens.max().item()

        # Preserve reference behavior: padded chunk embeddings are constructed
        # here but the original input tensor is fed into the final encoder.
        padded_list = []
        for tensor in outputs:
            pad_len = int(max_len - tensor.shape[1])
            padded_list.append(F.pad(tensor, (0, 0, 0, pad_len, 0, 0), mode='constant', value=0))

        if x_lens.ndim == 0:
            x_lens = x_lens.unsqueeze(-1)

        x = x.permute(1, 0, 2)
        encoder_out, encoder_out_lens = self.encoder(x, x_lens)
        encoder_out = encoder_out.permute(1, 0, 2)

        if learnable_lens is not None:
            learnable_lens = learnable_lens / 2

        total_chunks = encoder_out.shape[0]
        assert sum(output_chunks) == total_chunks, f'chunk count mismatch: {sum(output_chunks)} vs {total_chunks}'

        features = []
        feature_lens = []
        start = 0

        for count in output_chunks:
            end = start + count
            signal_chunks = encoder_out[start:end]

            feature, feature_len = self.concat_chunks_simple(signal_chunks, encoder_out_lens[start:end])

            features.append(feature)
            start = end
            feature_lens.append(feature_len)

        encoder_out = pad_sequence(features, batch_first=True)
        encoder_out_lens = torch.tensor(feature_lens).to(encoder_out.device)

        return encoder_out, encoder_out_lens

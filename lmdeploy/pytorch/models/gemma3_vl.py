# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .patch import build_model_from_hf_config
from .siglip import SiglipVisionModel
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin


class Gemma3RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.eps}'


class Gemma3MultiModalProjector(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size, dtype=dtype, device=device))

        self.mm_soft_emb_norm = Gemma3RMSNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(batch_size, seq_length, self.patches_per_image,
                                                                  self.patches_per_image)
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
        return projected_vision_outputs.type_as(vision_outputs)


class Gemma3VLInputProcessor(BaseModelInputProcessor):
    """Internvl input processor."""

    def __init__(self, config: PretrainedConfig, dtype) -> None:
        self.config = config
        self.dtype = dtype

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
        """Prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values'].to(self.dtype)
            offset = input_mm['offset']
            image_token_id = input_mm['image_token_id']
            num_pad = input_mm['image_tokens']
            if isinstance(num_pad, torch.Tensor):
                num_pad = num_pad.item()

            mm_data = MultiModalTensor(data=pixel_values,
                                       start=offset,
                                       end=offset + num_pad,
                                       meta=dict(image_token_id=image_token_id))
            input_imgs.append(mm_data)

        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=dict(image=input_imgs),
        )
        return result


class Gemma3ForConditionalGeneration(nn.Module, CudaGraphMixin, DeployModelMixin):

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        text_config = config.text_config
        self.sliding_window = text_config.sliding_window
        self.language_model = build_model_from_hf_config(text_config, dtype=dtype, device=device)
        self.vision_tower = SiglipVisionModel(config=config.vision_config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)
        self.multi_modal_projector = Gemma3MultiModalProjector(config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)
        self.input_processor = Gemma3VLInputProcessor(self.config, dtype=dtype)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.language_model.get_logits(hidden_states)

    def get_image_features(self, pixel_values: torch.Tensor):
        """Projects the last hidden state from the vision model into language
        model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_tower(pixel_values=pixel_values)
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        pixel_values: torch.FloatTensor = None,
        image_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        vision_embedding_indexing: torch.Tensor = None,
        text_embedding_indexing: torch.Tensor = None,
        **kwargs,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to
                 `-100` are ignored (masked), the loss is only computed for the tokens with labels in
                 `[0, ..., config.text_config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only
                for that token can save memory, which becomes pretty significant for long sequences or large vocabulary
                size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length
                dimension. This is useful when using packed tensor format (single dimension for batch and
                sequence length).
        """

        if inputs_embeds is None and pixel_values is not None:
            # extract feature
            vit_embeds = self.get_image_features(pixel_values)
            lang_embeds = self.get_input_embeddings()(input_ids)
            lang_embeds.masked_scatter_(image_mask[..., None], vit_embeds)

            inputs_embeds = lang_embeds
        if pixel_values is not None:
            kwargs = self.prepare_attn_masks(input_ids[0], position_ids[0], mask_dtype=pixel_values.dtype, **kwargs)

        hidden_states = self.language_model(input_ids,
                                            position_ids,
                                            inputs_embeds=inputs_embeds,
                                            past_key_values=past_key_values,
                                            attn_metadata=attn_metadata,
                                            **kwargs)

        return hidden_states

    # modified from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma3_mm.py#L539
    def prepare_attn_masks(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask_dtype: torch.dtype,
        **kwargs,
    ):
        kwargs['has_images'] = True
        start_idices = (positions == 0).cpu().nonzero()
        num_seqs = len(start_idices)
        seq_lens = []
        for i in range(num_seqs):
            start_idx = start_idices[i].item()
            if i < num_seqs - 1:
                end_idx = start_idices[i + 1].item()
            else:
                end_idx = len(input_ids)
            seq_lens.append(end_idx - start_idx)
        kwargs['seq_lens'] = seq_lens

        global_attn_masks = []
        local_attn_masks = []
        start_idx = 0
        for seq_len in seq_lens:
            end_idx = start_idx + seq_len
            input_token_ids = input_ids[start_idx:end_idx]
            start_idx = end_idx
            # Create a global causal mask.
            global_attn_mask = torch.empty(
                1,
                1,
                seq_len,
                seq_len,
                dtype=mask_dtype,
                device=input_ids.device,
            )
            global_attn_mask.fill_(float('-inf'))
            # Fill the lower triangle with 0.
            global_attn_mask = global_attn_mask.triu(diagonal=1)

            # Consider the bidirectional attention between image tokens.
            img_mask = torch.zeros_like(global_attn_mask)
            img_pos = (input_token_ids == self.config.image_token_index)
            img_mask[:, :, :, img_pos] += 1
            img_mask[:, :, img_pos, :] += 1
            global_attn_mask = torch.where(img_mask == 2, 0, global_attn_mask)
            global_attn_masks.append(global_attn_mask)

            # Create a local causal mask with sliding window (1024).
            local_attn_mask = torch.ones_like(global_attn_mask)
            local_attn_mask = torch.tril(local_attn_mask, diagonal=-self.sliding_window)
            local_attn_mask = torch.where(local_attn_mask == 0, global_attn_mask, float('-inf'))
            local_attn_masks.append(local_attn_mask)
        kwargs['global_attn_masks'] = global_attn_masks
        kwargs['local_attn_masks'] = local_attn_masks
        return kwargs

    def prepare_inputs_for_generation(
        self,
        past_key_values=None,
        inputs_embeds=None,
        context: StepContext = None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = self.language_model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
            **kwargs,
        )

        # vision inputs
        pixel_values = None
        image_mask = None
        if context.input_multimodals is not None:
            pixel_values = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            # flatten batch
            pixel_values = [data for im_data in pixel_values for data in im_data]
            if len(pixel_values) > 0:
                image_token_id = pixel_values[0].meta['image_token_id']
                image_mask = model_inputs['input_ids'] == image_token_id
                pixel_values = torch.cat([data.data for data in pixel_values])
            else:
                pixel_values = None
                image_mask = None
        model_inputs['image_mask'] = image_mask
        model_inputs['pixel_values'] = pixel_values
        return model_inputs

    def tie_weights(self):
        return self.language_model.tie_weights()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
        ]

        lang_prefix = 'language_model.'
        lang_prefix_length = len(lang_prefix)
        new_weights = dict()
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name.startswith(lang_prefix):
                new_key = name[lang_prefix_length:]
                new_weights[new_key] = loaded_weight
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)

        self.language_model.load_weights(new_weights.items())

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor

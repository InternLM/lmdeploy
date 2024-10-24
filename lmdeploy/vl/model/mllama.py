# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL.Image import Image
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.mllama.modeling_mllama import MllamaPreTrainedModel

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging


class MllamaVisionModelPatch(MllamaPreTrainedModel):

    def apply_class_embedding(self,
                              hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1,
                                                      hidden_size)
        class_embedding = class_embedding.to(hidden_state.device)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state

    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        aspect_ratio_mask: torch.Tensor,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # noqa
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # noqa

        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape  # noqa

        pixel_values = pixel_values.reshape(
            batch_size * num_concurrent_media * num_tiles, num_channels,
            height, width)
        aspect_ratio_ids = aspect_ratio_ids.reshape(
            batch_size * num_concurrent_media, -1)

        # Patch embedding
        patch_embeds = self.patch_embedding(
            pixel_values.to(self.dtype).to(self.device))
        hidden_state = patch_embeds.flatten(2).transpose(1, 2)

        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media,
                                            num_tiles, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(
            hidden_state, aspect_ratio_ids)

        # Add cls token
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media,
                                            num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state,
                                                       aspect_ratio_ids)

        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (
            0, 0, 0, num_padding_patches
        )  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode='constant', value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(
            batch_size * num_concurrent_media, -1)
        from transformers.models.mllama.modeling_mllama import \
            _prepare_aspect_ratio_attention_mask
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.dtype,
        )

        # Apply encoder
        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1,
                                         dim)
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        hidden_state = output[0]

        hidden_state = self.layernorm_post(hidden_state)

        # Apply global encoder
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media,
                                            num_tiles,
                                            num_patches + num_padding_patches,
                                            dim)
        hidden_state = self.post_tile_positional_embedding(
            hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles * (num_patches + num_padding_patches), dim)
        global_output = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_state = global_output[0]

        # Remove padding form hidden state
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media,
                                            num_tiles,
                                            num_patches + num_padding_patches,
                                            dim)
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media,
                                            num_tiles, num_patches, dim)

        # Collect intermediate layer outputs from encoder output
        all_intermediate_hidden_states = output[1]
        # rewrite to sync device during accelerate pipeline parallel
        device = hidden_state.device
        all_intermediate_hidden_states = [
            s.to(device) for s in all_intermediate_hidden_states
        ]
        intermediate_hidden_states = torch.stack(
            all_intermediate_hidden_states, dim=-1)
        intermediate_hidden_states = intermediate_hidden_states[
            ..., self.intermediate_layers_indices]

        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles,
            num_patches + num_padding_patches, -1)
        intermediate_hidden_states = intermediate_hidden_states[:, :, :
                                                                slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1)

        # Concatenate final hidden state and intermediate hidden states
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states],
                                 dim=-1)

        if output_hidden_states:
            hidden_states = tuple(all_intermediate_hidden_states) + tuple(
                global_output[1])
        else:
            hidden_states = None

        if output_attentions:
            # global transformer in contrast to `self.transformer` doesn't
            # always return hidden states so we might go index out-of-range
            global_attn = tuple(
                global_output[2]) if output_hidden_states else tuple(
                    global_output[1])
            attentions = tuple(output[2]) + global_attn
        else:
            attentions = None

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states, attentions]
                         if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )


def check_transformers():
    """check qwen_vl_utils."""
    try:
        from transformers import MllamaForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError(
            'please install latest transformers by '
            'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class MllamaVLModel(VisonModel):
    """llama3.2 model."""

    _arch = 'MllamaForConditionalGeneration'

    def build_model(self):
        check_transformers()

        from transformers.models.mllama.modeling_mllama import \
            MllamaVisionModel
        MllamaVisionModel.forward = MllamaVisionModelPatch.forward
        MllamaVisionModel.apply_class_embedding = MllamaVisionModelPatch.apply_class_embedding  # noqa
        from accelerate import init_empty_weights
        with init_empty_weights():
            config = self.hf_config
            config.quantization_config = {}  # disable vision part quantization
            # disable accelerate check_tied_parameters_in_config
            config.tie_word_embeddings = False
            from transformers import MllamaForConditionalGeneration
            model = MllamaForConditionalGeneration._from_config(config)
            if not self.with_llm:
                del model.language_model
            else:
                self.vl_model = model

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                max_memory=self.max_memory,
                no_split_module_classes=[
                    'MllamaPrecomputedPositionEmbedding',
                    'MllamaPrecomputedAspectRatioEmbedding',
                    'MllamaVisionEncoderLayer'
                ],
                dtype=config.torch_dtype)

        self.model = model.eval()

        # processor
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.image_token_id = 128256

    @torch.no_grad()
    def forward(self,
                images: List[Image],
                params: List[Dict] = None) -> List[torch.Tensor]:
        """forward."""
        # only support image input
        if params is not None:
            assert len(images) == len(
                params), 'different length of images and params'
        else:
            params = [{}] * len(images)
        # resize images with abnormal shape
        # TODO try catch image feature extraction in pipeline and
        # throw error back to users
        for i, image in enumerate(images):
            size = image.size
            if any([s < 3 for s in size]):
                images[i] = image.resize([s * 3 for s in size])
        image_inputs = self.processor.image_processor(images=images,
                                                      return_tensors='pt')
        pixel_values = image_inputs['pixel_values'].to(
            self.model.vision_model.device)
        pixel_values = pixel_values.type(self.model.vision_model.dtype)
        aspect_ratio_ids = image_inputs['aspect_ratio_ids'].to(
            self.model.vision_model.device)
        aspect_ratio_mask = image_inputs['aspect_ratio_mask'].to(
            self.model.vision_model.device)
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True)
        cross_attention_states = vision_outputs[0]
        cross_attention_states = self.model.multi_modal_projector(
            cross_attention_states)
        _, bsz, _, _, image_token_dim = tuple(cross_attention_states.shape)
        cross_attention_states = cross_attention_states.view(
            bsz, -1, image_token_dim).split([1] * len(images))
        return cross_attention_states

# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from typing import List

import torch
from PIL.Image import Image
from transformers import AutoProcessor

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging


@VISION_MODELS.register_module()
class LlavaNextVisionModel(VisonModel):
    """Llava hf vision model."""

    _arch = 'LlavaNextForConditionalGeneration'

    def build_model(self):
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from accelerate.utils import get_balanced_memory, infer_auto_device_map

        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from transformers import LlavaNextForConditionalGeneration
            model = LlavaNextForConditionalGeneration._from_config(
                self.hf_config)
            if not self.with_llm:
                del model.language_model
                for key in ['language_model']:
                    setattr(model, key, None)
            else:
                self.vl_model = model

        no_split_module_classes = ['CLIPEncoderLayer']
        max_memory = get_balanced_memory(
            model,
            max_memory=self.max_memory,
            dtype=torch.half,
            no_split_module_classes=no_split_module_classes)
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=no_split_module_classes,
            max_memory=max_memory,
            dtype=torch.half)

        same_device_keys = [('multi_modal_projector', 'image_newline')]
        for keys in same_device_keys:
            keys = [k for k in keys if k in device_map]
            if len(keys) <= 1:
                continue
            for k in keys[1:]:
                device_map[k] = device_map[keys[0]]

        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map=device_map if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=no_split_module_classes,
                dtype=torch.half)
        model.eval()
        self.model = model
        # processor
        processor = AutoProcessor.from_pretrained(self.model_path,
                                                  trust_remote_code=True)
        if hasattr(processor, 'tokenizer'):
            del processor.tokenizer
            processor.prtokenizer = None
        self.processor = processor.image_processor

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        from transformers.models.llava_next.modeling_llava_next import \
            image_size_to_num_patches
        """forward."""
        processed_inputs = self.processor(images,
                                          return_tensors='pt',
                                          input_data_format='channels_last')
        pixel_values = processed_inputs['pixel_values'].to(
            device=self.model.device, dtype=self.model.dtype)
        image_sizes = processed_inputs['image_sizes'].to(
            device=self.model.device, dtype=self.model.dtype)
        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.hf_config.image_grid_pinpoints,
                patch_size=self.hf_config.vision_config.image_size,
            ) for imsize in image_sizes
        ]
        # figure out if pixel_values is concatenated or stacked
        if pixel_values.dim() == 5:
            # stacking when input is
            # (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [
                pix_val[:num_patch]
                for pix_val, num_patch in zip(pixel_values, image_num_patches)
            ]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of
            # (num_patches, num_channels, height, width)
            raise ValueError(f'pixel_values of shape {pixel_values.shape}, '
                             'expect to be of 4 or 5 dimensions')
        image_outputs = self.model.vision_tower.forward(
            pixel_values, output_hidden_states=True)
        image_features = image_outputs.hidden_states[
            self.hf_config.vision_feature_layer]
        if self.hf_config.vision_feature_select_strategy == 'default':
            image_features = image_features[:, 1:]
        elif self.hf_config.vision_feature_select_strategy == 'full':
            image_features = image_features
        else:
            raise ValueError(
                'Unexpected select feature strategy: '
                f'{self.hf_config.vision_feature_select_strategy}')
        image_features = self.model.multi_modal_projector(image_features)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        image_features, feature_lens = self.model.pack_image_features(
            image_features,
            image_sizes,
            image_newline=self.model.image_newline,
        )
        outputs = torch.split(image_features,
                              feature_lens.cpu().numpy().tolist(),
                              dim=0)
        return outputs

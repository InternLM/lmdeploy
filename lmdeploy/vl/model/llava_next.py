# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from typing import List

import torch
from PIL.Image import Image
from transformers import AutoProcessor

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import (disable_logging,
                                     get_vision_encoder_device_map)


@VISION_MODELS.register_module()
class LlavaNextVisionModel(VisonModel):
    """Llava hf vision model."""

    _arch = 'LlavaNextForConditionalGeneration'

    def build_model(self):
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

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
        same_device_keys = [('multi_modal_projector', 'image_newline')]
        device_map = get_vision_encoder_device_map(model, self.max_memory,
                                                   no_split_module_classes,
                                                   same_device_keys)
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
        """forward."""
        processed_inputs = self.processor(images,
                                          return_tensors='pt',
                                          input_data_format='channels_last')
        pixel_values = processed_inputs['pixel_values'].to(
            device=self.model.device, dtype=self.model.dtype)
        image_sizes = processed_inputs['image_sizes'].to(
            device=self.model.device, dtype=self.model.dtype)
        image_features = self.model.get_image_features(
            pixel_values,
            image_sizes,
            vision_feature_layer=self.hf_config.vision_feature_layer,
            vision_feature_select_strategy=self.hf_config.
            vision_feature_select_strategy)
        image_features, feature_lens = self.model.pack_image_features(
            image_features,
            image_sizes,
            self.hf_config.vision_feature_select_strategy,
            image_newline=self.model.image_newline,
        )
        outputs = torch.split(image_features,
                              feature_lens.cpu().numpy().tolist(),
                              dim=0)
        return outputs

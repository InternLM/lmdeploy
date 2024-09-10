# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from typing import List

import torch
from PIL.Image import Image
from transformers import AutoProcessor

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging


@VISION_MODELS.register_module()
class LlavaHfVisionModel(VisonModel):
    """Llava hf vision model."""

    _arch = 'LlavaForConditionalGeneration'

    def build_model(self):
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from transformers import LlavaForConditionalGeneration
            model = LlavaForConditionalGeneration._from_config(self.hf_config)
            if not self.with_llm:
                del model.language_model
                for key in ['language_model']:
                    setattr(model, key, None)
            else:
                self.vl_model = model

        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                max_memory=self.max_memory,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=['CLIPEncoderLayer'],
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
        pixel_values = self.processor(
            images, return_tensors='pt',
            input_data_format='channels_last')['pixel_values']
        pixel_values = pixel_values.to(device=self.model.device,
                                       dtype=self.model.dtype)
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
        outputs = torch.split(image_features, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs

# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import warnings
from typing import Dict, List

import torch

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.llava_hf import VISION_MODELS, LlavaHfVisionModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class LlavaNextVisionModel(LlavaHfVisionModel):
    """Llava hf vision model."""

    _arch = 'LlavaNextForConditionalGeneration'

    def build_preprocessor(self):
        super().build_preprocessor()
        # build the model with empty weights. The model will be used in
        # `preprocess` to get the image token number
        from accelerate import init_empty_weights
        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from transformers import LlavaNextForConditionalGeneration
            self.model = LlavaNextForConditionalGeneration._from_config(
                self.hf_config)
            self.vl_model = self.model
            if not self.with_llm:
                del self.model.language_model

    def build_model(self):
        """build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        from accelerate import load_checkpoint_and_dispatch

        no_split_module_classes = ['CLIPEncoderLayer']
        same_device_keys = [('multi_modal_projector', 'image_newline')]
        device_map = self.get_vision_encoder_device_map(
            self.model, self.max_memory, no_split_module_classes,
            same_device_keys)

        with disable_logging():
            load_checkpoint_and_dispatch(
                model=self.model,
                checkpoint=self.model_path,
                device_map=device_map if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=no_split_module_classes,
                dtype=torch.half)
        self.model.eval()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refers to the spec of `super.preprocess()"""
        from transformers.models.llava_next.modeling_llava_next import \
            image_size_to_num_patches
        images = self.collect_images(messages)
        outputs = []
        for image, params in images:
            image = image.convert('RGB')
            result = self.processor(image,
                                    return_tensors='pt',
                                    input_data_format='channels_last')
            # ! infer image_num_patches from image_sizes
            image_num_patches = [
                image_size_to_num_patches(
                    image_size=imsize,
                    grid_pinpoints=self.hf_config.image_grid_pinpoints,
                    patch_size=self.hf_config.vision_config.image_size,
                ) for imsize in result['image_sizes']
            ]

            hidden_size = self.hf_config.text_config.hidden_size
            fake_image_features = torch.zeros(
                [image_num_patches[0], self.n_token_per_image, hidden_size])
            image_sizes = result['image_sizes']
            image_newline = torch.randn(self.hf_config.text_config.hidden_size)
            strategy = self.hf_config.vision_feature_select_strategy
            _, image_tokens = self.model.pack_image_features(
                [fake_image_features],
                image_sizes,
                vision_feature_select_strategy=strategy,
                image_newline=image_newline)
            result.update(
                dict(image_size=image.size,
                     image_patches=image_num_patches,
                     image_tokens=image_tokens,
                     image_token_id=0))
            outputs.append(result)
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @torch.no_grad()
    def forward(self,
                messages: List[Dict],
                max_batch_size: int = 1) -> List[Dict]:
        """extract image feature. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(List[Dict]): the outputs of `preprocess`
            max_batch_size(int): the max batch size when forwarding vision
                model
        Return:
            the message list with forwarding results included
        """
        inputs = [x['content'] for x in messages if x['role'] == 'preprocess']
        inputs = inputs[0]
        outputs = []
        for idx in range(0, len(inputs), max_batch_size):
            pixel_values = [
                x['pixel_values'].to(device=self.model.device,
                                     dtype=self.model.dtype)
                for x in inputs[idx:idx + max_batch_size]
            ]
            pixel_values = torch.cat(pixel_values, dim=0)
            image_sizes = [
                x['image_sizes'].to(device=self.model.device,
                                    dtype=self.model.dtype)
                for x in inputs[idx:idx + max_batch_size]
            ]
            image_sizes = torch.cat(image_sizes, dim=0)
            image_num_patches = [
                x['num_patch'] for x in inputs[idx:idx + max_batch_size]
            ]
            image_num_patches = list(itertools.chain(*image_num_patches))
            # figure out if pixel_values is concatenated or stacked
            if pixel_values.dim() == 5:
                # stacking when input is
                # (batch_size, num_patches, num_channels, height, width)
                _pixel_values_list = [
                    pix_val[:num_patch] for pix_val, num_patch in zip(
                        pixel_values, image_num_patches)
                ]
                pixel_values = torch.cat(_pixel_values_list, dim=0)
            elif pixel_values.dim() != 4:
                # otherwise has to be stacked from list of
                # (num_patches, num_channels, height, width)
                raise ValueError(
                    f'pixel_values of shape {pixel_values.shape}, '
                    'expect to be of 4 or 5 dimensions')
            logger.info(f'vision forward shape: {pixel_values.shape}')
            image_outputs = self.model.vision_tower.forward(
                pixel_values, output_hidden_states=True)
            image_features = image_outputs.hidden_states[
                self.hf_config.vision_feature_layer]
            strategy = self.hf_config.vision_feature_select_strategy
            if strategy == 'default':
                image_features = image_features[:, 1:]
            elif strategy == 'full':
                image_features = image_features
            else:
                raise ValueError('Unexpected select feature strategy: '
                                 f'{strategy}')
            image_features = self.model.multi_modal_projector(image_features)
            image_features = torch.split(image_features,
                                         image_num_patches,
                                         dim=0)
            image_features, feature_lens = self.model.pack_image_features(
                image_features,
                image_sizes,
                vision_feature_select_strategy=strategy,
                image_newline=self.model.image_newline,
            )
            image_features = torch.split(image_features,
                                         feature_lens.cpu().numpy().tolist(),
                                         dim=0)
            outputs.extend(image_features)
        messages.append(dict(role='forward', content=outputs))
        return messages

# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import warnings
from typing import Dict, List

import torch
from transformers import AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class LlavaNextVisionModel(VisonModel):
    """Llava hf vision model."""

    _arch = 'LlavaNextForConditionalGeneration'

    def build_preprocessor(self):
        processor = AutoProcessor.from_pretrained(self.model_path,
                                                  trust_remote_code=True)
        if hasattr(processor, 'tokenizer'):
            del processor.tokenizer
            processor.prtokenizer = None
        self.processor = processor.image_processor
        # build the model with empty weights. The model will be used in
        # `preprocess` to get the image token number
        from accelerate import init_empty_weights
        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from transformers import LlavaNextForConditionalGeneration
            self.model = LlavaNextForConditionalGeneration._from_config(
                self.hf_config)

    def build_model(self):
        from accelerate import load_checkpoint_and_dispatch
        from accelerate.utils import get_balanced_memory, infer_auto_device_map

        if not self.with_llm:
            del self.model.language_model
            for key in ['language_model']:
                setattr(self.model, key, None)
        else:
            self.vl_model = self.model

        no_split_module_classes = ['CLIPEncoderLayer']
        max_memory = get_balanced_memory(
            self.model,
            max_memory=self.max_memory,
            dtype=torch.half,
            no_split_module_classes=no_split_module_classes)
        device_map = infer_auto_device_map(
            self.model,
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
        outputs = []
        for item in messages[-1]['content']:
            item_type = item['type']
            if item_type == 'image':
                image = item['image'].convert('RGB')
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
                # TODO(remove hardcode 576)
                hidden_size = self.hf_config.text_config.hidden_size
                fake_image_features = torch.zeros(
                    [image_num_patches[0], 576, hidden_size])
                image_sizes = result['image_sizes']
                image_newline = torch.randn(
                    self.hf_config.text_config.hidden_size)
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
        return outputs

    @torch.no_grad()
    def forward(self, inputs: List[Dict]) -> List[torch.Tensor]:
        # from transformers.models.llava_next.modeling_llava_next import \
        #     image_size_to_num_patches

        pixel_values = [
            x['pixel_values'].to(device=self.model.device,
                                 dtype=self.model.dtype) for x in inputs
        ]
        pixel_values = torch.cat(pixel_values, dim=0)
        image_sizes = [
            x['image_sizes'].to(device=self.model.device,
                                dtype=self.model.dtype) for x in inputs
        ]
        image_sizes = torch.cat(image_sizes, dim=0)
        image_num_patches = [x['num_patch'] for x in inputs]
        image_num_patches = list(itertools.chain(*image_num_patches))
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
        strategy = self.hf_config.vision_feature_select_strategy
        if strategy == 'default':
            image_features = image_features[:, 1:]
        elif strategy == 'full':
            image_features = image_features
        else:
            raise ValueError('Unexpected select feature strategy: '
                             f'{strategy}')
        image_features = self.model.multi_modal_projector(image_features)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        image_features, feature_lens = self.model.pack_image_features(
            image_features,
            image_sizes,
            vision_feature_select_strategy=strategy,
            image_newline=self.model.image_newline,
        )
        outputs = torch.split(image_features,
                              feature_lens.cpu().numpy().tolist(),
                              dim=0)
        return outputs

    @classmethod
    def proc_messages(cls, messages, chat_template, sequence_start):
        """apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            n_images = [
                1 for item in message['content'] if item['type'] == 'image'
            ]
            n_images = sum(n_images)
            content = [
                item['text'] for item in message['content']
                if item['type'] == 'text'
            ]
            content = f'<img>{IMAGE_TOKEN * n_images}</img>\n' + content[0]
            prompt_messages.append(dict(role='user', content=content))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template,
                                                 sequence_start)
        return super().to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer,
                                      sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template,
                                                 sequence_start)
        return super().to_turbomind_aux(messages, prompt, IMAGE_TOKEN,
                                        tokenizer, sequence_start)

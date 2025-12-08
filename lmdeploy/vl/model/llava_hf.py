# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List

import torch
from transformers import AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class LlavaHfVisionModel(VisionModel):
    """Llava hf vision model."""

    _arch = 'LlavaForConditionalGeneration'

    def build_preprocessor(self):
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        if hasattr(processor, 'tokenizer'):
            del processor.tokenizer
            processor.prtokenizer = None
        self.processor = processor.image_processor
        image_size = self.hf_config.vision_config.image_size
        patch_size = self.hf_config.vision_config.patch_size
        self.n_token_per_image = (image_size // patch_size)**2
        if self.hf_config.vision_feature_select_strategy == 'full':
            self.n_token_per_image += 1

    def build_model(self):
        """Build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from transformers import LlavaForConditionalGeneration
            model = LlavaForConditionalGeneration._from_config(self.hf_config)
            self.vl_model = model
            if not self.with_llm:
                del model.language_model

        # fix for llava-hf/llava-interleave-qwen-7b-hf
        setattr(model.config, 'tie_word_embeddings', False)
        with disable_logging():
            load_checkpoint_and_dispatch(model=model,
                                         max_memory=self.max_memory,
                                         checkpoint=self.model_path,
                                         device_map='auto' if not self.with_llm else {'': 'cpu'},
                                         no_split_module_classes=['CLIPEncoderLayer', 'SiglipEncoderLayer'],
                                         dtype=torch.half)
        model.eval()
        self.model = model

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refers to `super.preprocess() for spec."""
        images = self.collect_images(messages)
        outputs = []
        for image, params in images:
            image = image.convert('RGB')
            pixel_values = self.processor(image, return_tensors='pt', input_data_format='channels_last').pixel_values
            outputs.append(
                dict(pixel_values=pixel_values,
                     image_size=image.size,
                     image_tokens=self.n_token_per_image,
                     image_token_id=self.image_token_id))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        """Extract image feature. ONLY implement it when the backend is
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
            pixel_values = [x['pixel_values'] for x in inputs[idx:idx + max_batch_size]]
            pixel_values = torch.cat(pixel_values, dim=0)
            pixel_values = pixel_values.to(device=self.model.device, dtype=self.model.dtype)
            logger.info(f'vision forward shape: {pixel_values.shape}')
            image_outputs = self.model.vision_tower.forward(pixel_values, output_hidden_states=True)
            image_features = image_outputs.hidden_states[self.hf_config.vision_feature_layer]
            if self.hf_config.vision_feature_select_strategy == 'default':
                image_features = image_features[:, 1:]
            elif self.hf_config.vision_feature_select_strategy == 'full':
                image_features = image_features
            else:
                raise ValueError('Unexpected select feature strategy: '
                                 f'{self.hf_config.vision_feature_select_strategy}')
            image_features = self.model.multi_modal_projector(image_features)
            image_features = torch.split(image_features, 1, dim=0)
            outputs.extend([x.squeeze() for x in image_features])
        messages.append(dict(role='forward', content=outputs))
        return messages

    @staticmethod
    def proc_messages(messages, chat_template, sequence_start):
        """Apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
                continue
            n_images = len([1 for x in message['content'] if x['type'] == 'image'])
            content = [item['text'] for item in message['content'] if item['type'] == 'text']
            prompt = (IMAGE_TOKEN + '\n') * n_images + content[0]
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

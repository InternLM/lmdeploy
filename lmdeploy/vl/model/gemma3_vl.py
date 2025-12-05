# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
from transformers import AutoConfig, AutoProcessor
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel

logger = get_logger('lmdeploy')


class Gemma3ImagesKwargs(ImagesKwargs):
    do_pan_and_scan: Optional[bool]
    pan_and_scan_min_crop_size: Optional[int]
    pan_and_scan_max_num_crops: Optional[int]
    pan_and_scan_min_ratio_to_activate: Optional[float]
    do_convert_rgb: Optional[bool]


class Gemma3ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Gemma3ImagesKwargs
    _defaults = {
        'text_kwargs': {
            'padding': False,
        },
        'images_kwargs': {
            'do_pan_and_scan': False,
            'pan_and_scan_min_crop_size': 256,
            'pan_and_scan_max_num_crops': 4,
            'pan_and_scan_min_ratio_to_activate': 1.2,
        },
    }


@VISION_MODELS.register_module()
class Gemma3VisionModel(VisionModel):
    """Gemma3 vision model."""

    _arch = 'Gemma3ForConditionalGeneration'

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: Dict[int, int] = None,
                 hf_config: AutoConfig = None,
                 backend: str = ''):
        super().__init__(model_path, with_llm, max_memory, hf_config, backend)

    def build_preprocessor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = self.processor.tokenizer
        self.image_token_id = tokenizer.encode(tokenizer.image_token)[-1]
        self.image_tokens = self.processor.image_seq_length
        self.tokenizer_init_kwargs = tokenizer.init_kwargs

    def build_model(self):
        """Build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        # TODO, implement for tubomind engine
        raise NotImplementedError()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refers to `super.preprocess() for spec."""
        from transformers.image_utils import make_nested_list_of_images
        output_kwargs = self.processor._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer_init_kwargs,
            **{
                'return_tensors': 'pt',
                'add_special_tokens': False
            },
        )
        images = self.collect_images(messages)
        images = [image.convert('RGB') for image, _ in images]
        num_image = len(images)
        images = make_nested_list_of_images(images)
        image_inputs = self.processor.image_processor(images, **output_kwargs['images_kwargs'])
        outputs = []
        for idx in range(num_image):
            pixel_values = image_inputs['pixel_values'][idx:idx + 1, ...]
            num_crops = image_inputs['num_crops'][:idx:idx + 1]
            data = dict(pixel_values=pixel_values,
                        num_crops=num_crops,
                        image_tokens=self.image_tokens,
                        image_token_id=self.image_token_id)
            outputs.append(data)

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
        # TODO, implement for turbomind engine
        raise NotImplementedError()

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
            prompt = ('\n\n' + IMAGE_TOKEN + '\n\n') * n_images + content[0]
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

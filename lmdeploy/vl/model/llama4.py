# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from transformers import AutoConfig

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel

logger = get_logger('lmdeploy')


def check_trans_version():
    """Check if the installed version of the 'transformers' library is smaller
    than the specified version."""
    import transformers
    from packaging import version

    min_version = '4.51.0'
    installed_version = transformers.__version__
    assert version.parse(installed_version) >= version.parse(min_version), (
        f'llama4 requires transformers version >= {min_version}, '
        f'but found version: {installed_version}. Please upgrade.')


@VISION_MODELS.register_module()
class LLama4VisionModel(VisionModel):
    """Llama4 vision model."""

    _arch = 'Llama4ForConditionalGeneration'

    @classmethod
    def match(cls, config: AutoConfig):
        """Check whether the config match the model."""
        arch = config.architectures[0]
        return arch == cls._arch

    def build_preprocessor(self):
        check_trans_version()
        from transformers.models.llama4 import Llama4Processor
        from transformers.models.llama4.processing_llama4 import Llama4ProcessorKwargs
        self.processor = Llama4Processor.from_pretrained(
            self.model_path,
            padding_side='left',
        )
        img_patch_token = self.processor.img_patch_token
        self.image_token_id = self.processor.tokenizer.encode(img_patch_token, add_special_tokens=False)[0]
        self.images_kwargs = self.processor._merge_kwargs(
            Llama4ProcessorKwargs,
            tokenizer_init_kwargs=self.processor.tokenizer.init_kwargs,
            return_tensors='pt',
            add_special_tokens=False,
        )['images_kwargs']

    def build_model(self):
        """Build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        # TODO, implement for tubomind engine
        raise NotImplementedError()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refers to `super.preprocess() for spec."""
        images = self.collect_images(messages)
        outputs = []
        processor = self.processor
        patch_size = processor.patch_size
        downsample_ratio = processor.downsample_ratio
        images_kwargs = self.images_kwargs
        for image, params in images:
            image_inputs = processor.image_processor(images=[image], **images_kwargs)
            pixel_values = image_inputs['pixel_values']
            image_height, image_width = image_inputs['pixel_values'][0].shape[-2:]
            num_patches_per_chunk = int((image_height // patch_size) * (image_width // patch_size) // downsample_ratio)
            aspect_ratios = image_inputs.pop('aspect_ratios')
            image_prompts = processor._prompt_split_image(aspect_ratios[0], num_patches_per_chunk)
            image_tokens = image_prompts.count('<|') - 2
            outputs.append(
                dict(pixel_values=pixel_values,
                     image_tokens=image_tokens,
                     image_token_id=self.image_token_id,
                     image_size=image.size,
                     image_prompts=image_prompts))
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
            elif message['role'] in ['preprocess', 'forward']:
                continue
            n_images = len([1 for x in message['content'] if x['type'] == 'image'])
            content = [x.get('text', '') for x in message['content'] if x['type'] == 'text']
            prompt = content[0]
            if IMAGE_TOKEN not in prompt:
                prompt = f'{IMAGE_TOKEN * n_images}' + prompt
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch_aux(self, messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start):
        """Auxiliary function to pack the preprocessing results in a format
        compatible with what is required by pytorch engine.

        Args:
            messages(List[Dict]): the output of `preprocess`
            prompt(str): the prompt after applying chat template
            IMAGE_TOKEN(str): a placeholder where image tokens will be
                inserted
            tokenzer: the tokenizer model
            sequence_start: starting flag of a sequence
        """
        # collect all preprocessing result from messages
        preps = [x['content'] for x in messages if x['role'] == 'preprocess']
        assert len(preps) == 1
        preps = preps[0]

        # split prompt into segments and validate data
        segs = prompt.split(IMAGE_TOKEN)
        assert len(segs) == len(preps) + 1, (f'the number of {IMAGE_TOKEN} is not equal '
                                             f'to input images, {len(segs) - 1} vs {len(preps)}')

        # calculate the image token offset for each image
        input_ids = []
        for i, seg in enumerate(segs):
            if i > 0 and i <= len(preps):
                prep = preps[i - 1]
                image_prompts = prep.pop('image_prompts', '')
                prep.update(offset=len(input_ids) + 1)
                assert self.image_token_id == prep['image_token_id']
                seg = image_prompts + seg
            token_ids = tokenizer.encode(seg, add_bos=((i == 0) and sequence_start))
            input_ids.extend(token_ids)
        return dict(prompt=prompt, input_ids=input_ids, multimodal=preps)

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

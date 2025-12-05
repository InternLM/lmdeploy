# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel
from lmdeploy.vl.model.utils import disable_logging

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class QwenVisionModel(VisionModel):
    """Qwen vision model."""

    _arch = 'QWenLMHeadModel'

    def build_preprocessor(self):
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        image_size = self.hf_config.visual['image_size']
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def build_model(self):
        """Build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        from accelerate import init_empty_weights
        with init_empty_weights():
            config = self.hf_config
            config.quantization_config = {}  # disable vision part quantization
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            self.vl_model = model
            if not self.with_llm:
                del model.lm_head
                for key in ['wte', 'h', 'ln_f']:
                    setattr(model.transformer, key, None)

        from accelerate.utils import get_balanced_memory, infer_auto_device_map
        max_memory = get_balanced_memory(model,
                                         max_memory=self.max_memory,
                                         dtype=torch.half,
                                         no_split_module_classes=['VisualAttentionBlock', 'Resampler'])
        device_map = infer_auto_device_map(model,
                                           no_split_module_classes=['VisualAttentionBlock', 'Resampler'],
                                           max_memory=max_memory,
                                           dtype=torch.half)
        same_device_keys = [('transformer.visual.conv1', 'transformer.visual.positional_embedding'),
                            ('transformer.visual.ln_post', 'transformer.visual.proj')]
        for (a, b) in same_device_keys:
            if a in device_map and b in device_map:
                device_map[b] = device_map[a]

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(model=model,
                                         checkpoint=self.model_path,
                                         device_map=device_map if not self.with_llm else {'': 'cpu'},
                                         no_split_module_classes=['VisualAttentionBlock'],
                                         dtype=torch.half)

        self.model = model.transformer.visual.eval()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refers to `super.preprocess() for spec."""
        images = self.collect_images(messages)
        outputs = []
        for image, params in images:
            image = image.convert('RGB')
            pixel_values = self.image_transform(image)
            outputs.append(
                dict(pixel_values=pixel_values,
                     image_size=image.size,
                     image_tokens=256,
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
            pixel_values = torch.stack(pixel_values, dim=0)
            logger.info(f'vision forward shape: {pixel_values.shape}')
            feats = self.model(pixel_values)
            feats = torch.split(feats, 1, dim=0)
            outputs.extend([x.squeeze() for x in feats])
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
            content = [x.get('text', '') for x in message['content'] if x['type'] == 'text']
            prompt = content[0]
            if IMAGE_TOKEN in prompt:
                pass
            else:
                prompt = ''.join([f'Picture {str(i)}:{IMAGE_TOKEN}\n' for i in range(n_images)]) + prompt
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

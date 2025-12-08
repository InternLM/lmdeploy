# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class CogVLMVisionModel(VisionModel):
    """CogVLM vision model."""

    _arch = 'CogVLMForCausalLM'

    def build_preprocessor(self):
        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((self.hf_config.vision_config['image_size'], ) * 2,
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        image_size = self.hf_config.vision_config['image_size']
        patch_size = self.hf_config.vision_config['patch_size']
        if self.hf_config.vision_config['num_positions'] == 1226:
            # cogvlm-chat-hf, https://huggingface.co/THUDM/cogvlm-chat-hf/blob/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c/modeling_cogvlm.py#L820 # noqa E501
            self.n_token_per_image = 2 + (image_size // patch_size)**2
        else:
            # cogvlm2, https://huggingface.co/THUDM/cogvlm2-llama3-chinese-chat-19B/blob/2c2226281325649d49b8aa237a932367c7da4f26/modeling_cogvlm.py#L819 # noqa E501
            self.n_token_per_image = 2 + (image_size // patch_size // 2)**2

    def build_model(self):
        if self.with_llm:
            from transformers import AutoModelForCausalLM
            self.vl_model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                                 device_map='cpu',
                                                                 trust_remote_code=True)
        else:
            raise NotImplementedError('turbomind has not supported cogvlm yet')

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refer to the spec of `super().preprocess`"""
        images = self.collect_images(messages)
        outputs = []
        for image, _ in images:
            image = image.convert('RGB')
            pixel_values = self.image_transform(image)
            outputs.append(
                dict(pixel_values=pixel_values,
                     image_size=image.size,
                     image_tokens=self.n_token_per_image,
                     image_token_id=self.image_token_id))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @staticmethod
    def proc_messages(messages, chat_template, sequence_start):
        """Apply chat template to get the prompt."""
        prompt_messages = []
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
                continue
            content = [x.get('text', '') for x in message['content'] if x['type'] == 'text']
            n_images = len([1 for x in message['content'] if x['type'] == 'image'])

            prompt_messages.append(dict(role='user', content=content[0], num_images=n_images))

        from lmdeploy.model import Vicuna
        llm_chat_template = Vicuna(eoa=chat_template.eoa, stop_words=chat_template.stop_words)
        prompt = ''
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for i, msg in enumerate(prompt_messages):
            num_images = msg.pop('num_images', 0)
            if num_images == 0:
                role = msg['role']
                msg = llm_chat_template.messages2prompt([msg], sequence_start and i == 0)
                msg = dict(role=role, content=msg)
            prompt_i = chat_template.messages2prompt([msg], sequence_start and i == 0)
            if num_images > 0:
                prompt_i = (IMAGE_TOKEN * num_images) + prompt_i
            prompt += prompt_i
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class CogVLMVisionModel(VisionModel):
    """CogVLM vision model."""

    _arch = 'CogVLMForCausalLM'

    def build_preprocessor(self, trust_remote_code: bool = False):
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

    def build_model(self, trust_remote_code: bool = False):
        if self.with_llm:
            from transformers import AutoModelForCausalLM
            self.vl_model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                                 device_map='cpu',
                                                                 trust_remote_code=trust_remote_code)
        else:
            raise NotImplementedError('turbomind has not supported cogvlm yet')

    def preprocess(self, messages: list[dict]) -> list[dict]:
        """Refer to the spec of `super().preprocess`"""
        images = self.collect_multimodal_items(messages)
        outputs = []
        for modality, image, _ in images:
            pixel_values = self.image_transform(image)
            outputs.append(
                dict(pixel_values=pixel_values,
                     image_size=image.size,
                     image_tokens=self.n_token_per_image,
                     image_token_id=self.image_token_id))
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    @staticmethod
    def proc_messages(messages, chat_template, tools=None, chat_template_kwargs=None):
        """Apply chat template to get the prompt."""
        chat_template_kwargs = chat_template_kwargs or {}
        prompt_messages = []
        image_prefixes = {}
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for idx, message in enumerate(messages):
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
                continue
            content = [x.get('text', '') for x in message['content'] if x['type'] == 'text']
            n_images = len([1 for x in message['content'] if x['type'] == 'image'])
            prompt = content[0]
            if n_images > 0:
                sentinel = f'__LMDEPLOY_COGVLM_IMAGE_{idx}__'
                image_prefixes[sentinel] = IMAGE_TOKEN * n_images
                prompt = f'{sentinel}{prompt}'
            prompt_messages.append(dict(role='user', content=prompt))

        prompt = chat_template.messages2prompt(prompt_messages, tools=tools, **chat_template_kwargs)
        for sentinel, image_prefix in image_prefixes.items():
            prompt = prompt.replace(f'{chat_template.user}{sentinel}', f'{image_prefix}{chat_template.user}', 1)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, tools=None, chat_template_kwargs=None, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, tools, chat_template_kwargs)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer)

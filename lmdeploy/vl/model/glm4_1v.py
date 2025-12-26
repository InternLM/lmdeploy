# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from transformers import AutoConfig

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel

logger = get_logger('lmdeploy')


@VISION_MODELS.register_module()
class GLM4_1_VisionModel(VisionModel):
    """GLM-4.1V-9B-Thinking model."""

    _arch = ['Glm4vForConditionalGeneration']

    @classmethod
    def match(cls, config: AutoConfig):
        """Check whether the config match the model."""
        arch = config.architectures[0] if config.architectures else None
        if arch in cls._arch and hasattr(config, 'vision_config'):
            return True
        return False

    def build_preprocessor(self):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = self.processor.tokenizer
        image_token = self.processor.image_token
        self.image_token_id = tokenizer.encode(image_token)[-1]

    def build_model(self):
        raise NotImplementedError('turbomind has not supported glm4v yet')

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Refer to `super().preprocess()` for spec."""
        images = self.collect_images(messages)
        optional_keys = {'resized_height', 'resized_width', 'min_pixels', 'max_pixels'}
        outputs = []
        for image, params in images:
            image = image.convert('RGB')

            item = dict(type='image', image=image)
            item.update({key: params[key] for key in params.keys() if key in optional_keys})
            result = self.processor.image_processor(images=image, videos=None, return_tensors='pt')
            merge_length = self.processor.image_processor.merge_size**2
            image_tokens = result['image_grid_thw'].prod(dim=1) // merge_length
            result.update(dict(image_size=image.size, image_tokens=image_tokens, image_token_id=self.image_token_id))
            outputs.append(result)
        messages.append(dict(role='preprocess', content=outputs))
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
            prompt = content[0]
            if IMAGE_TOKEN in prompt and '<|begin_of_image|>' not in prompt:
                prompt = prompt.replace(IMAGE_TOKEN, f'<|begin_of_image|>{IMAGE_TOKEN}<|end_of_image|>')
            else:
                prompt = f'<|begin_of_image|>{IMAGE_TOKEN}<|end_of_image|>' * \
                    n_images + prompt
            prompt_messages.append(dict(role=message['role'], content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start, **kwargs):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)

# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel


@VISION_MODELS.register_module()
class Qwen2VLModel(VisionModel):
    """Qwen2VL model."""

    _arch = ['Qwen2VLForConditionalGeneration', 'Qwen2_5_VLForConditionalGeneration']
    _turbomind_native_vision = True

    def build_preprocessor(self, trust_remote_code: bool = False):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        tokenizer = self.processor.tokenizer
        self.image_token = self.processor.image_token
        self.image_token_id = tokenizer.encode(self.image_token)[-1]

    def preprocess(self, messages: list[dict]) -> list[dict]:
        """Refer to `super().preprocess()` for spec."""
        images = self.collect_multimodal_items(messages)
        optional_keys = {'min_pixels', 'max_pixels'}
        outputs = []
        for modality, image, params in images:
            image_kwargs = {key: params[key] for key in params.keys() if key in optional_keys}
            result = self.processor.image_processor(images=[image], return_tensors='pt', **image_kwargs)
            merge_length = self.processor.image_processor.merge_size**2
            image_tokens = result['image_grid_thw'].prod(dim=1) // merge_length
            result.update(dict(image_size=image.size, image_tokens=image_tokens, image_token_id=self.image_token_id))
            outputs.append(result)
        messages.append(dict(role='preprocess', content=outputs))
        return messages

    def build_model(self, trust_remote_code: bool = False):
        arch = self.hf_config.architectures[0]
        if arch == 'Qwen2VLForConditionalGeneration':
            from transformers import Qwen2VLForConditionalGeneration as AutoModelCls
        elif arch == 'Qwen2_5_VLForConditionalGeneration':
            from transformers import Qwen2_5_VLForConditionalGeneration as AutoModelCls
        else:
            raise ValueError(f'Unsupported arch={arch}')

        self.vl_model = AutoModelCls.from_pretrained(self.model_path,
                                                     device_map='cpu',
                                                     trust_remote_code=trust_remote_code)

    def proc_messages(self, messages, chat_template, tools=None, chat_template_kwargs=None):
        """Apply chat template to get the prompt."""
        chat_template_kwargs = chat_template_kwargs or {}
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        messages = [x for x in messages if x['role'] not in ['preprocess', 'forward']]
        if VisionModel.IMAGE_TOKEN_included(messages):
            # backward compatibility
            for message in messages:
                role, content = message['role'], message['content']
                if role != 'user' or isinstance(content, str):
                    prompt_messages.append(message)
                    continue
                content = [x['text'] for x in content if x['type'] == 'text']
                prompt = ''.join(content)
                prompt = prompt.replace(IMAGE_TOKEN, f'<|vision_start|>{self.image_token}<|vision_end|>')
                prompt_messages.append(dict(role='user', content=prompt))
        else:
            for message in messages:
                role, content = message['role'], message['content']
                if role != 'user' or isinstance(content, str):
                    prompt_messages.append(message)
                    continue
                _content = []
                for item in content:
                    if item['type'] == 'text':
                        _content.append(item['text'])
                    elif item['type'] in ['image', 'image_url']:
                        _content.append(f'<|vision_start|>{self.image_token}<|vision_end|>')
                    else:
                        raise ValueError(f'Unsupported message type: {item["type"]}')
                message = dict(role=role, content=''.join(_content))
                prompt_messages.append(message)
        prompt = chat_template.messages2prompt(prompt_messages, tools=tools, **chat_template_kwargs)
        return prompt, self.image_token

    def to_pytorch(self,
                   messages,
                   chat_template,
                   tokenizer,
                   tools=None,
                   chat_template_kwargs=None,
                   **kwargs):
        """Return to the information needed by pytorch engine."""
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, tools, chat_template_kwargs)
        return self.to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer)

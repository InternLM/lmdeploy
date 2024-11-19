# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging


def check_qwen_vl_deps_install():
    """check qwen_vl_utils."""
    try:
        import qwen_vl_utils  # noqa: F401
    except ImportError:
        raise ImportError(
            'please install qwen_vl_utils by pip install qwen_vl_utils'  # noqa: E501
        )
    try:
        from transformers import Qwen2VLForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError(
            'please install latest transformers by '
            'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen2VLModel(VisonModel):
    """Qwen2VL model."""

    _arch = 'Qwen2VLForConditionalGeneration'

    def build_preprocessor(self):
        check_qwen_vl_deps_install()
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def build_model(self):
        from accelerate import init_empty_weights
        with init_empty_weights():
            config = self.hf_config
            config.quantization_config = {}  # disable vision part quantization
            # disable accelerate check_tied_parameters_in_config
            # for Qwen2-VL-2B-Instruct
            config.tie_word_embeddings = False

            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration._from_config(config)
            if not self.with_llm:
                del model.model
                del model.lm_head
            else:
                self.vl_model = model
            model.half()

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map='auto' if not self.with_llm else {'': 'cpu'},
                max_memory=self.max_memory,
                no_split_module_classes=['Qwen2VLVisionBlock'],
                dtype=torch.half)

        self.model = model.eval()

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """refer to `super().preprocess()` for spec."""
        from qwen_vl_utils import process_vision_info

        optional_keys = {
            'resized_height', 'resized_width', 'min_pixels', 'max_pixels'
        }
        outputs = []
        for x in messages[-1]['content']:
            item_type = x['type']
            if item_type == 'image':
                image = x['image'].convert('RGB')
                item = dict(type='image', image=image)
                item.update(
                    {key: x[key]
                     for key in x.keys() if key in optional_keys})
                image_inputs, _ = process_vision_info([dict(content=[item])])
                result = self.processor.image_processor(images=image_inputs,
                                                        videos=None,
                                                        return_tensors='pt')
                result.update(
                    dict(
                        image_size=image.size,
                        image_tokens=1,  # TODO
                        image_token_id=0))
                outputs.append(result)
        return outputs

    @torch.no_grad()
    def forward(self, inputs: List[Dict]) -> List[torch.Tensor]:
        assert 0, 'TODO: support turbomind engine'

    @classmethod
    def proc_messages(cls, messages, chat_template, sequence_start):
        # apply chat template to get the prompt
        IMAGE_TOKEN = '<|image_pad|>'
        prompt_messages = []
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            content = []
            for item in message['content']:
                item_type = item['type']
                if item_type == 'text':
                    content.append(item['text'])
                elif item_type == 'image':
                    content.append(
                        f'<|vision_start|>{IMAGE_TOKEN}<|vision_end|>')
                else:
                    assert 0, (
                        f'unsupported type {item_type} in {message["content"]}'
                    )
            prompt_messages.append(dict(role='user', content=''.join(content)))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        """return to the information needed by pytorch engine."""
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template,
                                                 sequence_start)
        return super().to_pytorch_aux(messages, prompt, IMAGE_TOKEN, tokenizer,
                                      sequence_start)

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        assert 0, 'TODO: support turbomind engine'

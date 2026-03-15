# Copyright (c) OpenMMLab. All rights reserved.
import torch
from transformers import AutoProcessor

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS
from lmdeploy.vl.model.utils import disable_logging

from .qwen3 import Qwen3VLModel

logger = get_logger('lmdeploy')


def check_transformers():
    try:
        from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen3_5Model(Qwen3VLModel):
    """Qwen3_5 model."""

    _arch = ['Qwen3_5ForConditionalGeneration', 'Qwen3_5MoeForConditionalGeneration']

    def build_preprocessor(self):
        check_transformers()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = self.processor.tokenizer
        self.image_token = self.processor.image_token
        self.image_token_id = tokenizer.encode(self.image_token)[-1]
        self.mm_processor_kwargs = None

    def build_model(self):
        check_transformers()
        arch = self.hf_config.architectures[0]
        if arch == 'Qwen3_5ForConditionalGeneration':
            from transformers import Qwen3_5ForConditionalGeneration as AutoModelCls
        elif arch == 'Qwen3_5MoeForConditionalGeneration':
            from transformers import Qwen3_5MoeForConditionalGeneration as AutoModelCls
        else:
            raise ValueError(f'Unsupported arch={arch}')

        if self.with_llm:
            self.vl_model = AutoModelCls.from_pretrained(self.model_path, device_map='cpu')
        else:
            from accelerate import init_empty_weights
            with init_empty_weights():
                config = self.hf_config
                config.tie_word_embeddings = False
                if hasattr(config, 'text_config'):
                    config.text_config.tie_word_embeddings = False
                model = AutoModelCls._from_config(config)
                del model.model.language_model
                del model.lm_head
                model.half()

            from accelerate import load_checkpoint_and_dispatch
            with disable_logging():
                load_checkpoint_and_dispatch(model=model,
                                             checkpoint=self.model_path,
                                             device_map='auto' if not self.with_llm else {'': 'cpu'},
                                             max_memory=self.max_memory,
                                             no_split_module_classes=['Qwen3_5VisionBlock', 'Qwen3_5MoeVisionBlock'],
                                             dtype=torch.half)
            self.model = model.model.eval()

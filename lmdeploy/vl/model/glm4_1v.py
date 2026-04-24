# Copyright (c) OpenMMLab. All rights reserved.

from transformers import AutoConfig

from lmdeploy.utils import get_logger
from lmdeploy.vl.constants import MultimodalSpecialTokens
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

        self.image_token = self.processor.image_token
        self.image_token_id = self.processor.image_token_id

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            image_token_id=self.image_token_id
        )

    def apply_chat_template(self, messages, chat_template, sequence_start, chat_template_kwargs=None):
        """Apply chat template to get the prompt."""
        return chat_template.messages2prompt(messages, sequence_start, **(chat_template_kwargs or {}))

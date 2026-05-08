# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoProcessor

from lmdeploy.vl.model.base import VISION_MODELS, MultimodalSpecialTokens, VisionModel


def check_transformers():
    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration  # noqa: F401
    except ImportError:
        raise ImportError('please install latest transformers by '
                          'pip install git+https://github.com/huggingface/transformers.git')


@VISION_MODELS.register_module()
class Qwen3OmniModel(VisionModel):
    """Qwen3Omni model."""

    _arch = ['Qwen3OmniMoeForConditionalGeneration']

    def build_preprocessor(self, trust_remote_code: bool = False):
        check_transformers()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        tokenizer = self.processor.tokenizer

        # image tokens
        self.image_token = self.processor.image_token
        self.image_token_id = tokenizer.encode(self.image_token)[-1]

        # video tokens
        self.video_token = self.processor.video_token
        self.video_token_id = tokenizer.encode(self.video_token)[-1]

        # audio tokens
        self.audio_token = self.processor.audio_token
        self.audio_token_id = tokenizer.encode(self.audio_token)[-1]

        self.mm_tokens = MultimodalSpecialTokens(image_token=self.image_token,
                                                 video_token=self.video_token,
                                                 audio_token=self.audio_token,
                                                 image_token_id=self.image_token_id,
                                                 video_token_id=self.video_token_id,
                                                 audio_token_id=self.audio_token_id)

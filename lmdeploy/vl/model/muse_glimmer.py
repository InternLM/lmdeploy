# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoProcessor

from lmdeploy.vl.model.base import VISION_MODELS, MultimodalSpecialTokens
from lmdeploy.vl.model.qwen3 import Qwen3VLModel


def check_transformers():
    try:
        from transformers.models.muse_glimmer.processing_muse_glimmer import MuseGlimmerProcessor  # noqa: F401
    except ImportError as e:
        raise ImportError(
            'Muse-Glimmer requires a Transformers version containing '
            '`MuseGlimmerProcessor`; install the latest Transformers source.') from e


@VISION_MODELS.register_module()
class MuseGlimmerModel(Qwen3VLModel):
    """Muse-Glimmer image/video frontend."""

    _arch = ['MuseGlimmerForConditionalGeneration']
    _turbomind_native_vision = True

    def build_preprocessor(self, trust_remote_code: bool = False):
        check_transformers()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=trust_remote_code)
        self.image_token = self.processor.image_token
        self.image_token_id = self.processor.image_token_id
        self.video_token = self.processor.video_token
        self.video_token_id = self.processor.video_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            video_token=self.video_token,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )

    def build_model(self, trust_remote_code: bool = False):
        check_transformers()
        from transformers import MuseGlimmerForConditionalGeneration
        self.vl_model = MuseGlimmerForConditionalGeneration.from_pretrained(
            self.model_path, device_map='cpu')

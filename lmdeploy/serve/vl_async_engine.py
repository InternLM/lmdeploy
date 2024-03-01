# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

import numpy as np

from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.vl.constants import IMAGE_DUMMY_TOKEN_INDEX, IMAGE_TOKEN
from lmdeploy.vl.engine import ImageEncoder
from lmdeploy.vl.templates import VLPromptType, get_vl_prompt_template


class VLAsyncEngine(AsyncEngine):
    """Visual Language Async inference engine."""

    def __init__(self, model_path: str, **kwargs) -> None:
        super().__init__(model_path, **kwargs)
        self.vl_encoder = ImageEncoder(model_path)
        self.vl_prompt_template = get_vl_prompt_template(
            model_path, self.chat_template, self.model_name)

    def __call__(self, prompts: Union[str, VLPromptType, List[Dict],
                                      List[Union[str, VLPromptType]],
                                      List[List[Dict]]], **kwargs):
        if isinstance(prompts, str) or isinstance(prompts, tuple):
            _prompts = self.vl_prompt_template.prompt_to_messages(prompts)
        elif isinstance(prompts[0], tuple) or isinstance(prompts[0], str):
            _prompts = [
                self.vl_prompt_template.prompt_to_messages(x) for x in prompts
            ]
        else:
            _prompts = prompts

        return super().__call__(_prompts, **kwargs)

    async def _get_prompt_input(self, prompt: Dict, do_preprocess: bool,
                                sequence_start: bool):
        if do_preprocess:
            decorated = self.vl_prompt_template.messages2prompt(
                prompt, sequence_start)
        segs = decorated.split(IMAGE_TOKEN)

        results = {}
        input_ids = []
        if len(segs) > 1:
            images = await self.vl_prompt_template.async_collect_pil_images(
                prompt)
            features = await self.vl_encoder.async_infer(images)
            features = [x.cpu().numpy() for x in features]
            input_ids = []
            begins = []
            ends = []
            for i, seg in enumerate(segs):
                if i > 0:
                    image_dim = features[i - 1].shape[0]
                    begins.append(len(input_ids))
                    ends.append(begins[-1] + image_dim)
                    input_ids.extend([IMAGE_DUMMY_TOKEN_INDEX] * image_dim)
                seg_ids = self.tokenizer.encode(seg,
                                                add_bos=((i == 0)
                                                         and sequence_start))
                input_ids.extend(seg_ids)
            ranges = np.stack([begins, ends], axis=1).tolist()
            results['input_embeddings'] = features
            results['input_embedding_ranges'] = ranges
        else:
            input_ids = self.tokenizer.encode(prompt, add_bos=sequence_start)

        results['input_ids'] = input_ids
        return results

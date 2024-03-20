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
        if self.model_name == 'base':
            raise RuntimeError(
                'please specify chat template as guided in https://lmdeploy.readthedocs.io/en/latest/inference/vl_pipeline.html#set-chat-template'  # noqa: E501
            )
        self.vl_encoder = ImageEncoder(model_path)
        self.vl_prompt_template = get_vl_prompt_template(
            model_path, self.chat_template, self.model_name)

    def _convert_prompts(self,
                         prompts: Union[VLPromptType, List[Dict],
                                        List[VLPromptType], List[List[Dict]]]):
        """convert prompts to openai format."""
        if isinstance(prompts, str) or isinstance(prompts, tuple):
            _prompts = self.vl_prompt_template.prompt_to_messages(prompts)
        elif isinstance(prompts[0], tuple) or isinstance(prompts[0], str):
            _prompts = [
                self.vl_prompt_template.prompt_to_messages(x) for x in prompts
            ]
        else:
            _prompts = prompts
        return _prompts

    async def _get_prompt_input(self, prompt: Dict, do_preprocess: bool,
                                sequence_start: bool):
        """get input_ids, embeddings and offsets."""
        if do_preprocess:
            decorated = self.vl_prompt_template.messages2prompt(
                prompt, sequence_start)
        else:
            decorated = prompt
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
            input_ids = self.tokenizer.encode(decorated,
                                              add_bos=sequence_start)

        results['input_ids'] = input_ids
        results['prompt'] = decorated
        return results

    def batch_infer(self, prompts: Union[VLPromptType, List[Dict],
                                         List[VLPromptType], List[List[Dict]]],
                    **kwargs):
        """Inference a batch of prompts."""
        prompts = self._convert_prompts(prompts)
        return super().batch_infer(prompts, **kwargs)

    def stream_infer(self, prompts: Union[VLPromptType, List[Dict],
                                          List[VLPromptType],
                                          List[List[Dict]]], **kwargs):
        """Inference a batch of prompts with stream mode."""
        prompts = self._convert_prompts(prompts)
        return super().stream_infer(prompts, **kwargs)

    def __call__(self, prompts: Union[VLPromptType, List[Dict],
                                      List[VLPromptType], List[List[Dict]]],
                 **kwargs):
        """Inference a batch of prompts."""
        prompts = self._convert_prompts(prompts)
        return super().__call__(prompts, **kwargs)

    def chat(self, prompts: VLPromptType, **kwargs):
        """chat."""
        _prompts = self._convert_prompts(prompts)
        sess = super().chat(_prompts, **kwargs)

        # recover prompts & history
        sess._prompt = prompts
        last_round = sess.history[-1]
        sess.history[-1] = (prompts, last_round[-1])
        return sess

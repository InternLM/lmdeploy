# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Any, Dict, List, Literal, Optional, Union

from lmdeploy import Tokenizer
from lmdeploy.model import MODELS, BaseChatTemplate
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class MultimodalProcessor:
    """Processor for handling prompt preprocessing, message content merging,
    and multimodal processing."""

    def __init__(self,
                 tokenizer: Tokenizer,
                 chat_template: BaseChatTemplate,
                 vl_encoder=None,
                 backend: Optional[str] = None):
        """Initialize MultimodalProcessor.

        Args:
            tokenizer: Tokenizer instance for encoding prompts.
            chat_template: Chat template instance for message processing.
            vl_encoder: Optional ImageEncoder instance for multimodal processing.
            backend: Optional backend name ('turbomind' or 'pytorch') for multimodal processing.
        """
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.vl_encoder = vl_encoder
        self.backend = backend

    @staticmethod
    def merge_message_content(msg: Dict) -> Dict:
        """Merge multimodal content blocks and ensure content field exists.

        This function normalizes message content to match vLLM's behavior:
        1. Missing content field -> add content='' (empty string)
        2. None content -> convert to content='' (empty string)
        3. String content -> return as-is
        4. List content (multimodal) -> merge all text blocks with newline separator

        Args:
            msg: A message dict with 'role' and optionally 'content' field

        Returns:
            A message dict with 'content' field guaranteed to exist

        Note:
            This implementation is based on vLLM's content processing logic.
            vLLM uses "\n".join() to merge multiple text blocks from multimodal content.

        References:
            - vLLM content normalization:
              https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/chat_utils.py
              See _parse_chat_message_content() and _parse_chat_message_content_parts()
            - vLLM text merging logic:
              text_prompt = "\n".join(texts)
        """
        # If content is missing or None, convert to empty string (matches vLLM behavior)
        # This prevents Jinja2 template errors when rendering chat templates
        if 'content' not in msg or msg['content'] is None:
            result = dict(msg)
            result['content'] = ''
            return result

        # If content is already a string, return as-is
        if isinstance(msg['content'], str):
            return msg

        # If content is a list, merge all text blocks into a single string
        # This matches vLLM's behavior: text_prompt = "\n".join(texts)
        content_parts = []
        for block in msg['content']:
            if isinstance(block, dict) and block.get('type') == 'text':
                content_parts.append(block.get('text', ''))
        merged_content = '\n'.join(content_parts)

        # Preserve all other fields in the message (e.g., tool_calls)
        result = dict(msg)
        result['content'] = merged_content
        return result

    @staticmethod
    async def async_convert_multimodal_data(messages: List[Dict]) -> List[Dict]:
        """Convert user-input multimodal data into GPT4V message format."""
        from lmdeploy.vl.utils import load_image

        if isinstance(messages, Dict):
            messages = [messages]
        assert isinstance(messages, List)

        out_messages = [None] * len(messages)

        def _inner_call(i, in_messages, out_messages):
            role = in_messages[i]['role']
            content = in_messages[i]['content']
            assert role in ['system', 'user', 'assistant'], \
                f'unsupported role "{role}"'
            if role != 'user' or isinstance(content, str):
                # the content is a user's prompt or an assistant's prompt,
                # returning it directly
                out_messages[i] = in_messages[i]
                return
            # the role is a user and the content is a list, in which there
            # might be image_url or image_data
            assert isinstance(content, List)
            message = dict(role=role, content=[])
            for item in content:
                # image url or base64-encoded image data
                if item['type'] == 'image_url':
                    """
                    convert the following item:
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': 'image url or base64-encoded image data',
                            'key': 'value'  # parameters used in image processing
                            ...
                        }
                    }
                    to:
                    {
                        'type': 'image',
                        'image': Pillow.Image,
                        'key': 'value'   # parameters used in image processing
                        ...
                    }
                    """  # noqa
                    data = item['image_url'].copy()
                    try:
                        url = data.pop('url')
                        image = load_image(url)
                        data.update(type='image', image=image)
                        message['content'].append(data)
                    except KeyError:
                        logger.error(f'invalid format {message}')
                elif item['type'] == 'image_data':
                    """
                    convert the following item:
                    {
                        'type': 'image_data',
                        'image_data': {
                            'data': Pillow.Image,
                            'key': 'value'  # parameters used in image processing
                            ...
                        }
                    }
                    to:
                    {
                        'type': 'image',
                        'image': Pillow.Image,
                        'key': 'value'   # parameters used in image processing
                        ...
                    }
                    """  # noqa
                    data = item['image_data'].copy()
                    try:
                        image = data.pop('data')
                        data.update(type='image', image=image)
                        message['content'].append(data)
                    except KeyError:
                        logger.error(f'invalid format {message}')
                elif item['type'] in ['text', 'time_series']:
                    message['content'].append(item)
                else:
                    logger.error(f'unexpected content type {message}')
            out_messages[i] = message

        await asyncio.gather(*[
            asyncio.get_event_loop().run_in_executor(None, _inner_call, i, messages, out_messages)
            for i in range(len(messages))
        ])
        return out_messages

    async def get_prompt_input(self,
                               prompt: Union[str, List[Dict]],
                               do_preprocess: bool,
                               sequence_start: bool,
                               adapter_name: str,
                               tools: Optional[List[object]] = None,
                               reasoning_effort: Optional[Literal['low', 'medium', 'high']] = None,
                               chat_template_kwargs: Optional[Dict] = None,
                               mm_processor_kwargs: Optional[Dict[str, Any]] = None,
                               **kwargs):
        """Process prompt and return prompt string and input_ids.

        Handles both text-only and multimodal prompts. If multimodal input is detected
        and vl_encoder is available, processes images accordingly.

        Args:
            prompt: Input prompt as string or list of message dicts.
            do_preprocess: Whether to apply chat template preprocessing.
            sequence_start: Indicator for starting a sequence.
            adapter_name: Adapter name for selecting chat template.
            tools: Optional list of tools.
            reasoning_effort: Optional reasoning effort level.
            chat_template_kwargs: Optional kwargs for chat template.
            mm_processor_kwargs: Optional kwargs for multimodal processor.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict with 'prompt' (str) and 'input_ids' (List[int]) keys for text-only,
            or dict with multimodal data for multimodal prompts.
        """
        # Handle string input
        if isinstance(prompt, str):
            return await self._get_text_prompt_input(prompt=prompt,
                                                     do_preprocess=do_preprocess,
                                                     sequence_start=sequence_start,
                                                     adapter_name=adapter_name,
                                                     tools=tools,
                                                     reasoning_effort=reasoning_effort,
                                                     chat_template_kwargs=chat_template_kwargs,
                                                     **kwargs)

        # Handle list input
        elif isinstance(prompt, list):
            # Check if multimodal input exists
            has_multimodal_input = self._has_multimodal_input(prompt)

            # If no multimodal input or no vl_encoder, use text-only processing
            if not has_multimodal_input or self.vl_encoder is None:
                return await self._get_text_prompt_input(prompt=prompt,
                                                         do_preprocess=do_preprocess,
                                                         sequence_start=sequence_start,
                                                         adapter_name=adapter_name,
                                                         tools=tools,
                                                         reasoning_effort=reasoning_effort,
                                                         chat_template_kwargs=chat_template_kwargs,
                                                         **kwargs)

            # Process multimodal input
            return await self._get_multimodal_prompt_input(messages=prompt,
                                                           do_preprocess=do_preprocess,
                                                           sequence_start=sequence_start,
                                                           adapter_name=adapter_name,
                                                           tools=tools,
                                                           chat_template_kwargs=chat_template_kwargs,
                                                           mm_processor_kwargs=mm_processor_kwargs,
                                                           **kwargs)
        else:
            raise RuntimeError(f'unsupported prompt type: {type(prompt)}')

    def _has_multimodal_input(self, messages: List[Dict]) -> bool:
        """Check if messages contain multimodal input (images)."""
        return any(
            isinstance(message.get('content'), list) and any(
                item.get('type') in ['image_url', 'image_data', 'time_series'] for item in message['content'])
            for message in messages)

    async def _get_text_prompt_input(self,
                                     prompt: Union[str, List[Dict]],
                                     do_preprocess: bool,
                                     sequence_start: bool,
                                     adapter_name: str,
                                     tools: Optional[List[object]] = None,
                                     reasoning_effort: Optional[Literal['low', 'medium', 'high']] = None,
                                     chat_template_kwargs: Optional[Dict] = None,
                                     **kwargs):
        """Process text-only prompt and return prompt string and input_ids."""
        # Change multimodal data to openai text messages
        if isinstance(prompt, list):
            prompt = [self.merge_message_content(msg) for msg in prompt]
        if do_preprocess:
            # use adapter's chat template if possible
            chat_template = self.chat_template
            if adapter_name in MODELS.module_dict:
                chat_template = MODELS.module_dict[adapter_name]()
        else:
            chat_template = BaseChatTemplate()
        chat_template_kwargs = chat_template_kwargs or {}
        prompt = chat_template.messages2prompt(prompt,
                                               sequence_start,
                                               tools=tools,
                                               reasoning_effort=reasoning_effort,
                                               **chat_template_kwargs)
        if prompt is None:
            raise ValueError(
                f'You are using base template to handle chat task. Please specify a `--chat-template` name chosen from `lmdeploy list` if you want to use OpenAI messages input.'  # noqa
            )
        input_ids = self.tokenizer.encode(prompt, add_bos=sequence_start)
        return {'prompt': prompt, 'input_ids': input_ids}

    async def _get_multimodal_prompt_input(self,
                                           messages: List[Dict],
                                           do_preprocess: bool,
                                           sequence_start: bool,
                                           adapter_name: str,
                                           tools: Optional[List[object]] = None,
                                           chat_template_kwargs: Optional[Dict] = None,
                                           mm_processor_kwargs: Optional[Dict[str, Any]] = None,
                                           **kwargs):
        """Process multimodal prompt and return processed data for inference
        engines."""
        chat_template = self.chat_template if do_preprocess else BaseChatTemplate()
        messages = await self.async_convert_multimodal_data(messages)
        results = await self.vl_encoder.preprocess(messages, mm_processor_kwargs)

        if self.backend == 'turbomind':
            # for tm engine, this module perform vision embedding after image
            # preprocessing. It utilizes the hf model's vision embeddings
            # functions and returns the input_ids, input_embeddings,
            # embedding_ranges and so on. All the returned values are passed
            # to tm engine for token generation
            results = await self.vl_encoder.async_infer(results)
            results = await self.vl_encoder.wrap_for_turbomind(results,
                                                               chat_template,
                                                               self.tokenizer,
                                                               sequence_start,
                                                               tools=tools,
                                                               chat_template_kwargs=chat_template_kwargs)
        elif self.backend == 'pytorch':
            # for pt engine, this module only conduct the image preprocessing
            # It leaves the vision embedding to the pt engine
            results = await self.vl_encoder.wrap_for_pytorch(results,
                                                             chat_template,
                                                             self.tokenizer,
                                                             sequence_start,
                                                             tools=tools,
                                                             chat_template_kwargs=chat_template_kwargs)
        return results

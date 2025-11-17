# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import Any, Dict, List, Optional, Union

import requests

from lmdeploy.utils import get_logger


def get_model_list(api_url: str, headers: dict = None):
    """Get model list from api server."""
    response = requests.get(api_url, headers=headers)
    logger = get_logger('lmdeploy')
    if not response.ok:
        logger.error(f'Failed to get the model list: {api_url}'
                     f' returns {response.status_code}')
        return None
    elif not hasattr(response, 'text'):
        logger.warning('Failed to get the model list.')
        return None
    else:
        model_list = response.json()
        model_list = model_list.pop('data', [])
        return [item['id'] for item in model_list]


def json_loads(content):
    """Loads content to json format."""
    try:
        content = json.loads(content)
        return content
    except:  # noqa
        logger = get_logger('lmdeploy')
        logger.warning(f'weird json content {content}')
        return ''


class APIClient:
    """Chatbot for LLaMA series models with turbomind as inference engine.

    Args:
        api_server_url (str): communicating address 'http://<ip>:<port>' of
            api_server
        api_key (str | None): api key. Default to None, which means no
            api key will be used.
    """

    def __init__(self, api_server_url: str, api_key: Optional[str] = None, **kwargs):
        self.api_server_url = api_server_url
        self.chat_completions_v1_url = f'{api_server_url}/v1/chat/completions'
        self.completions_v1_url = f'{api_server_url}/v1/completions'
        self.models_v1_url = f'{api_server_url}/v1/models'
        self.encode_v1_url = f'{api_server_url}/v1/encode'
        self._available_models = None
        self.api_key = api_key
        self.headers = {'content-type': 'application/json'}
        if api_key is not None:
            self.headers['Authorization'] = f'Bearer {api_key}'

    @property
    def available_models(self):
        """Show available models."""
        if self._available_models is not None:
            return self._available_models
        self._available_models = get_model_list(self.models_v1_url, headers=self.headers)
        return self._available_models

    def encode(self,
               input: Union[str, List[str]],
               do_preprocess: Optional[bool] = False,
               add_bos: Optional[bool] = True):
        """Encode prompts.

        Args:
            input: the prompt to be encoded. In str or List[str] format.
            do_preprocess: whether do preprocess or not. Default to False.
            add_bos: True when it is the beginning of a conversation. False
                when it is not. Default to True.
        Return: (input_ids, length)
        """
        response = requests.post(self.encode_v1_url,
                                 headers=self.headers,
                                 json=dict(input=input, do_preprocess=do_preprocess, add_bos=add_bos),
                                 stream=False)
        if hasattr(response, 'text'):
            output = json_loads(response.text)
            return output['input_ids'], output['length']
        return None, None

    def chat_completions_v1(
        self,
        model: str,
        messages: Union[str, List[Dict[str, str]]],
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 1.0,
        logprobs: Optional[bool] = False,
        top_logprobs: Optional[int] = 0,
        n: Optional[int] = 1,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = False,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0,
        user: Optional[str] = None,
        repetition_penalty: Optional[float] = 1.0,
        ignore_eos: Optional[bool] = False,
        skip_special_tokens: Optional[bool] = True,
        spaces_between_special_tokens: Optional[bool] = True,
        top_k: int = 40,
        min_new_tokens: Optional[int] = None,
        min_p: float = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
        stream_options: Optional[Dict] = None,
        **kwargs,
    ):
        """Chat completion v1.

        Args:
            model: model name. Available from self.available_models.
            messages: string prompt or chat history in OpenAI format. Chat
                history example: `[{"role": "user", "content": "hi"}]`.
            temperature (float): to modulate the next token probability
            top_p (float): If set to float < 1, only the smallest set of most
                probable tokens with probabilities that add up to top_p or
                higher are kept for generation.
            n (int): How many chat completion choices to generate for each
                input message. Only support one here.
            stream: whether to stream the results or not. Default to false.
            max_completion_tokens (int | None): output token nums. Default to None.
            max_tokens (int | None): output token nums. Default to None.
                Deprecated: Use max_completion_tokens instead.
            stop (str | List[str] | None): To stop generating further
              tokens. Only accept stop words that's encoded to one token idex.
            repetition_penalty (float): The parameter for repetition penalty.
                1.0 means no penalty
            ignore_eos (bool): indicator for ignoring eos
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be True.
            spaces_between_special_tokens (bool): Whether or not to add spaces
                around special tokens. The behavior of Fast tokenizers is to have
                this to False. This is setup to True in slow tokenizers.
            top_k (int): The number of the highest probability vocabulary
                tokens to keep for top-k-filtering
            min_new_tokens (int): To generate at least numbers of tokens.
            min_p (float): Minimum token probability, which will be scaled by the
                probability of the most likely token. It must be a value between
                0 and 1. Typical values are in the 0.01-0.2 range, comparably
                selective as setting `top_p` in the 0.99-0.8 range (use the
                opposite of normal `top_p` values)
            logit_bias (Dict): Bias to logits. Only supported in pytorch engine.
            stream_options: Options for streaming response. Only set this when you
                set stream: true.

        Yields:
            json objects in openai formats
        """
        pload = {k: v for k, v in locals().copy().items() if k[:2] != '__' and k not in ['self']}
        response = requests.post(self.chat_completions_v1_url, headers=self.headers, json=pload, stream=stream)
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b'\n'):
            if chunk:
                if stream:
                    decoded = chunk.decode('utf-8')
                    if decoded == 'data: [DONE]':
                        continue
                    if decoded[:6] == 'data: ':
                        decoded = decoded[6:]
                    output = json_loads(decoded)
                    yield output
                else:
                    decoded = chunk.decode('utf-8')
                    output = json_loads(decoded)
                    yield output

    def completions_v1(
        self,
        model: str,
        prompt: Union[str, List[Any]],
        suffix: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        n: Optional[int] = 1,
        max_completion_tokens: Optional[int] = 16,
        max_tokens: Optional[int] = 16,
        stream: Optional[bool] = False,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: Optional[float] = 1.0,
        top_k: Optional[int] = 40,
        user: Optional[str] = None,
        # additional argument of lmdeploy
        repetition_penalty: Optional[float] = 1.0,
        ignore_eos: Optional[bool] = False,
        skip_special_tokens: Optional[bool] = True,
        spaces_between_special_tokens: Optional[bool] = True,
        stream_options: Optional[Dict] = None,
        **kwargs,
    ):
        """Chat completion v1.

        Args:
            model (str): model name. Available from /v1/models.
            prompt (str): the input prompt.
            suffix (str): The suffix that comes after a completion of inserted
                text.
            max_completion_tokens (int | None): output token nums. Default to 16.
            max_tokens (int): output token nums
                Deprecated: Use max_completion_tokens instead.
            temperature (float): to modulate the next token probability
            top_p (float): If set to float < 1, only the smallest set of most
                probable tokens with probabilities that add up to top_p or
                higher are kept for generation.
            top_k (int): The number of the highest probability vocabulary
                tokens to keep for top-k-filtering
            n (int): How many chat completion choices to generate for each
                input message. Only support one here.
            stream: whether to stream the results or not. Default to false.
            stop (str | List[str] | None): To stop generating further
              tokens. Only accept stop words that's encoded to one token idex.
            repetition_penalty (float): The parameter for repetition penalty.
                1.0 means no penalty
            user (str): A unique identifier representing your end-user.
            ignore_eos (bool): indicator for ignoring eos
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be True.
            spaces_between_special_tokens (bool): Whether or not to add spaces
                around special tokens. The behavior of Fast tokenizers is to have
                this to False. This is setup to True in slow tokenizers.
            stream_options: Options for streaming response. Only set this when you
                set stream: true.

        Yields:
            json objects in openai formats
        """
        pload = {k: v for k, v in locals().copy().items() if k[:2] != '__' and k not in ['self']}
        response = requests.post(self.completions_v1_url, headers=self.headers, json=pload, stream=stream)
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b'\n'):
            if chunk:
                if stream:
                    decoded = chunk.decode('utf-8')
                    if decoded == 'data: [DONE]':
                        continue
                    if decoded[:6] == 'data: ':
                        decoded = decoded[6:]
                    output = json_loads(decoded)
                    yield output
                else:
                    decoded = chunk.decode('utf-8')
                    output = json_loads(decoded)
                    yield output

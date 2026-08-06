# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from lmdeploy.serve.openai.protocol import (
    EmbeddingsRequest,
    EncodeRequest,
    EncodeResponse,
    PoolingRequest,
    PoolingResponse,
    PPLRequest,
    PPLResponse,
    UsageInfo,
)
from lmdeploy.serve.openai.utils import create_error_response
from lmdeploy.serve.utils.server_utils import validate_json_request


def register(router: APIRouter, server_context) -> None:

    @router.post('/v1/embeddings', tags=['unsupported'])
    async def create_embeddings(request: EmbeddingsRequest,
                                raw_request: Request = None):
        """Creates embeddings for the text."""
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     'Unsupported by turbomind.')

    @router.post('/v1/encode', dependencies=[Depends(validate_json_request)])
    async def encode(request: EncodeRequest, raw_request: Request = None):
        """Encode prompts.

        The request should be a JSON object with the following fields:

        - **input**: the prompt to be encoded. In str or list[str] format.
        - **do_preprocess**: whether do preprocess or not. Default to False.
        - **add_bos**: Whether to add a BOS token when encoding. Default to True.
        """

        def encode(prompt: str, do_preprocess: bool, add_bos: bool):
            if do_preprocess:
                prompt = server_context.async_engine.chat_template.get_prompt(
                    prompt)
            input_ids = server_context.async_engine.tokenizer.encode(
                prompt, add_bos=add_bos)
            return input_ids

        if isinstance(request.input, str):
            encoded = encode(request.input, request.do_preprocess,
                             request.add_bos)
            return EncodeResponse(input_ids=encoded, length=len(encoded))
        else:
            encoded, length = [], []
            for prompt in request.input:
                ids = encode(prompt, request.do_preprocess, request.add_bos)
                encoded.append(ids)
                length.append(len(ids))
            return EncodeResponse(input_ids=encoded, length=length)

    @router.post('/pooling', dependencies=[Depends(validate_json_request)])
    async def pooling(request: PoolingRequest, raw_request: Request = None):
        """Pooling prompts for reward model.

        In vLLM documentation, https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#pooling-api_1,
        the input format of Pooling API is the same as Embeddings API.

        Go to https://platform.openai.com/docs/api-reference/embeddings/create
        for the Embeddings API specification.

        The request should be a JSON object with the following fields:

        - **model** (str): model name. Available from /v1/models.
        - **input** (list[int] | list[list[int]] | str | list[str]): input text to be embed
        """

        async_engine = server_context.async_engine

        request_input = request.input
        model_name = request.model or async_engine.model_name

        # Normalize all inputs to be a batch (List[List[int]])
        if isinstance(request_input, str):
            input_ids = [async_engine.tokenizer.encode(request_input)]
        elif isinstance(request_input, list):
            if not request_input:
                return create_error_response(HTTPStatus.BAD_REQUEST,
                                             'Input list cannot be empty.')
            if isinstance(request_input[0], str):  # list[str]
                input_ids = [
                    async_engine.tokenizer.encode(p) for p in request_input
                ]
            elif isinstance(request_input[0], int):  # list[int]
                input_ids = [request_input]
            elif isinstance(request_input[0], list):  # list[list[int]]
                input_ids = request_input
            else:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    'Input list contains an invalid type.')
        else:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         'Input must be a string or a list.')

        batch_scores = await async_engine.async_get_reward_score(input_ids)
        prompt_tokens = sum(len(ids) for ids in input_ids)
        usage = UsageInfo.build(prompt_tokens=prompt_tokens,
                                completion_tokens=0,
                                cached_tokens=0)

        data = []
        for i, score in enumerate(batch_scores):
            data.append({
                'index': i,
                'object': 'pooling',
                'data': score,
            })

        response = PoolingResponse(model=model_name, data=data, usage=usage)
        return response.model_dump()

    @router.post('/get_ppl', dependencies=[Depends(validate_json_request)])
    async def get_ppl(request: PPLRequest, raw_request: Request = None):
        """Get the perplexity (mean cross-entropy loss) of the input prompt.

        The request should be a JSON object with the following fields:

        - **input** (str | list[int]): the input to score, either raw text or token
          ids. Text is tokenized with ``tokenizer.encode`` (no chat template is
          applied).
        """
        async_engine = server_context.async_engine

        request_input = request.input

        # pydantic already validated `input` as `str | list[int]`; text ->
        # tokenizer.encode, otherwise the token ids are used as-is
        if isinstance(request_input, str):
            input_ids = async_engine.tokenizer.encode(request_input)
        else:
            input_ids = request_input
        if not input_ids:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         'Input must not be empty.')

        try:
            ppl = await async_engine.async_get_ppl(input_ids)
        except ValueError as e:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

        response = PPLResponse(ppl=ppl)
        return response.model_dump()

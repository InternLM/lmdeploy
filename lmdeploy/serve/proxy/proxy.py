# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os
import os.path as osp
import random
import threading
import time
from collections import deque
from http import HTTPStatus
from typing import Deque, Dict, List, Literal, Optional, Union

import numpy as np
import requests
import uvicorn
import yaml
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from lmdeploy.serve.openai.api_server import (check_api_key,
                                              create_error_response)
from lmdeploy.serve.openai.protocol import (  # noqa: E501
    ChatCompletionRequest, CompletionRequest, ModelCard, ModelList,
    ModelPermission)
from lmdeploy.serve.proxy.constants import (API_TIMEOUT_LEN,
                                            LATENCY_DEEQUE_LEN, ErrorCodes,
                                            Strategy, err_msg)
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class Status(BaseModel):
    """Status protocol consists of models' information."""
    models: Optional[List[str]] = Field(default=[], examples=[[]])
    unfinished: int = 0
    latency: Deque = Field(default=deque(maxlen=LATENCY_DEEQUE_LEN),
                           examples=[[]])
    speed: Optional[int] = Field(default=None, examples=[None])


class Node(BaseModel):
    """Node protocol consists of url and status."""
    url: str
    status: Optional[Status] = None


CONTROLLER_HEART_BEAT_EXPIRATION = int(
    os.getenv('LMDEPLOY_CONTROLLER_HEART_BEAT_EXPIRATION', 90))


def heart_beat_controller(proxy_controller):
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        logger.info('Start heart beat check')
        proxy_controller.remove_stale_nodes_by_expiration()


class NodeManager:
    """Manage all the sub nodes.

    Args:
        config_path (str): the path of the config file.
        strategy (str): the strategy to dispatch node to handle the requests.
            - random: not fully radom, but decided by the speed of nodes.
            - min_expected_latency: will compute the expected latency to
                process the requests. The sooner of the node, the more requests
                will be dispatched to it.
            - min_observed_latency: Based on previous finished requests. The
                sooner they get processed, the more requests will be dispatched
                to.
    """

    def __init__(self,
                 config_path: Optional[str] = None,
                 strategy: str = 'min_expected_latency') -> None:
        self.nodes = dict()
        self.strategy = Strategy.from_str(strategy)
        self.latencies = dict()
        self.config_path = osp.join(osp.dirname(osp.realpath(__file__)),
                                    'proxy_config.yml')
        if config_path is not None:
            self.config_path = config_path
        if osp.exists(self.config_path):
            with open(self.config_path, 'r') as config_file:
                self.nodes = yaml.safe_load(config_file)['nodes']
                for url, status in self.nodes.items():
                    status = Status(**status)
                    self.nodes[url] = status
        self.heart_beat_thread = threading.Thread(target=heart_beat_controller,
                                                  args=(self, ),
                                                  daemon=True)
        self.heart_beat_thread.start()

    def update_config_file(self):
        """Update the config file."""
        nodes = copy.deepcopy(self.nodes)
        for url, status in nodes.items():
            nodes[url] = status.model_dump()
            nodes[url]['latency'] = list(status.latency)
        with open(self.config_path, 'w') as config_file:  # update cfg yml
            yaml.dump(dict(nodes=nodes), config_file)

    def add(self, node_url: str, status: Optional[Status] = None):
        """Add a node to the manager.

        Args:
            node_url (str): A http url. Can be the url generated by
                `lmdeploy serve api_server`.
            description (Dict): The description of the node. An example:
                {'http://0.0.0.0:23333': {models: ['internlm-chat-7b]},
                speed: -1}. The speed here can be RPM or other metric. All the
                values of nodes should be the same metric.
        """
        if status is None:
            status = self.nodes.get(node_url, Status())
        if status.models != []:  # force register directly
            self.nodes[node_url] = status
            self.update_config_file()
            return
        try:
            from lmdeploy.serve.openai.api_client import APIClient
            client = APIClient(api_server_url=node_url)
            status.models = client.available_models
            self.nodes[node_url] = status
        except requests.exceptions.RequestException as e:  # noqa
            return self.handle_api_timeout(node_url)
        self.update_config_file()

    def remove(self, node_url: str):
        """Remove a node."""
        if node_url in self.nodes.keys():
            self.nodes.pop(node_url)
            self.update_config_file()

    def remove_stale_nodes_by_expiration(self):
        """remove stale nodes."""
        to_be_deleted = []
        for node_url in self.nodes.keys():
            url = f'{node_url}/health'
            headers = {'accept': 'application/json'}
            try:
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    to_be_deleted.append(node_url)
            except:  # noqa
                to_be_deleted.append(node_url)
        for node_url in to_be_deleted:
            self.remove(node_url)
            logger.info(f'Removed node_url: {node_url}')

    @property
    def model_list(self):
        """Supported model list."""
        model_names = []
        for node_url, node_status in self.nodes.items():
            model_names.extend(node_status.models)
        return model_names

    @property
    def status(self):
        """Return the status."""
        return self.nodes

    def get_node_url(self, model_name: str):
        """Add a node to the manager.

        Args:
            model_name (str): A http url. Can be the url generated by
                `lmdeploy serve api_server`.
        Return:
            A node url or None.
        """

        def get_matched_urls():
            urls_with_speeds, speeds, urls_without_speeds = [], [], []
            for node_url, node_status in self.nodes.items():
                if model_name in node_status.models:
                    if node_status.speed is not None:
                        urls_with_speeds.append(node_url)
                        speeds.append(node_status.speed)
                    else:
                        urls_without_speeds.append(node_url)
            all_matched_urls = urls_with_speeds + urls_without_speeds
            if len(all_matched_urls) == 0:
                return None
            # some nodes does not contain speed
            # we can set them the average speed value
            average_speed = sum(speeds) / len(speeds) if len(speeds) else 1
            all_the_speeds = speeds + [average_speed
                                       ] * len(urls_without_speeds)
            return all_matched_urls, all_the_speeds

        if self.strategy == Strategy.RANDOM:
            all_matched_urls, all_the_speeds = get_matched_urls()
            if len(all_matched_urls) == 0:
                return None
            speed_sum = sum(all_the_speeds)
            weights = [speed / speed_sum for speed in all_the_speeds]
            index = random.choices(range(len(all_matched_urls)),
                                   weights=weights)[0]
            url = all_matched_urls[index]
            return url
        elif self.strategy == Strategy.MIN_EXPECTED_LATENCY:
            all_matched_urls, all_the_speeds = get_matched_urls()
            if len(all_matched_urls) == 0:
                return None
            min_latency = float('inf')
            min_index = 0
            # random traverse nodes for low concurrency situation
            all_indexes = [i for i in range(len(all_the_speeds))]
            random.shuffle(all_indexes)
            for index in all_indexes:
                latency = self.nodes[
                    all_matched_urls[index]].unfinished / all_the_speeds[index]
                if min_latency > latency:
                    min_latency = latency
                    min_index = index
            url = all_matched_urls[min_index]
            return url
        elif self.strategy == Strategy.MIN_OBSERVED_LATENCY:
            all_matched_urls, latencies = [], []
            for node_url, node_status in self.nodes.items():
                if model_name in node_status.models:
                    if len(node_status.latency):
                        latencies.append(np.mean(np.array(
                            node_status.latency)))
                    else:
                        latencies.append(float('inf'))
                    all_matched_urls.append(node_url)
            if len(all_matched_urls) == 0:
                return None
            index = np.argmin(np.array(latencies))
            return all_matched_urls[index]
        else:
            raise ValueError(f'Invalid strategy: {self.strategy}')

    async def check_request_model(self, model_name) -> Optional[JSONResponse]:
        """Check if a request is valid."""
        if model_name in self.model_list:
            return
        ret = create_error_response(
            HTTPStatus.NOT_FOUND, f'The model `{model_name}` does not exist.')
        return ret

    def handle_unavailable_model(self, model_name):
        """Handle unavailable model.

        Args:
            model_name (str): the model in the request.
        """
        logger.warn(f'no model name: {model_name}')
        ret = {
            'error_code': ErrorCodes.MODEL_NOT_FOUND,
            'text': err_msg[ErrorCodes.MODEL_NOT_FOUND],
        }
        return json.dumps(ret).encode() + b'\n'

    def handle_api_timeout(self, node_url):
        """Handle the api time out."""
        logger.warn(f'api timeout: {node_url}')
        ret = {
            'error_code': ErrorCodes.API_TIMEOUT,
            'text': err_msg[ErrorCodes.API_TIMEOUT],
        }
        # exception happened, reduce unfinished num
        self.nodes[node_url].unfinished -= 1
        return json.dumps(ret).encode() + b'\n'

    def stream_generate(self, request: Dict, node_url: str, node_path: str):
        """Return a generator to handle the input request.

        Args:
            request (Dict): the input request.
            node_url (str): the node url.
            node_path (str): the node path. Such as `/v1/chat/completions`.
        """
        try:
            response = requests.post(
                node_url + node_path,
                json=request,
                stream=request['stream'],
                timeout=API_TIMEOUT_LEN,
            )
            for chunk in response.iter_lines(decode_unicode=False,
                                             delimiter=b'\n'):
                if chunk:
                    yield chunk + b'\n\n'
        except requests.exceptions.RequestException as e:  # noqa
            yield self.handle_api_timeout(node_url)

    async def generate(self, request: Dict, node_url: str, node_path: str):
        """Return a the response of the input request.

        Args:
            request (Dict): the input request.
            node_url (str): the node url.
            node_path (str): the node path. Such as `/v1/chat/completions`.
        """
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(node_url + node_path,
                                             json=request,
                                             timeout=API_TIMEOUT_LEN)
                return response.text
        except requests.exceptions.RequestException as e:  # noqa
            return self.handle_api_timeout(node_url)

    def pre_call(self, node_url):
        """Preprocess before the request get processed.

        Args:
            node_url (str): the node url.
        """
        self.nodes[node_url].unfinished += 1
        return time.time()

    def post_call(self, node_url: str, start: int):
        """Post process after the response finished.

        Args:
            node_url (str): the node url.
            start (int): the start time point. time.time()
        """
        self.nodes[node_url].unfinished -= 1
        self.nodes[node_url].latency.append(time.time() - start)

    def create_background_tasks(self, url: str, start: int):
        """To create a background task.

        Args:
            node_url (str): the node url.
            start (int): the start time point. time.time()
        """
        background_tasks = BackgroundTasks()
        background_tasks.add_task(self.post_call, url, start)
        return background_tasks


app = FastAPI(docs_url='/')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
node_manager = NodeManager()


@app.get('/v1/models', dependencies=[Depends(check_api_key)])
def available_models():
    """Show available models."""
    model_cards = []
    for model_name in node_manager.model_list:
        model_cards.append(
            ModelCard(id=model_name,
                      root=model_name,
                      permission=[ModelPermission()]))
    return ModelList(data=model_cards)


@app.get('/nodes/status', dependencies=[Depends(check_api_key)])
def node_status():
    """Show nodes status."""
    try:
        return node_manager.status
    except:  # noqa
        return False


@app.post('/nodes/add', dependencies=[Depends(check_api_key)])
def add_node(node: Node, raw_request: Request = None):
    """Add a node to the manager.

    - url (str): A http url. Can be the url generated by
        `lmdeploy serve api_server`.
    - status (Dict): The description of the node. An example:
        {models: ['internlm-chat-7b],  speed: 1}. The speed here can be
        RPM or other metric. All the values of nodes should be the same metric.
    """
    try:
        node_manager.add(node.url, node.status)
        return 'Added successfully'
    except:  # noqa
        return 'Failed to add, please check the input url.'


@app.post('/nodes/remove', dependencies=[Depends(check_api_key)])
def remove_node(node_url: str):
    """Show available models."""
    try:
        node_manager.remove(node_url)
        return 'Deleted successfully'
    except:  # noqa
        return 'Failed to delete, please check the input url.'


@app.post('/v1/chat/completions', dependencies=[Depends(check_api_key)])
async def chat_completions_v1(request: ChatCompletionRequest,
                              raw_request: Request = None):
    """Completion API similar to OpenAI's API.

    Refer to  `https://platform.openai.com/docs/api-reference/chat/create`
    for the API specification.

    The request should be a JSON object with the following fields:
    - model: model name. Available from /v1/models.
    - messages: string prompt or chat history in OpenAI format. A example
        for chat history is `[{"role": "user", "content":"knock knock"}]`.
    - temperature (float): to modulate the next token probability
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - n (int): How many chat completion choices to generate for each input
        message. Only support one here.
    - stream: whether to stream the results or not. Default to false.
    - max_tokens (int): output token nums
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.

    Additional arguments supported by LMDeploy:
    - ignore_eos (bool): indicator for ignoring eos
    - session_id (int): if not specified, will set random value

    Currently we do not support the following features:
    - function_call (Users should implement this by themselves)
    - logit_bias (not supported yet)
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    check_response = await node_manager.check_request_model(request.model)
    if check_response is not None:
        return check_response
    node_url = node_manager.get_node_url(request.model)
    if not node_url:
        return node_manager.handle_unavailable_model(request.model)

    request_dict = request.model_dump()
    start = node_manager.pre_call(node_url)
    if request.stream is True:
        response = node_manager.stream_generate(request_dict, node_url,
                                                '/v1/chat/completions')
        background_task = node_manager.create_background_tasks(node_url, start)
        return StreamingResponse(response, background=background_task)
    else:
        response = await node_manager.generate(request_dict, node_url,
                                               '/v1/chat/completions')
        node_manager.post_call(node_url, start)
        return JSONResponse(json.loads(response))


@app.post('/v1/completions', dependencies=[Depends(check_api_key)])
async def completions_v1(request: CompletionRequest,
                         raw_request: Request = None):
    """Completion API similar to OpenAI's API.

    Go to `https://platform.openai.com/docs/api-reference/completions/create`
    for the API specification.

    The request should be a JSON object with the following fields:
    - model (str): model name. Available from /v1/models.
    - prompt (str): the input prompt.
    - suffix (str): The suffix that comes after a completion of inserted text.
    - max_tokens (int): output token nums
    - temperature (float): to modulate the next token probability
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - n (int): How many chat completion choices to generate for each input
        message. Only support one here.
    - stream: whether to stream the results or not. Default to false.
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - user (str): A unique identifier representing your end-user.
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.

    Additional arguments supported by LMDeploy:
    - ignore_eos (bool): indicator for ignoring eos
    - session_id (int): if not specified, will set random value
    - top_k (int): The number of the highest probability vocabulary
        tokens to keep for top-k-filtering

    Currently we do not support the following features:
    - logprobs (not supported yet)
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    check_response = await node_manager.check_request_model(request.model)
    if check_response is not None:
        return check_response
    node_url = node_manager.get_node_url(request.model)
    if not node_url:
        return node_manager.handle_unavailable_model(request.model)

    request_dict = request.model_dump()
    start = node_manager.pre_call(node_url)
    if request.stream is True:
        response = node_manager.stream_generate(request_dict, node_url,
                                                '/v1/completions')
        background_task = node_manager.create_background_tasks(node_url, start)
        return StreamingResponse(response, background=background_task)
    else:
        response = await node_manager.generate(request_dict, node_url,
                                               '/v1/completions')
        node_manager.post_call(node_url, start)
        return JSONResponse(json.loads(response))


def proxy(server_name: str = '0.0.0.0',
          server_port: int = 8000,
          strategy: Literal['random', 'min_expected_latency',
                            'min_observed_latency'] = 'min_expected_latency',
          api_keys: Optional[Union[List[str], str]] = None,
          ssl: bool = False,
          log_level: str = 'INFO',
          **kwargs):
    """To launch the proxy server.

    Args:
        server_name (str): the server name of the proxy. Default to '0.0.0.0'.
        server_port (str): the server port. Default to 8000.
        strategy ('random' | 'min_expected_latency' | 'min_observed_latency'):
            the strategy to dispatch requests to nodes. Default to
            'min_expected_latency'
        api_keys (List[str] | str | None): Optional list of API keys. Accepts string type as
            a single api_key. Default to None, which means no api key applied.
        ssl (bool): Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.
    """  # noqa
    node_manager.strategy = Strategy.from_str(strategy)
    if api_keys is not None:
        if isinstance(api_keys, str):
            api_keys = api_keys.split(',')
        from lmdeploy.serve.openai.api_server import VariableInterface
        VariableInterface.api_keys = api_keys
    ssl_keyfile, ssl_certfile = None, None
    if ssl:
        ssl_keyfile = os.environ['SSL_KEYFILE']
        ssl_certfile = os.environ['SSL_CERTFILE']
    logger.setLevel(log_level)
    uvicorn.run(app=app,
                host=server_name,
                port=server_port,
                log_level='info',
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile)


if __name__ == '__main__':
    import fire

    fire.Fire(proxy)

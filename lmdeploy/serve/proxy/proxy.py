# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
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

import aiohttp
import numpy as np
import requests
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from lmdeploy.pytorch.disagg.config import DistServeRDMAConfig, EngineRole, RDMALinkType, ServingStrategy
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest
from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.pytorch.disagg.messages import PDConnectionMessage
from lmdeploy.serve.openai.api_server import check_api_key, create_error_response
from lmdeploy.serve.openai.protocol import ModelCard  # noqa: E501
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest, ModelList, ModelPermission
from lmdeploy.serve.proxy.constants import AIOHTTP_TIMEOUT, LATENCY_DEQUE_LEN, ErrorCodes, RoutingStrategy, err_msg
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class Status(BaseModel):
    """Status protocol consists of models' information."""
    role: EngineRole = EngineRole.Hybrid
    models: Optional[List[str]] = Field(default=[], examples=[[]])
    unfinished: int = 0
    latency: Deque = Field(default=deque(maxlen=LATENCY_DEQUE_LEN), examples=[[]])
    speed: Optional[int] = Field(default=None, examples=[None])


class Node(BaseModel):
    """Node protocol consists of url and status."""
    url: str
    status: Optional[Status] = None


CONTROLLER_HEART_BEAT_EXPIRATION = int(os.getenv('LMDEPLOY_CONTROLLER_HEART_BEAT_EXPIRATION', 90))


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
                 serving_strategy: str = 'Hybrid',
                 routing_strategy: str = 'min_expected_latency',
                 migration_protocol: str = 'RDMA',
                 link_type: str = 'RoCE',
                 with_gdr: bool = True,
                 cache_status: Optional[bool] = True) -> None:
        self.nodes = dict()
        self.serving_strategy = ServingStrategy[serving_strategy]
        self.routing_strategy = RoutingStrategy.from_str(routing_strategy)

        self.cache_status = cache_status
        self.latencies = dict()
        self.config_path = osp.join(osp.dirname(osp.realpath(__file__)), 'proxy_config.json')
        if config_path is not None:
            self.config_path = config_path
        if osp.exists(self.config_path) and self.cache_status:
            with open(self.config_path, 'r') as config_file:
                if os.path.getsize(self.config_path) > 0:
                    logger.info(f'loading node configuration: {self.config_path}')
                    config = json.load(config_file)
                    self.nodes = {
                        node_url: Status.model_validate_json(node_status)
                        for node_url, node_status in config.items()
                    }
        self.heart_beat_thread = threading.Thread(target=heart_beat_controller, args=(self, ), daemon=True)
        self.heart_beat_thread.start()
        self.aiotimeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)

        # For PD Disaggregation
        self.migration_protocol = MigrationProtocol[migration_protocol]
        self.rdma_config = DistServeRDMAConfig(with_gdr=with_gdr, link_type=RDMALinkType[link_type])
        self.pd_connection_pool = PDConnectionPool()
        self.dummy_prefill = False

    def get_nodes(self, role: EngineRole) -> Dict:
        items = list(self.nodes.items())
        return {node_url: node_status for (node_url, node_status) in items if node_status.role == role}

    @property
    def hybrid_nodes(self):
        return self.get_nodes(EngineRole.Hybrid)

    @property
    def prefill_nodes(self):
        return self.get_nodes(EngineRole.Prefill)

    @property
    def decode_nodes(self):
        return self.get_nodes(EngineRole.Decode)

    def update_config_file(self):
        """Update the config file."""
        nodes = copy.deepcopy(self.nodes)
        for _, status in nodes.items():
            status.latency = deque(list(status.latency)[-LATENCY_DEQUE_LEN:])
        if self.cache_status:
            with open(self.config_path, 'w') as config_file:  # update cfg yml
                json.dump({
                    node_url: node_status.model_dump_json()
                    for node_url, node_status in nodes.items()
                },
                          config_file,
                          indent=2)

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
            self.remove(node_url)
            self.nodes[node_url] = status
            self.update_config_file()
            return
        try:
            from lmdeploy.serve.openai.api_client import APIClient
            client = APIClient(api_server_url=node_url)
            status.models = client.available_models
            self.nodes[node_url] = status
        except requests.exceptions.RequestException as e:  # noqa
            logger.error(f'exception happened when adding node {node_url}, {e}')
            return self.handle_api_timeout(node_url)
        self.update_config_file()

    def remove(self, node_url: str):
        """Remove a node."""
        if node_url in self.nodes.keys():
            self.nodes.pop(node_url)
            self.update_config_file()
            self.pd_connection_pool.dereg_instance(node_url)

    def terminate_node(self, node_url: str):
        """Terminate a node."""
        success = True
        if node_url in self.nodes:
            self.nodes.pop(node_url)
            headers = {'accept': 'application/json'}
            try:
                response = requests.get(f'{node_url}/terminate', headers=headers)
                if response.status_code != 200:
                    success = False
                    logger.error(f'Failed to terminate node {node_url}, '
                                 f'error_code={response.status_code}, '
                                 f'error_msg={response.text}')
            except Exception as e:  # noqa
                logger.error(f'exception happened when terminating node {node_url}, {e}')
                success = False
        else:
            logger.error(f'terminating node {node_url} failed since it does not exist. '
                         'May try /nodes/status to check the node list')
            success = False
        self.update_config_file()
        return success

    def terminate_all_nodes(self):
        """Terminate all nodes."""
        node_url_li = list(self.nodes.keys())
        all_success = True
        for node_url in node_url_li:
            if not self.terminate_node(node_url):
                all_success = False
        return all_success

    def remove_stale_nodes_by_expiration(self):
        """Remove stale nodes."""
        to_be_deleted = []
        node_urls = list(self.nodes.keys())
        for node_url in node_urls:
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
            logger.info(f'Removed node_url: {node_url} '
                        'due to heart beat expiration')

    @property
    def model_list(self):
        """Supported model list."""
        model_names = []
        items = list(self.nodes.items())
        for _, status in items:
            model_names.extend(status.models)
        return model_names

    @property
    def status(self):
        """Return the status."""
        return self.nodes

    def get_node_url(self, model_name: str, role: EngineRole = EngineRole.Hybrid):
        """Add a node to the manager.

        Args:
            model_name (str): A http url. Can be the url generated by
                `lmdeploy serve api_server`.
        Return:
            A node url or None.
        """

        def get_matched_urls():
            urls_with_speeds, speeds, urls_without_speeds = [], [], []
            for node_url, status in self.get_nodes(role).items():
                if model_name in status.models:
                    if status.speed is not None:
                        urls_with_speeds.append(node_url)
                        speeds.append(status.speed)
                    else:
                        urls_without_speeds.append(node_url)
            all_matched_urls = urls_with_speeds + urls_without_speeds
            if len(all_matched_urls) == 0:
                return None
            # some nodes does not contain speed
            # we can set them the average speed value
            average_speed = sum(speeds) / len(speeds) if len(speeds) else 1
            all_the_speeds = speeds + [average_speed] * len(urls_without_speeds)
            return all_matched_urls, all_the_speeds

        if self.routing_strategy == RoutingStrategy.RANDOM:
            all_matched_urls, all_the_speeds = get_matched_urls()
            if len(all_matched_urls) == 0:
                return None
            speed_sum = sum(all_the_speeds)
            weights = [speed / speed_sum for speed in all_the_speeds]
            index = random.choices(range(len(all_matched_urls)), weights=weights)[0]
            url = all_matched_urls[index]
            return url
        elif self.routing_strategy == RoutingStrategy.MIN_EXPECTED_LATENCY:
            all_matched_urls, all_the_speeds = get_matched_urls()
            if len(all_matched_urls) == 0:
                return None
            min_latency = float('inf')
            min_index = 0
            # random traverse nodes for low concurrency situation
            all_indexes = [i for i in range(len(all_the_speeds))]
            random.shuffle(all_indexes)
            for index in all_indexes:
                latency = self.get_nodes(role)[all_matched_urls[index]].unfinished / all_the_speeds[index]
                if min_latency > latency:
                    min_latency = latency
                    min_index = index
            url = all_matched_urls[min_index]
            return url
        elif self.routing_strategy == RoutingStrategy.MIN_OBSERVED_LATENCY:
            all_matched_urls, latencies = [], []
            for node_url, node_status in self.get_nodes(role).items():
                if model_name in node_status.models:
                    if len(node_status.latency):
                        latencies.append(np.mean(np.array(node_status.latency)))
                    else:
                        latencies.append(float('inf'))
                    all_matched_urls.append(node_url)
            if len(all_matched_urls) == 0:
                return None
            index = np.argmin(np.array(latencies))
            return all_matched_urls[index]
        else:
            raise ValueError(f'Invalid strategy: {self.routing_strategy}')

    async def check_request_model(self, model_name) -> Optional[JSONResponse]:
        """Check if a request is valid."""
        if model_name in self.model_list:
            return
        ret = create_error_response(HTTPStatus.NOT_FOUND, f'The model {model_name!r} does not exist.')
        return ret

    def handle_unavailable_model(self, model_name):
        """Handle unavailable model.

        Args:
            model_name (str): the model in the request.
        """
        logger.warning(f'no model name: {model_name}')
        ret = {
            'error_code': ErrorCodes.MODEL_NOT_FOUND,
            'text': err_msg[ErrorCodes.MODEL_NOT_FOUND],
        }
        return json.dumps(ret).encode() + b'\n'

    def handle_api_timeout(self, node_url):
        """Handle the api time out."""
        logger.warning(f'api timeout: {node_url}')
        ret = {
            'error_code': ErrorCodes.API_TIMEOUT.value,
            'text': err_msg[ErrorCodes.API_TIMEOUT],
        }
        return json.dumps(ret).encode() + b'\n'

    async def stream_generate(self, request: Dict, node_url: str, endpoint: str):
        """Return a generator to handle the input request.

        Args:
            request (Dict): the input request.
            node_url (str): the node url.
            endpoint (str): the endpoint. Such as `/v1/chat/completions`.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(node_url + endpoint, json=request, timeout=self.aiotimeout) as response:
                    async for line in response.content:
                        if line.strip():
                            yield line + b'\n\n'
        except (Exception, GeneratorExit, aiohttp.ClientError) as e:  # noqa
            logger.error(f'catched an exception: {e}')
            # exception happened, reduce unfinished num
            yield self.handle_api_timeout(node_url)

    async def generate(self, request: Dict, node_url: str, endpoint: str):
        """Return a the response of the input request.

        Args:
            request (Dict): the input request.
            node_url (str): the node url.
            endpoint (str): the endpoint. Such as `/v1/chat/completions`.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(node_url + endpoint, json=request, timeout=self.aiotimeout) as response:
                    return await response.text()
        except (Exception, GeneratorExit, aiohttp.ClientError, asyncio.CancelledError) as e:  # noqa  # yapf: disable
            logger.error(f'catched an exception: {e}')
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
        if node_url in self.nodes:
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
        model_cards.append(ModelCard(id=model_name, root=model_name, permission=[ModelPermission()]))
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
        res = node_manager.add(node.url, node.status)
        if res is not None:
            logger.error(f'add node {node.url} failed, {res}')
            return res
        logger.info(f'add node {node.url} successfully')
        return 'Added successfully'
    except:  # noqa
        return 'Failed to add, please check the input url.'


@app.post('/nodes/remove', dependencies=[Depends(check_api_key)])
def remove_node(node: Node):
    """Show available models."""
    try:
        node_url = node.url
        node_manager.remove(node_url)
        logger.info(f'delete node {node_url} successfully')
        return 'Deleted successfully'
    except:  # noqa
        logger.error(f'delete node {node.url} failed.')
        return 'Failed to delete, please check the input url.'


@app.post('/nodes/terminate', dependencies=[Depends(check_api_key)])
def terminate_node(node: Node):
    """Terminate nodes."""
    try:
        node_url = node.url
        success = node_manager.terminate_node(node_url)
        if not success:
            return f'Failed to terminate node {node_url}'
        return 'Terminated successfully'
    except:  # noqa
        logger.error(f'Terminate node {node_url} failed.')
        return 'Failed to terminate node {node_url}, please check the input url.'


@app.get('/nodes/terminate_all', dependencies=[Depends(check_api_key)])
def terminate_node_all():
    """Terminate nodes."""
    try:
        success = node_manager.terminate_all_nodes()
        if not success:
            return 'Failed to terminate all nodes'
        return 'All nodes terminated successfully'
    except:  # noqa
        logger.error('Failed to terminate all nodes')
        return 'Failed to terminate all nodes.'


@app.post('/distserve/connection_warmup')
async def connection_warmup():
    await asyncio.gather(*[
        node_manager.pd_connection_pool.connect(
            PDConnectionMessage(
                p_url=p_url,
                d_url=d_url,
                protocol=node_manager.migration_protocol,
                rdma_config=node_manager.rdma_config,
            )) for p_url in node_manager.prefill_nodes for d_url in node_manager.decode_nodes
    ])
    return JSONResponse({'SUCCESS': True})


@app.post('/distserve/gc')
async def cache_block_gc_to_be_migrated():
    # TODO (JimyMa): add garbage collection of to be migrated request
    raise NotImplementedError


@app.post('/v1/chat/completions', dependencies=[Depends(check_api_key)])
async def chat_completions_v1(request: ChatCompletionRequest, raw_request: Request = None):
    """Completion API similar to OpenAI's API.

    Refer to  `https://platform.openai.com/docs/api-reference/chat/create`
    for the API specification.

    The request should be a JSON object with the following fields:
    - model: model name. Available from /v1/models.
    - messages: string prompt or chat history in OpenAI format. Chat history
        example: `[{"role": "user", "content": "hi"}]`.
    - temperature (float): to modulate the next token probability
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - n (int): How many chat completion choices to generate for each input
        message. **Only support one here**.
    - stream: whether to stream the results or not. Default to false.
    - max_completion_tokens (int | None): output token nums. Default to None.
    - max_tokens (int | None): output token nums. Default to None.
        Deprecated: Use max_completion_tokens instead.
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.
    - response_format (Dict | None): To generate response according to given
        schema. Examples: `{"type": "json_schema", "json_schema": {"name":
        "test","schema": {"properties": {"name": {"type": "string"}},
        "required": ["name"], "type": "object"}}}`
        or `{"type": "regex_schema", "regex_schema": "call me [A-Za-z]{1,10}"}`
    - logit_bias (Dict): Bias to logits. Only supported in pytorch engine.
    - tools (List): A list of tools the model may call. Currently, only
        internlm2 functions are supported as a tool. Use this to specify a
        list of functions for which the model can generate JSON inputs.
    - tool_choice (str | object): Controls which (if any) tool is called by
        the model. `none` means the model will not call any tool and instead
        generates a message. Specifying a particular tool via {"type":
        "function", "function": {"name": "my_function"}} forces the model to
        call that tool. `auto` or `required` will put all the tools information
        to the model.

    Additional arguments supported by LMDeploy:
    - top_k (int): The number of the highest probability vocabulary
        tokens to keep for top-k-filtering
    - ignore_eos (bool): indicator for ignoring eos
    - skip_special_tokens (bool): Whether or not to remove special tokens
        in the decoding. Default to be True.
    - min_new_tokens (int): To generate at least numbers of tokens.
    - min_p (float): Minimum token probability, which will be scaled by the
        probability of the most likely token. It must be a value between
        0 and 1. Typical values are in the 0.01-0.2 range, comparably
        selective as setting `top_p` in the 0.99-0.8 range (use the
        opposite of normal `top_p` values)

    Currently we do not support the following features:
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    check_response = await node_manager.check_request_model(request.model)
    if check_response is not None:
        return check_response

    if node_manager.serving_strategy == ServingStrategy.Hybrid:
        node_url = node_manager.get_node_url(request.model)
        if not node_url:
            return node_manager.handle_unavailable_model(request.model)

        logger.info(f'A request is dispatched to {node_url}')
        request_dict = request.model_dump()
        start = node_manager.pre_call(node_url)
        if request.stream is True:
            response = node_manager.stream_generate(request_dict, node_url, '/v1/chat/completions')
            background_task = node_manager.create_background_tasks(node_url, start)
            return StreamingResponse(response, background=background_task)
        else:
            response = await node_manager.generate(request_dict, node_url, '/v1/chat/completions')
            node_manager.post_call(node_url, start)
            return JSONResponse(json.loads(response))
    elif node_manager.serving_strategy == ServingStrategy.DistServe:
        request_dict = request.model_dump()

        # Prefill
        prefill_request_dict = copy.deepcopy(request_dict)
        prefill_request_dict['max_tokens'] = 1
        prefill_request_dict['stream'] = False
        prefill_request_dict['with_cache'] = True
        prefill_request_dict['preserve_cache'] = True

        prefill_info = {}
        p_url = 'dummy:dummy'
        if not node_manager.dummy_prefill:
            p_url = node_manager.get_node_url(request.model, EngineRole.Prefill)
            if not p_url:
                return node_manager.handle_unavailable_model(request.model)
            logger.info(f'A Prefill request is dispatched to {p_url}')

            start = node_manager.pre_call(p_url)
            prefill_info = json.loads(await node_manager.generate(prefill_request_dict, p_url, '/v1/chat/completions'))
            node_manager.post_call(p_url, start)

        # # Decode
        d_url = node_manager.get_node_url(request.model, EngineRole.Decode)
        if not d_url:
            return node_manager.handle_unavailable_model(request.model)
        logger.info(f'A Decode request is dispatched to {d_url}')

        if not node_manager.dummy_prefill:
            if not node_manager.pd_connection_pool.is_connected(p_url, d_url):
                await node_manager.pd_connection_pool.connect(
                    PDConnectionMessage(
                        p_url=p_url,
                        d_url=d_url,
                        protocol=node_manager.migration_protocol,
                        rdma_config=node_manager.rdma_config,
                    ))

        remote_session_id = int(prefill_info.get('id')) if prefill_info.get('id') else 0
        remote_block_ids = prefill_info.get('cache_block_ids') or []
        remote_token_id = prefill_info.get('remote_token_ids')[-1] if prefill_info.get('remote_token_ids') else 0

        request_dict['migration_request'] = MigrationRequest(
            protocol=node_manager.migration_protocol,
            remote_engine_id=p_url,
            remote_session_id=remote_session_id,
            remote_block_ids=remote_block_ids,
            remote_token_id=remote_token_id,
            is_dummy_prefill=node_manager.dummy_prefill).model_dump(mode='json')

        start = node_manager.pre_call(d_url)
        if not node_manager.dummy_prefill:
            node_manager.pd_connection_pool.shelf_prefill_session((p_url, d_url), prefill_info['id'])
        if request.stream is True:
            response = node_manager.stream_generate(request_dict, d_url, '/v1/chat/completions')
            background_task = node_manager.create_background_tasks(d_url, start)
            resp = StreamingResponse(response, background=background_task)
        else:
            response = await node_manager.generate(request_dict, d_url, '/v1/chat/completions')
            node_manager.post_call(d_url, start)
            resp = JSONResponse(json.loads(response))

        if not node_manager.dummy_prefill:
            node_manager.pd_connection_pool.unshelf_prefill_session((p_url, d_url), prefill_info['id'])

        return resp

    else:
        raise ValueError(f'No serving strategy named {node_manager.serving_strategy}')


@app.post('/v1/completions', dependencies=[Depends(check_api_key)])
async def completions_v1(request: CompletionRequest, raw_request: Request = None):
    """Completion API similar to OpenAI's API.

    Go to `https://platform.openai.com/docs/api-reference/completions/create`
    for the API specification.

    The request should be a JSON object with the following fields:
    - model (str): model name. Available from /v1/models.
    - prompt (str): the input prompt.
    - suffix (str): The suffix that comes after a completion of inserted text.
    - max_completion_tokens (int | None): output token nums. Default to None.
    - max_tokens (int): output token nums. Default to 16.
        Deprecated: Use max_completion_tokens instead.
    - temperature (float): to modulate the next token probability
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - n (int): How many chat completion choices to generate for each input
        message. **Only support one here**.
    - stream: whether to stream the results or not. Default to false.
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - user (str): A unique identifier representing your end-user.
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.

    Additional arguments supported by LMDeploy:
    - ignore_eos (bool): indicator for ignoring eos
    - skip_special_tokens (bool): Whether or not to remove special tokens
        in the decoding. Default to be True.
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
    if node_manager.serving_strategy == ServingStrategy.Hybrid:
        node_url = node_manager.get_node_url(request.model)
        if not node_url:
            return node_manager.handle_unavailable_model(request.model)

        logger.info(f'A request is dispatched to {node_url}')
        request_dict = request.model_dump()
        start = node_manager.pre_call(node_url)
        if request.stream is True:
            response = node_manager.stream_generate(request_dict, node_url, '/v1/completions')
            background_task = node_manager.create_background_tasks(node_url, start)
            return StreamingResponse(response, background=background_task)
        else:
            response = await node_manager.generate(request_dict, node_url, '/v1/completions')
            node_manager.post_call(node_url, start)
            return JSONResponse(json.loads(response))
    elif node_manager.serving_strategy == ServingStrategy.DistServe:
        request_dict = request.model_dump()

        # Prefill
        prefill_request_dict = copy.deepcopy(request_dict)
        prefill_request_dict['max_tokens'] = 1
        prefill_request_dict['stream'] = False
        prefill_request_dict['with_cache'] = True
        prefill_request_dict['preserve_cache'] = True

        if not node_manager.dummy_prefill:
            try:
                p_url = node_manager.get_node_url(request.model, EngineRole.Prefill)
            except Exception as e:
                logger.error(f'error Msg: {str(e)}')
                return {'status': 'Instance sch error, cannot find available p_url'}

            if not p_url:
                return node_manager.handle_unavailable_model(request.model)
            logger.info(f'A Prefill request is dispatched to {p_url}')

            start = node_manager.pre_call(p_url)
            prefill_info = json.loads(await node_manager.generate(prefill_request_dict, p_url, '/v1/completions'))
            node_manager.post_call(p_url, start)
        else:
            p_url = 'dummy:dummy'
            prefill_info = {}

        # Decode
        try:
            d_url = node_manager.get_node_url(request.model, EngineRole.Decode)
        except Exception as e:
            logger.error(f'error Msg: {str(e)}')
            return {'status': 'Instance sch error, cannot find available p_url'}

        if not d_url:
            return node_manager.handle_unavailable_model(request.model)
        logger.info(f'A Decode request is dispatched to {d_url}')

        if not node_manager.dummy_prefill:
            if not node_manager.pd_connection_pool.is_connected(p_url, d_url):
                try:
                    await node_manager.pd_connection_pool.connect(
                        PDConnectionMessage(
                            p_url=p_url,
                            d_url=d_url,
                            protocol=node_manager.migration_protocol,
                            rdma_config=node_manager.rdma_config,
                        ))
                except Exception as e:
                    logger.error(f'error Msg: {str(e)}')
                    return {'status': f'Connection error, cannot establish connection {(p_url, d_url)}'}
            node_manager.pd_connection_pool.shelf_prefill_session((p_url, d_url), prefill_info['id'])

        remote_session_id = int(prefill_info.get('id')) if prefill_info.get('id') else 0
        remote_block_ids = prefill_info.get('cache_block_ids') or []
        remote_token_id = prefill_info.get('remote_token_ids')[-1] if prefill_info.get('remote_token_ids') else 0
        request_dict['migration_request'] = MigrationRequest(
            protocol=node_manager.migration_protocol,
            remote_engine_id=p_url,
            remote_session_id=remote_session_id,
            remote_block_ids=remote_block_ids,
            remote_token_id=remote_token_id,
            is_dummy_prefill=node_manager.dummy_prefill).model_dump(mode='json')

        start = node_manager.pre_call(d_url)
        if not node_manager.dummy_prefill:
            node_manager.pd_connection_pool.shelf_prefill_session((p_url, d_url), prefill_info['id'])
        if request.stream is True:
            response = node_manager.stream_generate(request_dict, d_url, '/v1/completions')
            background_task = node_manager.create_background_tasks(d_url, start)
            resp = StreamingResponse(response, background=background_task)
        else:
            response = await node_manager.generate(request_dict, d_url, '/v1/completions')
            node_manager.post_call(d_url, start)
            resp = JSONResponse(json.loads(response))
        if not node_manager.dummy_prefill:
            node_manager.pd_connection_pool.unshelf_prefill_session((p_url, d_url), prefill_info.get('id'))
        return resp
    else:
        raise ValueError(f'No serving strategy named {node_manager.serving_strategy}')


def proxy(server_name: str = '0.0.0.0',
          server_port: int = 8000,
          serving_strategy: Literal['Hybrid', 'DistServe'] = 'Hybrid',
          routing_strategy: Literal['random', 'min_expected_latency', 'min_observed_latency'] = 'min_expected_latency',
          api_keys: Optional[Union[List[str], str]] = None,
          ssl: bool = False,
          log_level: str = 'INFO',
          disable_cache_status: bool = False,
          link_type: Literal['RoCE', 'IB'] = 'RoCE',
          migration_protocol: Literal['RDMA'] = 'RDMA',
          dummy_prefill: bool = False,
          **kwargs):
    """To launch the proxy server.

    Args:
        server_name (str): the server name of the proxy. Default to '0.0.0.0'.
        server_port (str): the server port. Default to 8000.
        serving_strategy ('Hybrid' | 'DistServe'):  the strategy to serving. Hybrid default.
            DistServe for PD Disaggregation.
        route_strategy ('random' | 'min_expected_latency' | 'min_observed_latency'):
            the strategy to dispatch requests to nodes. Default to
            'min_expected_latency'
        api_keys (List[str] | str | None): Optional list of API keys. Accepts string type as
            a single api_key. Default to None, which means no api key applied.
        ssl (bool): Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.
        log_level (str): Set the log level. Default to INFO.
        disable_cache_status (str): Whether to cache the proxy status to
             proxy_config.yml.
        migration_protocol: migration protocol when PD disaggregation. RDMA default.
    """  # noqa
    node_manager.serving_strategy = ServingStrategy[serving_strategy]
    node_manager.routing_strategy = RoutingStrategy.from_str(routing_strategy)
    node_manager.migration_protocol = MigrationProtocol[migration_protocol]
    node_manager.dummy_prefill = dummy_prefill

    node_manager.rdma_config = DistServeRDMAConfig(
        link_type=RDMALinkType[link_type],
        with_gdr=True,
    )
    node_manager.cache_status = not disable_cache_status
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
    uvicorn_log_level = os.getenv('UVICORN_LOG_LEVEL', 'info').lower()
    uvicorn.run(app=app,
                host=server_name,
                port=server_port,
                log_level=uvicorn_log_level,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile)


if __name__ == '__main__':
    import fire

    fire.Fire(proxy)

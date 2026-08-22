# Copyright (c) OpenMMLab. All rights reserved.
# yapf: disable
from __future__ import annotations

import asyncio
import os
import re
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:

    from lmdeploy.serve.parsers import ResponseParser

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Mount

from lmdeploy.archs import get_task
from lmdeploy.messages import (
    PytorchEngineConfig,
    SpeculativeConfig,
    TurbomindEngineConfig,
)
from lmdeploy.metrics.metrics_processor import metrics_processor
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.serve.anthropic import create_anthropic_router
from lmdeploy.serve.core import AsyncEngine, EngineHealthMonitor
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.core.generation_config import (
    resolve_default_gen_config,
)
from lmdeploy.serve.openai.endpoints import create_openai_router
from lmdeploy.serve.openai.responses import create_responses_router
from lmdeploy.serve.openai.utils import get_model_list
from lmdeploy.serve.utils.server_utils import (
    AuthenticationMiddleware,
    EngineSleepingMiddleware,
    protocol_error_response,
)
from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from lmdeploy.serve.managers import Session

# yapf: enable

logger = get_logger('lmdeploy')


class ServerContext:
    """Runtime dependencies and state shared by serving endpoints."""

    def __init__(self):
        self.async_engine: AsyncEngine | None = None
        self.health_monitor: EngineHealthMonitor | None = None
        self.proxy_url: str | None = None
        self.api_server_url: str | None = None
        self.allow_terminate_by_client = False
        self.enable_abort_handling = False
        self.response_parser_cls: type[ResponseParser] | None = None
        self.default_gen_config: dict = {}

    @property
    def engine_config(self):
        return self.async_engine.backend_config

    @property
    def session_manager(self):
        return self.async_engine.session_mgr

    def create_session(self, user_session_id: int | None = None) -> Session:
        session_mgr = self.session_manager
        if user_session_id is None or user_session_id == -1:
            # user doesn't input session_id, so we need to generate a new one
            session = session_mgr.get()
        else:
            # find the inside session_id by user_session_id, create a new one
            # if it doesn't exist and update the user_session_id_map
            session_id = session_mgr.map_user_session_id(user_session_id)
            session = session_mgr.get(session_id)
        # Stamp epoch for ``stop_all_session`` / ``abort_all`` coordination in ``AsyncEngine.generate``.
        session.epoch = self.async_engine.epoch
        return session

    def find_session(self, user_session_id: int) -> Session | None:
        """Find the session by user_session_id.

        Users cannot access inner session_id directly.
        """
        session_mgr = self.session_manager
        session_id = session_mgr.user_session_id_map.get(user_session_id, None)
        if session_id is None:
            return None
        return session_mgr.get(session_id, create_if_not_exists=False)

def handle_torchrun():
    """To disable mmengine logging logic when using torchrun."""

    def dummy_get_device_id():
        return 0

    if int(os.environ.get('LOCAL_RANK', -1)) > 0:
        from lmdeploy.vl.model.utils import _set_func

        # the replacement can't be recovered
        _set_func('mmengine.logging.logger._get_device_id',
                  dummy_get_device_id)


def register_to_proxy(context: ServerContext):
    """Register the API server with its proxy, when configured."""
    async_engine = context.async_engine
    if context.proxy_url is None:
        return
    elif getattr(async_engine.engine, 'is_dummy', False):
        logger.info('Dummy node started')
        return
    try:
        import requests
        engine_config = context.engine_config
        engine_role = engine_config.role.value if hasattr(
            engine_config, 'role') else 1
        url = f'{context.proxy_url}/nodes/add'
        data = {
            'url': context.api_server_url,
            'status': {
                'models': get_model_list(context),
                'role': engine_role
            }
        }
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code,
                                detail=response.text)
    except Exception as e:
        logger.error(f'Service registration failed: {e}')


async def validation_exception_handler(request: Request,
                                       exc: RequestValidationError):
    """Handler for RequestValidationError."""
    first_error = next(iter(exc.errors()), {})
    if not isinstance(first_error, dict):
        first_error = {'msg': str(first_error)}
    location = '.'.join(str(part) for part in first_error.get('loc', ())
                        if part != 'body')
    detail = first_error.get('msg', 'Invalid request body.')
    message = f'{location}: {detail}' if location else detail
    error = RequestError(ErrorCode.INVALID_REQUEST, message)
    return protocol_error_response(request.url.path, error)


class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: FastAPI, max_concurrent_requests: int):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def dispatch(self, request: Request, call_next):
        async with self.semaphore:
            response = await call_next(request)
            return response


def set_parsers(context: ServerContext,
                reasoning_parser_name: str | None = None,
                tool_parser_name: str | None = None,
                **kwargs):
    from lmdeploy.serve.parsers import ResponseParserManager
    name = 'default'
    arch = context.async_engine.arch
    if arch == 'GptOssForCausalLM':
        name = 'gpt-oss'
    elif arch == 'MuseGlimmerForConditionalGeneration':
        name = 'muse-glimmer'
    cls = ResponseParserManager.get(name)
    if cls is None:
        raise ValueError(
            f'The response parser {name} is not in the parser list: '
            f'{ResponseParserManager.module_dict.keys()}')
    tokenizer = context.async_engine.tokenizer.model.model
    cls.set_parsers(reasoning_parser_name=reasoning_parser_name,
                    tool_parser_name=tool_parser_name,
                    tokenizer=tokenizer)
    context.response_parser_cls = cls


def mount_metrics(app: FastAPI,
                  backend_config: PytorchEngineConfig | TurbomindEngineConfig):
    if not getattr(backend_config, 'enable_metrics', False):
        return

    from prometheus_client import REGISTRY, make_asgi_app
    registry = REGISTRY

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount('/metrics', make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile('^/metrics(?P<path>.*)$')
    app.routes.append(metrics_route)


def create_lifespan_handler(backend_config: PytorchEngineConfig
                            | TurbomindEngineConfig,
                            async_engine: AsyncEngine,
                            context: ServerContext):
    """Factory function to create a lifespan handler."""

    @asynccontextmanager
    async def lifespan_handler(app: FastAPI):
        task = None
        health_monitor = EngineHealthMonitor(async_engine)
        context.health_monitor = health_monitor
        try:
            async_engine.start_loop(asyncio.get_running_loop(), use_async_api=True)
            register_to_proxy(context)
            health_monitor.start()
            if getattr(backend_config, 'enable_metrics', False):
                metrics_processor.start_metrics_handler(enable_metrics=True)
                log_interval = 10.

                async def _force_log():
                    while True:
                        await asyncio.sleep(log_interval)

                        # periodically update schedule metrics, as they change less frequently than iteration stats
                        schedule_metrics = await async_engine.get_schedule_metrics(
                        )
                        await metrics_processor.update_schedule_stats(
                            schedule_metrics)

                        await async_engine.do_log_stats()

                task = asyncio.create_task(_force_log())

            yield
        finally:
            await health_monitor.stop()
            if context.health_monitor is health_monitor:
                context.health_monitor = None
            if task:
                task.cancel()
            await metrics_processor.stop_metrics_handler()
            async_engine.close()

    return lifespan_handler


def serve(model_path: str,
          model_name: str | None = None,
          backend: Literal['turbomind', 'pytorch'] = 'turbomind',
          backend_config: PytorchEngineConfig | TurbomindEngineConfig
          | None = None,
          chat_template_config: ChatTemplateConfig | None = None,
          server_name: str = '0.0.0.0',
          server_port: int = 23333,
          allow_origins: list[str] = ['*'],
          allow_credentials: bool = True,
          allow_methods: list[str] = ['*'],
          allow_headers: list[str] = ['*'],
          log_level: str = 'ERROR',
          api_keys: list[str] | str | None = None,
          ssl: bool = False,
          proxy_url: str | None = None,
          max_log_len: int | None = None,
          disable_fastapi_docs: bool = False,
          max_concurrent_requests: int | None = None,
          trust_remote_code: bool = False,
          reasoning_parser: str | None = None,
          tool_call_parser: str | None = None,
          allow_terminate_by_client: bool = False,
          enable_abort_handling: bool = False,
          speculative_config: SpeculativeConfig | None = None,
          allowed_media_domains: list[str] | None = None,
          generation_config: str = 'auto',
          **kwargs):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the path of a model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm2-chat-7b",
                    "Qwen/Qwen2.5-7B-Instruct"
                    and so on.
        model_name (str): the name of the served model. It can be accessed
            by the RESTful API `/v1/models`. If it is not specified,
            `model_path` will be adopted
        backend (str): either `turbomind` or `pytorch` backend. Default to
            `turbomind` backend.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): beckend
            config instance. Default to none.
        chat_template_config (ChatTemplateConfig): chat template configuration
            Default to None.
        server_name (str): host ip for serving
        server_port (int): server port
        tp (int): tensor parallel
        allow_origins (list[str]): a list of allowed origins for CORS
        allow_credentials (bool): whether to allow credentials for CORS
        allow_methods (list[str]): a list of allowed HTTP methods for CORS
        allow_headers (list[str]): a list of allowed HTTP headers for CORS
        log_level(str): set log level whose value among [CRITICAL, ERROR,
            WARNING, INFO, DEBUG]
        api_keys (list[str] | str | None): Optional list of API keys. Accepts
            string type as a single api_key. Default to None, which means no
            api key applied.
        ssl (bool): Enable SSL. Requires OS Environment variables
            'SSL_KEYFILE' and 'SSL_CERTFILE'.
        proxy_url (str): The proxy url to register the api_server.
        max_log_len (int): Max number of prompt characters or prompt tokens
            being printed in log. Default: Unlimited
        max_concurrent_requests: This refers to the number of concurrent
            requests that the server can handle. The server is designed to
            process the engine's tasks once the maximum number of concurrent
            requests is reached, regardless of any additional requests sent by
            clients concurrently during that time. Default to None.
        reasoning_parser (str): The reasoning parser name.
        tool_call_parser (str): The tool call parser name.
        allow_terminate_by_client (bool): Allow request from client to terminate server.
        allowed_media_domains (list[str] | None): Optional exact hostname
            allowlist for HTTP(S) media URLs.
    """
    if os.getenv('TM_LOG_LEVEL') is None:
        os.environ['TM_LOG_LEVEL'] = log_level
    logger.setLevel(log_level)

    server_context = ServerContext()
    server_context.allow_terminate_by_client = allow_terminate_by_client
    server_context.enable_abort_handling = enable_abort_handling
    from lmdeploy.serve.parsers import validate_parser_names
    reasoning_parser, tool_call_parser = validate_parser_names(
        reasoning_parser, tool_call_parser)

    server_context.default_gen_config = resolve_default_gen_config(
        generation_config,
        model_path,
        trust_remote_code,
    )

    ssl_keyfile, ssl_certfile, http_or_https = None, None, 'http'
    if ssl:
        ssl_keyfile = os.environ['SSL_KEYFILE']
        ssl_certfile = os.environ['SSL_CERTFILE']
        http_or_https = 'https'

    handle_torchrun()
    _, pipeline_class = get_task(backend,
                                 model_path,
                                 trust_remote_code=trust_remote_code,
                                 backend_config=backend_config)
    if isinstance(backend_config, PytorchEngineConfig):
        backend_config.enable_mp_engine = True
        # router replay
        if backend_config.enable_return_routed_experts:
            backend_config.enable_transfer_obj_ref = True
    server_context.async_engine = pipeline_class(
        model_path=model_path,
        model_name=model_name,
        backend=backend,
        backend_config=backend_config,
        chat_template_config=chat_template_config,
        max_log_len=max_log_len,
        trust_remote_code=trust_remote_code,
        speculative_config=speculative_config,
        allowed_media_domains=allowed_media_domains,
        **kwargs)
    set_parsers(server_context, reasoning_parser, tool_call_parser)

    # create FastAPI lifespan events
    lifespan = create_lifespan_handler(backend_config,
                                       server_context.async_engine,
                                       server_context)

    if disable_fastapi_docs:
        app = FastAPI(docs_url=None,
                      redoc_url=None,
                      openapi_url=None,
                      lifespan=lifespan)
    else:
        app = FastAPI(docs_url='/', lifespan=lifespan)

    router = create_openai_router(server_context)
    app.include_router(router)
    app.include_router(create_anthropic_router(server_context))
    app.include_router(create_responses_router(server_context))
    app.add_exception_handler(RequestValidationError,
                              validation_exception_handler)
    mount_metrics(app, backend_config)

    if allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )

    if api_keys is not None and (tokens := [key for key in api_keys if key]):
        app.add_middleware(AuthenticationMiddleware, tokens=tokens)

    def is_engine_sleeping() -> bool:
        eng = server_context.async_engine
        return eng is not None and eng.is_sleeping

    app.add_middleware(EngineSleepingMiddleware,
                       is_sleeping=is_engine_sleeping)

    # set the maximum number of concurrent requests
    if max_concurrent_requests is not None:
        app.add_middleware(ConcurrencyLimitMiddleware,
                           max_concurrent_requests=max_concurrent_requests)

    if proxy_url is not None:
        server_context.proxy_url = proxy_url
        server_context.api_server_url = f'{http_or_https}://{server_name}:{server_port}'  # noqa
    for i in range(3):
        print(
            f'HINT:    Please open \033[93m\033[1m{http_or_https}://'
            f'{server_name}:{server_port}\033[0m in a browser for detailed api'
            ' usage!!!')
    uvicorn.run(app=app,
                host=server_name,
                port=server_port,
                log_level=os.getenv('UVICORN_LOG_LEVEL', 'info').lower(),
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                timeout_keep_alive=int(
                    os.environ.get('UVICORN_TIMEOUT_KEEP_ALIVE', 5)))


if __name__ == '__main__':
    import fire

    fire.Fire(serve)

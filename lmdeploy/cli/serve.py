# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.disagg.config import EngineRole, MigrationBackend
from lmdeploy.utils import get_max_batch_size

from .cli import CLI
from .utils import ArgumentHelper, DefaultsAndTypesHelpFormatter, convert_args, get_chat_template, get_lora_adapters


class SubCliServe:
    """Serve LLMs and interact on terminal or web UI."""
    _help = 'Serve LLMs with gradio, openai API or triton server.'
    _desc = _help
    parser = CLI.subparsers.add_parser(
        'serve',
        help=_help,
        description=_desc,
    )
    subparsers = parser.add_subparsers(title='Commands', description='This group has the following commands:')

    @staticmethod
    def add_parser_gradio():
        """Add parser for gradio command."""
        parser = SubCliServe.subparsers.add_parser('gradio',
                                                   formatter_class=DefaultsAndTypesHelpFormatter,
                                                   description=SubCliServe.gradio.__doc__,
                                                   help=SubCliServe.gradio.__doc__)
        parser.set_defaults(run=SubCliServe.gradio)
        parser.add_argument('model_path_or_server',
                            type=str,
                            help='The path of the deployed model or the tritonserver url or '
                            'restful api url. for example: - ./workspace - 0.0.0.0:23333'
                            ' - http://0.0.0.0:23333')
        parser.add_argument('--server-name', type=str, default='0.0.0.0', help='The ip address of gradio server')
        parser.add_argument('--server-port', type=int, default=6006, help='The port of gradio server')
        parser.add_argument('--share',
                            action='store_true',
                            help='Whether to create a publicly shareable link'
                            ' for the app')

        # common args
        ArgumentHelper.backend(parser)
        ArgumentHelper.max_log_len(parser)

        # model args
        ArgumentHelper.revision(parser)
        ArgumentHelper.download_dir(parser)

        # chat template args
        ArgumentHelper.chat_template(parser)

        # pytorch engine args
        pt_group = parser.add_argument_group('PyTorch engine arguments')
        ArgumentHelper.device(pt_group)
        ArgumentHelper.eager_mode(pt_group)

        # common engine args
        dtype_act = ArgumentHelper.dtype(pt_group)
        tp_act = ArgumentHelper.tp(pt_group)
        session_len_act = ArgumentHelper.session_len(pt_group)
        max_batch_size_act = ArgumentHelper.max_batch_size(pt_group)
        cache_max_entry_act = ArgumentHelper.cache_max_entry_count(pt_group)
        cache_block_seq_len_act = ArgumentHelper.cache_block_seq_len(pt_group)
        prefix_caching_act = ArgumentHelper.enable_prefix_caching(pt_group)
        max_prefill_token_num_act = ArgumentHelper.max_prefill_token_num(pt_group)
        model_format_act = ArgumentHelper.model_format(pt_group)
        # turbomind args
        tb_group = parser.add_argument_group('TurboMind engine arguments')
        # common engine args
        tb_group._group_actions.append(dtype_act)
        tb_group._group_actions.append(tp_act)
        tb_group._group_actions.append(session_len_act)
        tb_group._group_actions.append(max_batch_size_act)
        tb_group._group_actions.append(cache_max_entry_act)
        tb_group._group_actions.append(cache_block_seq_len_act)
        tb_group._group_actions.append(prefix_caching_act)
        tb_group._group_actions.append(max_prefill_token_num_act)
        tb_group._group_actions.append(model_format_act)

        ArgumentHelper.quant_policy(tb_group)
        ArgumentHelper.rope_scaling_factor(tb_group)
        ArgumentHelper.communicator(tb_group)

    @staticmethod
    def add_parser_api_server():
        """Add parser for api_server command."""
        parser = SubCliServe.subparsers.add_parser('api_server',
                                                   formatter_class=DefaultsAndTypesHelpFormatter,
                                                   description=SubCliServe.api_server.__doc__,
                                                   help=SubCliServe.api_server.__doc__)
        parser.set_defaults(run=SubCliServe.api_server)
        parser.add_argument('model_path',
                            type=str,
                            help='The path of a model. it could be one of the following '
                            'options: - i) a local directory path of a turbomind model'
                            ' which is converted by `lmdeploy convert` command or '
                            'download from ii) and iii). - ii) the model_id of a '
                            'lmdeploy-quantized model hosted inside a model repo on '
                            'huggingface.co, such as "internlm/internlm-chat-20b-4bit",'
                            ' "lmdeploy/llama2-chat-70b-4bit", etc. - iii) the model_id'
                            ' of a model hosted inside a model repo on huggingface.co,'
                            ' such as "internlm/internlm-chat-7b", "qwen/qwen-7b-chat "'
                            ', "baichuan-inc/baichuan2-7b-chat" and so on')
        parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Host ip for serving')
        parser.add_argument('--server-port', type=int, default=23333, help='Server port')
        parser.add_argument('--allow-origins',
                            nargs='+',
                            type=str,
                            default=['*'],
                            help='A list of allowed origins for cors')
        parser.add_argument('--allow-credentials', action='store_true', help='Whether to allow credentials for cors')
        parser.add_argument('--allow-methods',
                            nargs='+',
                            type=str,
                            default=['*'],
                            help='A list of allowed http methods for cors')
        parser.add_argument('--allow-headers',
                            nargs='+',
                            type=str,
                            default=['*'],
                            help='A list of allowed http headers for cors')
        parser.add_argument('--proxy-url', type=str, default=None, help='The proxy url for api server.')
        parser.add_argument('--max-concurrent-requests',
                            type=int,
                            default=None,
                            help='This refers to the number of concurrent requests that '
                            'the server can handle. The server is designed to process the '
                            'engine’s tasks once the maximum number of concurrent requests is '
                            'reached, regardless of any additional requests sent by clients '
                            'concurrently during that time. Default to None.')
        # common args
        ArgumentHelper.backend(parser)
        ArgumentHelper.log_level(parser)
        ArgumentHelper.api_keys(parser)
        ArgumentHelper.ssl(parser)
        ArgumentHelper.model_name(parser)
        ArgumentHelper.max_log_len(parser)
        ArgumentHelper.disable_fastapi_docs(parser)
        ArgumentHelper.allow_terminate_by_client(parser)
        # chat template args
        ArgumentHelper.chat_template(parser)

        # parsers
        ArgumentHelper.tool_call_parser(parser)
        ArgumentHelper.reasoning_parser(parser)

        # model args
        ArgumentHelper.revision(parser)
        ArgumentHelper.download_dir(parser)

        # pytorch engine args
        pt_group = parser.add_argument_group('PyTorch engine arguments')

        ArgumentHelper.adapters(pt_group)
        ArgumentHelper.device(pt_group)
        ArgumentHelper.eager_mode(pt_group)

        # common engine args
        dtype_act = ArgumentHelper.dtype(pt_group)
        tp_act = ArgumentHelper.tp(pt_group)
        session_len_act = ArgumentHelper.session_len(pt_group)
        max_batch_size_act = ArgumentHelper.max_batch_size(pt_group)
        cache_max_entry_act = ArgumentHelper.cache_max_entry_count(pt_group)
        cache_block_seq_len_act = ArgumentHelper.cache_block_seq_len(pt_group)
        prefix_caching_act = ArgumentHelper.enable_prefix_caching(pt_group)
        max_prefill_token_num_act = ArgumentHelper.max_prefill_token_num(pt_group)
        quant_policy = ArgumentHelper.quant_policy(pt_group)
        model_format = ArgumentHelper.model_format(pt_group)
        dp_act = ArgumentHelper.dp(pt_group)
        num_nodes_act = ArgumentHelper.num_nodes(pt_group)
        ArgumentHelper.ep(pt_group)
        ArgumentHelper.enable_microbatch(pt_group)
        ArgumentHelper.enable_eplb(pt_group)
        ArgumentHelper.enable_metrics(pt_group)
        ArgumentHelper.role(pt_group)
        ArgumentHelper.migration_backend(pt_group)
        # multi-node serving args
        node_rank_act = ArgumentHelper.node_rank(pt_group)

        # turbomind args
        tb_group = parser.add_argument_group('TurboMind engine arguments')
        # common engine args
        tb_group._group_actions.append(dtype_act)
        tb_group._group_actions.append(tp_act)
        tb_group._group_actions.append(dp_act)
        tb_group._group_actions.append(session_len_act)
        tb_group._group_actions.append(max_batch_size_act)
        tb_group._group_actions.append(cache_max_entry_act)
        tb_group._group_actions.append(cache_block_seq_len_act)
        tb_group._group_actions.append(prefix_caching_act)
        tb_group._group_actions.append(max_prefill_token_num_act)
        tb_group._group_actions.append(quant_policy)
        tb_group._group_actions.append(model_format)
        tb_group._group_actions.append(num_nodes_act)
        tb_group._group_actions.append(node_rank_act)
        ArgumentHelper.rope_scaling_factor(tb_group)
        ArgumentHelper.num_tokens_per_iter(tb_group)
        ArgumentHelper.max_prefill_iters(tb_group)
        ArgumentHelper.communicator(tb_group)
        ArgumentHelper.ngpus_per_node(tb_group)

        # vlm args
        vision_group = parser.add_argument_group('Vision model arguments')
        ArgumentHelper.vision_max_batch_size(vision_group)

    @staticmethod
    def add_parser_api_client():
        """Add parser for api_client command."""
        parser = SubCliServe.subparsers.add_parser('api_client',
                                                   formatter_class=DefaultsAndTypesHelpFormatter,
                                                   description=SubCliServe.api_client.__doc__,
                                                   help=SubCliServe.api_client.__doc__)
        parser.set_defaults(run=SubCliServe.api_client)
        parser.add_argument('api_server_url', type=str, help='The URL of api server')
        parser.add_argument('--api-key',
                            type=str,
                            default=None,
                            help='api key. Default to None, which means no '
                            'api key will be used')
        ArgumentHelper.session_id(parser)

    @staticmethod
    def add_parser_proxy():
        """Add parser for proxy server command."""
        parser = SubCliServe.subparsers.add_parser('proxy',
                                                   formatter_class=DefaultsAndTypesHelpFormatter,
                                                   description=SubCliServe.proxy.__doc__,
                                                   help=SubCliServe.proxy.__doc__)
        parser.set_defaults(run=SubCliServe.proxy)
        parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Host ip for proxy serving')
        parser.add_argument('--server-port', type=int, default=8000, help='Server port of the proxy')
        parser.add_argument('--serving-strategy',
                            type=str,
                            choices=['Hybrid', 'DistServe'],
                            default='Hybrid',
                            help='the strategy to serve, Hybrid for colocating Prefill and Decode'
                            'workloads into same engine, DistServe for Prefill-Decode Disaggregation')
        parser.add_argument('--routing-strategy',
                            type=str,
                            choices=['random', 'min_expected_latency', 'min_observed_latency'],
                            default='min_expected_latency',
                            help='the strategy to dispatch requests to nodes')
        parser.add_argument('--disable-cache-status',
                            action='store_true',
                            help='Whether to disable cache status of the '
                            'proxy. If set, the proxy will forget the status '
                            'of the previous time')

        # For Disaggregation
        parser.add_argument('--migration-protocol',
                            type=str,
                            choices=['RDMA', 'NVLINK'],
                            default='RDMA',
                            help='transport protocol of KV migration')
        parser.add_argument('--link-type', type=str, choices=['RoCE', 'IB'], default='RoCE', help='RDMA Link Type')
        parser.add_argument('--disable-gdr', action='store_true', help='with GPU Direct Memory Access')
        ArgumentHelper.api_keys(parser)
        ArgumentHelper.ssl(parser)
        ArgumentHelper.log_level(parser)

    @staticmethod
    def gradio(args):
        """Serve LLMs with web UI using gradio."""

        from lmdeploy.archs import autoget_backend
        from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
        from lmdeploy.serve.gradio.app import run
        max_batch_size = args.max_batch_size if args.max_batch_size \
            else get_max_batch_size(args.device)
        backend = args.backend

        if backend != 'pytorch' and ':' not in args.model_path_or_server:
            # set auto backend mode
            backend = autoget_backend(args.model_path_or_server)
        if backend == 'pytorch':
            backend_config = PytorchEngineConfig(dtype=args.dtype,
                                                 tp=args.tp,
                                                 max_batch_size=max_batch_size,
                                                 cache_max_entry_count=args.cache_max_entry_count,
                                                 block_size=args.cache_block_seq_len,
                                                 session_len=args.session_len,
                                                 enable_prefix_caching=args.enable_prefix_caching,
                                                 device_type=args.device,
                                                 quant_policy=args.quant_policy,
                                                 eager_mode=args.eager_mode,
                                                 max_prefill_token_num=args.max_prefill_token_num,
                                                 model_format=args.model_format)
        else:
            backend_config = TurbomindEngineConfig(dtype=args.dtype,
                                                   tp=args.tp,
                                                   max_batch_size=max_batch_size,
                                                   session_len=args.session_len,
                                                   model_format=args.model_format,
                                                   quant_policy=args.quant_policy,
                                                   rope_scaling_factor=args.rope_scaling_factor,
                                                   cache_max_entry_count=args.cache_max_entry_count,
                                                   cache_block_seq_len=args.cache_block_seq_len,
                                                   enable_prefix_caching=args.enable_prefix_caching,
                                                   max_prefill_token_num=args.max_prefill_token_num,
                                                   communicator=args.communicator)
        chat_template_config = get_chat_template(args.chat_template)
        run(args.model_path_or_server,
            server_name=args.server_name,
            server_port=args.server_port,
            backend=backend,
            backend_config=backend_config,
            chat_template_config=chat_template_config,
            share=args.share,
            max_log_len=args.max_log_len)

    @staticmethod
    def api_server(args):
        """Serve LLMs with restful api using fastapi."""
        from lmdeploy.archs import autoget_backend

        max_batch_size = args.max_batch_size if args.max_batch_size \
            else get_max_batch_size(args.device)
        backend = args.backend
        if backend != 'pytorch':
            # set auto backend mode
            backend = autoget_backend(args.model_path)

        if backend == 'pytorch':
            from lmdeploy.messages import PytorchEngineConfig
            adapters = get_lora_adapters(args.adapters)
            backend_config = PytorchEngineConfig(dtype=args.dtype,
                                                 tp=args.tp,
                                                 dp=args.dp,
                                                 ep=args.ep,
                                                 max_batch_size=max_batch_size,
                                                 cache_max_entry_count=args.cache_max_entry_count,
                                                 block_size=args.cache_block_seq_len,
                                                 session_len=args.session_len,
                                                 adapters=adapters,
                                                 enable_prefix_caching=args.enable_prefix_caching,
                                                 device_type=args.device,
                                                 quant_policy=args.quant_policy,
                                                 eager_mode=args.eager_mode,
                                                 max_prefill_token_num=args.max_prefill_token_num,
                                                 enable_microbatch=args.enable_microbatch,
                                                 enable_eplb=args.enable_eplb,
                                                 enable_metrics=args.enable_metrics,
                                                 role=EngineRole[args.role],
                                                 migration_backend=MigrationBackend[args.migration_backend],
                                                 model_format=args.model_format)
        else:
            from lmdeploy.messages import TurbomindEngineConfig
            backend_config = TurbomindEngineConfig(dtype=args.dtype,
                                                   tp=args.tp,
                                                   dp=args.dp,
                                                   nnodes=args.nnodes,
                                                   ngpus_per_node=args.ngpus_per_node,
                                                   node_rank=args.node_rank,
                                                   max_batch_size=max_batch_size,
                                                   session_len=args.session_len,
                                                   model_format=args.model_format,
                                                   quant_policy=args.quant_policy,
                                                   rope_scaling_factor=args.rope_scaling_factor,
                                                   cache_max_entry_count=args.cache_max_entry_count,
                                                   cache_block_seq_len=args.cache_block_seq_len,
                                                   enable_prefix_caching=args.enable_prefix_caching,
                                                   max_prefill_token_num=args.max_prefill_token_num,
                                                   communicator=args.communicator)
        chat_template_config = get_chat_template(args.chat_template)

        from lmdeploy.messages import VisionConfig
        vision_config = VisionConfig(args.vision_max_batch_size)
        if args.dp == 1:
            from lmdeploy.serve.openai.api_server import serve as run_api_server

            run_api_server(args.model_path,
                           model_name=args.model_name,
                           backend=backend,
                           backend_config=backend_config,
                           chat_template_config=chat_template_config,
                           vision_config=vision_config,
                           server_name=args.server_name,
                           server_port=args.server_port,
                           allow_origins=args.allow_origins,
                           allow_credentials=args.allow_credentials,
                           allow_methods=args.allow_methods,
                           allow_headers=args.allow_headers,
                           allow_terminate_by_client=args.allow_terminate_by_client,
                           log_level=args.log_level.upper(),
                           api_keys=args.api_keys,
                           ssl=args.ssl,
                           proxy_url=args.proxy_url,
                           max_log_len=args.max_log_len,
                           disable_fastapi_docs=args.disable_fastapi_docs,
                           max_concurrent_requests=args.max_concurrent_requests,
                           reasoning_parser=args.reasoning_parser,
                           tool_call_parser=args.tool_call_parser)
        else:
            from lmdeploy.serve.openai.launch_server import launch_server

            launch_server(args.nnodes,
                          args.node_rank,
                          args.model_path,
                          model_name=args.model_name,
                          backend=backend,
                          backend_config=backend_config,
                          chat_template_config=chat_template_config,
                          vision_config=vision_config,
                          server_name=args.server_name,
                          server_port=args.server_port,
                          allow_origins=args.allow_origins,
                          allow_credentials=args.allow_credentials,
                          allow_methods=args.allow_methods,
                          allow_headers=args.allow_headers,
                          allow_terminate_by_client=args.allow_terminate_by_client,
                          log_level=args.log_level.upper(),
                          api_keys=args.api_keys,
                          ssl=args.ssl,
                          proxy_url=args.proxy_url,
                          max_log_len=args.max_log_len,
                          disable_fastapi_docs=args.disable_fastapi_docs,
                          max_concurrent_requests=args.max_concurrent_requests,
                          reasoning_parser=args.reasoning_parser,
                          tool_call_parser=args.tool_call_parser)

    @staticmethod
    def api_client(args):
        """Interact with restful api server in terminal."""
        from lmdeploy.serve.openai.api_client import main as run_api_client
        kwargs = convert_args(args)
        run_api_client(**kwargs)

    @staticmethod
    def proxy(args):
        """Proxy server that manages distributed api_server nodes."""
        from lmdeploy.serve.proxy.proxy import proxy
        kwargs = convert_args(args)
        proxy(**kwargs)

    @staticmethod
    def add_parsers():
        SubCliServe.add_parser_gradio()
        SubCliServe.add_parser_api_server()
        SubCliServe.add_parser_api_client()
        SubCliServe.add_parser_proxy()

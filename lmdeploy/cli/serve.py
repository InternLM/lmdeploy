# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import DictAction

from .cli import CLI
from .utils import ArgumentHelper, DefaultsAndTypesHelpFormatter, convert_args


class SubCliServe:
    """Serve LLMs and interact on terminal or web UI."""
    _help = 'Serve LLMs with gradio, openai API or triton server.'
    _desc = _help
    parser = CLI.subparsers.add_parser(
        'serve',
        help=_help,
        description=_desc,
    )
    subparsers = parser.add_subparsers(
        title='Commands', description='This group has the following commands:')

    @staticmethod
    def add_parser_gradio():
        parser = SubCliServe.subparsers.add_parser(
            'gradio',
            formatter_class=DefaultsAndTypesHelpFormatter,
            description=SubCliServe.gradio.__doc__,
            help=SubCliServe.gradio.__doc__)
        parser.set_defaults(run=SubCliServe.gradio)
        parser.add_argument(
            'model_path_or_server',
            type=str,
            help='The path of the deployed model or the tritonserver url or '
            'restful api url. for example: - ./workspace - 0.0.0.0:23333'
            ' - http://0.0.0.0:23333')
        parser.add_argument('--server-name',
                            type=str,
                            default='0.0.0.0',
                            help='The ip address of gradio server')
        parser.add_argument('--server-port',
                            type=int,
                            default=6006,
                            help='The port of gradio server')
        parser.add_argument('--batch-size',
                            type=int,
                            default=32,
                            help='Batch size for running turbomind directly')
        # common args
        ArgumentHelper.backend(parser)
        ArgumentHelper.tp(parser)
        ArgumentHelper.max_batch_size(parser)
        ArgumentHelper.meta_instruction(parser)
        ArgumentHelper.cap(parser)

        # turbomind args
        tb_group = parser.add_argument_group('TurboMind engine arguments')
        ArgumentHelper.model_format(tb_group)
        ArgumentHelper.quant_policy(tb_group)
        ArgumentHelper.rope_scaling_factor(tb_group)
        ArgumentHelper.use_logn_attn(tb_group)

        # pytorch args
        pt_group = parser.add_argument_group('PyTorch engine arguments')
        ArgumentHelper.block_size(pt_group)
        ArgumentHelper.model_name(pt_group)

    @staticmethod
    def add_parser_api_server():
        parser = SubCliServe.subparsers.add_parser(
            'api_server',
            formatter_class=DefaultsAndTypesHelpFormatter,
            description=SubCliServe.api_server.__doc__,
            help=SubCliServe.api_server.__doc__)
        parser.set_defaults(run=SubCliServe.api_server)
        parser.add_argument(
            'model_path',
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
        # common args
        ArgumentHelper.backend(parser)
        ArgumentHelper.tp(parser)
        ArgumentHelper.max_batch_size(parser)
        ArgumentHelper.log_level(parser)
        ArgumentHelper.meta_instruction(parser)
        ArgumentHelper.cap(parser)

        # turbomind args
        tb_group = parser.add_argument_group('TurboMind engine arguments')
        ArgumentHelper.model_format(tb_group)
        ArgumentHelper.quant_policy(tb_group)
        ArgumentHelper.rope_scaling_factor(tb_group)
        ArgumentHelper.use_logn_attn(tb_group)

        # pytorch args
        pt_group = parser.add_argument_group('PyTorch engine arguments')
        ArgumentHelper.block_size(pt_group)
        ArgumentHelper.model_name(pt_group)

        parser.add_argument('--server-name',
                            type=str,
                            default='0.0.0.0',
                            help='Host ip for serving')
        parser.add_argument('--server-port',
                            type=int,
                            default=23333,
                            help='Server port')
        parser.add_argument('--instance-num',
                            type=int,
                            default=64,
                            help='Number of instances of turbomind model')
        parser.add_argument('--allow-origins',
                            nargs='+',
                            type=str,
                            default=['*'],
                            help='A list of allowed origins for cors')
        parser.add_argument('--allow-credentials',
                            action='store_true',
                            help='Whether to allow credentials for cors')
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
        parser.add_argument('--qos-config-path',
                            type=str,
                            default='',
                            help='Qos policy config path')

    @staticmethod
    def add_parser_api_client():
        parser = SubCliServe.subparsers.add_parser(
            'api_client',
            formatter_class=DefaultsAndTypesHelpFormatter,
            description=SubCliServe.api_client.__doc__,
            help=SubCliServe.api_client.__doc__)
        parser.set_defaults(run=SubCliServe.api_client)
        parser.add_argument('api_server_url',
                            type=str,
                            help='The URL of api server')
        ArgumentHelper.session_id(parser)

    @staticmethod
    def add_parser_triton_client():
        parser = SubCliServe.subparsers.add_parser(
            'triton_client',
            formatter_class=DefaultsAndTypesHelpFormatter,
            description=SubCliServe.triton_client.__doc__,
            help=SubCliServe.triton_client.__doc__)
        parser.set_defaults(run=SubCliServe.triton_client)
        parser.add_argument(
            'tritonserver_addr',
            type=str,
            help='The address in format "ip:port" of triton inference server')
        ArgumentHelper.session_id(parser)
        ArgumentHelper.cap(parser)
        ArgumentHelper.stream_output(parser)
        parser.add_argument(
            '--kwargs',
            nargs='*',
            default=None,
            action=DictAction,
            help='Other key-values arguments in xxx=yyy format')

    @staticmethod
    def gradio(args):
        """Serve LLMs with web UI using gradio."""
        from lmdeploy.model import ChatTemplateConfig
        from lmdeploy.serve.gradio.app import run
        if args.backend == 'pytorch':
            from lmdeploy.pytorch.config import EngineConfig
            backend_config = EngineConfig(tp=args.tp,
                                          max_batch_size=args.max_batch_size,
                                          block_size=args.block_size)
        else:
            from lmdeploy.turbomind.engine_config import EngineConfig
            backend_config = EngineConfig(
                tp=args.tp,
                max_batch_size=args.max_batch_size,
                model_format=args.model_format,
                quant_policy=args.quant_policy,
                rope_scaling_factor=args.rope_scaling_factor,
                use_logn_attn=args.use_logn_attn)
        chat_template_config = ChatTemplateConfig(
            model_name=args.model_name,
            meta_instruction=args.meta_instruction,
            capability=args.cap)
        run(args.model_path_or_server,
            server_name=args.server_name,
            server_port=args.server_port,
            batch_size=args.batch_size,
            backend=args.backend,
            backend_config=backend_config,
            chat_template_config=chat_template_config,
            tp=args.tp,
            model_name=args.model_name)

    @staticmethod
    def api_server(args):
        """Serve LLMs with restful api using fastapi."""
        from lmdeploy.model import ChatTemplateConfig
        from lmdeploy.serve.openai.api_server import serve as run_api_server
        if args.backend == 'pytorch':
            from lmdeploy.pytorch.config import EngineConfig
            backend_config = EngineConfig(tp=args.tp,
                                          max_batch_size=args.max_batch_size,
                                          block_size=args.block_size)
        else:
            from lmdeploy.turbomind.engine_config import EngineConfig
            backend_config = EngineConfig(
                tp=args.tp,
                max_batch_size=args.max_batch_size,
                model_format=args.model_format,
                quant_policy=args.quant_policy,
                rope_scaling_factor=args.rope_scaling_factor,
                use_logn_attn=args.use_logn_attn)
        chat_template_config = ChatTemplateConfig(
            model_name=args.model_name,
            meta_instruction=args.meta_instruction,
            capability=args.cap)
        run_api_server(args.model_path,
                       model_name=args.model_name,
                       backend=args.backend,
                       backend_config=backend_config,
                       chat_template_config=chat_template_config,
                       server_name=args.server_name,
                       server_port=args.server_port,
                       instance_num=args.instance_num,
                       tp=args.tp,
                       allow_origins=args.allow_origins,
                       allow_credentials=args.allow_credentials,
                       allow_methods=args.allow_methods,
                       allow_headers=args.allow_headers,
                       log_level=args.log_level.upper(),
                       qos_config_path=args.qos_config_path)

    @staticmethod
    def api_client(args):
        """Interact with restful api server in terminal."""
        from lmdeploy.serve.openai.api_client import main as run_api_client
        kwargs = convert_args(args)
        run_api_client(**kwargs)

    @staticmethod
    def triton_client(args):
        """Interact with Triton Server using gRPC protocol."""
        from lmdeploy.serve.client import main as run_triton_client
        kwargs = convert_args(args)
        other_kwargs = kwargs.pop('kwargs', None)
        if other_kwargs is not None:
            kwargs.update(other_kwargs)
        run_triton_client(**kwargs)

    @staticmethod
    def add_parsers():
        SubCliServe.add_parser_gradio()
        SubCliServe.add_parser_api_server()
        SubCliServe.add_parser_api_client()
        SubCliServe.add_parser_triton_client()

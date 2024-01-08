# Copyright (c) OpenMMLab. All rights reserved.
from .cli import CLI
from .utils import (DefaultsAndTypesHelpFormatter, convert_args,
                    get_engine_parser)


class SubCliChat(object):
    _help = 'Chat through terminal with pytorch or turbomind model.'
    _desc = _help
    parser = CLI.subparsers.add_parser('chat', help=_help, description=_desc)
    subparsers = parser.add_subparsers(
        title='Commands', description='This group has the following commands:')

    @staticmethod
    def add_parser_torch():
        parser = SubCliChat.subparsers.add_parser(
            'torch',
            parents=[get_engine_parser(add_pytorch=True)],
            formatter_class=DefaultsAndTypesHelpFormatter,
            help=SubCliChat.torch.__doc__,
            description=SubCliChat.torch.__doc__,
        )
        parser.set_defaults(run=SubCliChat.torch)
        parser.add_argument('model_path',
                            type=str,
                            help='The huggingface model path')
        parser.add_argument('--model-name',
                            type=str,
                            default=None,
                            help='Name of the input model')
        parser.add_argument('--session-id',
                            type=int,
                            default=1,
                            help='The identical id of a session')
        parser.add_argument('--top-k',
                            type=float,
                            default=40,
                            help='Sampling top k')
        parser.add_argument('--top-p',
                            type=float,
                            default=0.8,
                            help='Sampling top p')
        parser.add_argument('--temperature',
                            type=float,
                            default=0.8,
                            help='Sampling temperature')
        parser.add_argument('--repetition-penalty',
                            type=float,
                            default=1.0,
                            help='Parameter to penalize repetition')
        parser.add_argument('--trust-remote-code',
                            action='store_false',
                            default=True,
                            help='Trust remote code')

    @staticmethod
    def add_parser_turbomind():
        parser = SubCliChat.subparsers.add_parser(
            'turbomind',
            parents=[get_engine_parser(add_turbomind=True)],
            formatter_class=DefaultsAndTypesHelpFormatter,
            help=SubCliChat.turbomind.__doc__,
            description=SubCliChat.turbomind.__doc__,
        )
        parser.set_defaults(run=SubCliChat.turbomind)
        group = parser.add_argument_group('engine arguments')
        parser.add_argument('model_path',
                            type=str,
                            help='The path of the deployed model')
        group.add_argument('--model-name',
                           type=str,
                           default=None,
                           help='The name of deployed model')
        group.add_argument('--session-id',
                           type=int,
                           default=1,
                           help='The identical id of a session')
        group.add_argument(
            '--cap',
            type=str,
            default='chat',
            choices=['completion', 'infilling', 'chat', 'python'],
            help='The capability of a model. For example, '
            'codellama has the ability among ["completion", '
            '"infilling", "chat", "python"]')
        parser.add_argument('--stream-output',
                            action='store_true',
                            help='Indicator for streaming output or not')
        parser.add_argument('--request-output-len',
                            type=int,
                            default=512,
                            help='Output token nums')

    @staticmethod
    def torch(args):
        """Chat with PyTorch inference engine through terminal."""
        from lmdeploy.messages import EngineGenerationConfig
        from lmdeploy.pytorch.chat import run_chat
        from lmdeploy.pytorch.config import EngineConfig

        engine_config = EngineConfig(model_name=args.model_name, tp=args.tp)
        gen_config = EngineGenerationConfig(
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
        run_chat(args.model_path,
                 engine_config,
                 gen_config=gen_config,
                 session_id=args.session_id,
                 trust_remote_code=args.trust_remote_code)

    @staticmethod
    def turbomind(args):
        """Chat with TurboMind inference engine through terminal."""
        from lmdeploy.turbomind.chat import main
        kwargs = convert_args(args)
        main(**kwargs)

    @staticmethod
    def add_parsers():
        """Add all parsers."""
        SubCliChat.add_parser_torch()
        SubCliChat.add_parser_turbomind()

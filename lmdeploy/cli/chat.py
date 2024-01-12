# Copyright (c) OpenMMLab. All rights reserved.
from .cli import CLI
from .utils import ArgumentHelper, DefaultsAndTypesHelpFormatter, convert_args


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
            formatter_class=DefaultsAndTypesHelpFormatter,
            help=SubCliChat.torch.__doc__,
            description=SubCliChat.torch.__doc__,
        )
        parser.set_defaults(run=SubCliChat.torch)
        parser.add_argument('model_path',
                            type=str,
                            help='The huggingface model path')
        # engine args
        engine_group = parser.add_argument_group('Engine arguments')
        ArgumentHelper.model_name(engine_group)
        ArgumentHelper.tp(engine_group)
        ArgumentHelper.session_len(engine_group)
        ArgumentHelper.adapters(engine_group)

        # other args
        parser.add_argument('--trust-remote-code',
                            action='store_false',
                            default=True,
                            help='Trust remote code')

    @staticmethod
    def add_parser_turbomind():
        parser = SubCliChat.subparsers.add_parser(
            'turbomind',
            formatter_class=DefaultsAndTypesHelpFormatter,
            help=SubCliChat.turbomind.__doc__,
            description=SubCliChat.turbomind.__doc__,
        )
        parser.set_defaults(run=SubCliChat.turbomind)
        parser.add_argument(
            'model_path',
            type=str,
            help='The path of the deployed model. '
            'It can be in format of huggingface or turbomind. '
            'When it is turbomind model, all arguments for engine'
            'config would be ignored, so you need to change the `config.ini`')
        # engine arguments
        engine_group = parser.add_argument_group('Engine arguments')
        ArgumentHelper.tp(engine_group)
        ArgumentHelper.model_format(engine_group)
        ArgumentHelper.quant_policy(engine_group)
        ArgumentHelper.model_name(engine_group)
        ArgumentHelper.cache_max_entry_count(engine_group)
        ArgumentHelper.rope_scaling_factor(engine_group)
        ArgumentHelper.session_len(engine_group)
        # other arguments
        ArgumentHelper.cap(parser)

    @staticmethod
    def torch(args):
        """Chat with PyTorch inference engine through terminal."""
        from lmdeploy.pytorch.chat import run_chat
        from lmdeploy.pytorch.config import EngineConfig

        engine_config = EngineConfig(model_name=args.model_name,
                                     tp=args.tp,
                                     session_len=args.session_len,
                                     adapters=args.adapters)
        run_chat(args.model_path,
                 engine_config,
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

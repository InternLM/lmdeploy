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
        parser.add_argument('--customize-torch',
                            type=str,
                            choices=['True', 'False'],
                            default='False',
                            help='default sxx')

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
        parser.add_argument('--turbomind')

    @staticmethod
    def torch(args):
        """Chat with PyTorch inference engine through terminal."""
        from lmdeploy.pytorch.chat import main
        kwargs = convert_args(args)
        # TODO
        main(**kwargs)

    @staticmethod
    def turbomind(args):
        """Chat with TurboMind inference engine through terminal."""
        from lmdeploy.turbomind.chat import main
        kwargs = convert_args(args)
        # TODO
        main(**kwargs)

    @staticmethod
    def add_parsers():
        """Add all parsers."""
        SubCliChat.add_parser_torch()
        SubCliChat.add_parser_turbomind()

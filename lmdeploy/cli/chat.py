# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import DictAction

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
        # enging args
        engine_group = parser.add_argument_group('Engine arguments')
        ArgumentHelper.model_name(engine_group)
        ArgumentHelper.tp(engine_group)
        ArgumentHelper.max_batch_size(engine_group)
        ArgumentHelper.block_size(engine_group)

        # generation args
        gen_group = parser.add_argument_group('Generation arguments')
        ArgumentHelper.top_k(gen_group)
        ArgumentHelper.top_p(gen_group)
        ArgumentHelper.temperature(gen_group)
        ArgumentHelper.repetition_penalty(gen_group)

        # other args
        ArgumentHelper.session_len(parser)
        parser.add_argument('--adapter',
                            default=None,
                            action=DictAction,
                            help='Used key-values pairs in xxx=yyy format'
                            ' to set the path lora adapter')
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
        parser.add_argument('model_path',
                            type=str,
                            help='The path of the deployed model')
        # engine arguments
        engine_group = parser.add_argument_group('Engine arguments')
        ArgumentHelper.tp(engine_group)
        ArgumentHelper.max_batch_size(engine_group)
        ArgumentHelper.model_format(engine_group)
        ArgumentHelper.quant_policy(engine_group)
        ArgumentHelper.rope_scaling_factor(engine_group)
        ArgumentHelper.use_logn_attn(engine_group)
        ArgumentHelper.model_name(engine_group)

        # other arguments
        ArgumentHelper.session_len(parser)
        ArgumentHelper.cap(parser)
        ArgumentHelper.stream_output(parser)
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

        engine_config = EngineConfig(model_name=args.model_name,
                                     tp=args.tp,
                                     max_batch_size=args.max_batch_size,
                                     adapters=args.adapter)
        gen_config = EngineGenerationConfig(
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
        run_chat(args.model_path,
                 engine_config,
                 gen_config=gen_config,
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

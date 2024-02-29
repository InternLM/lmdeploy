# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

from ..version import __version__
from .utils import ArgumentHelper, DefaultsAndTypesHelpFormatter, convert_args


class CLI(object):
    _desc = 'The CLI provides a unified API for converting, ' \
            'compressing and deploying large language models.'
    parser = argparse.ArgumentParser(prog='lmdeploy',
                                     description=_desc,
                                     add_help=True)
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=__version__)
    subparsers = parser.add_subparsers(
        title='Commands',
        description='lmdeploy has following commands:',
        dest='command')

    @staticmethod
    def add_parser_convert():
        """Add parser for convert command."""
        parser = CLI.subparsers.add_parser(
            'convert',
            formatter_class=DefaultsAndTypesHelpFormatter,
            description=CLI.convert.__doc__,
            help=CLI.convert.__doc__)
        # define arguments
        parser.add_argument(
            'model_name',
            type=str,
            help='The name of the to-be-deployed model, such as llama-7b, '
            'llama-13b, vicuna-7b and etc. You can run `lmdeploy list` to '
            'get the supported model names')
        parser.add_argument('model_path',
                            type=str,
                            help='The directory path of the model')
        ArgumentHelper.model_format(parser)
        ArgumentHelper.tp(parser)
        # other args
        parser.add_argument('--tokenizer-path',
                            type=str,
                            default=None,
                            help='The path of tokenizer model')
        parser.add_argument('--dst-path',
                            type=str,
                            default='workspace',
                            help='The destination path that saves outputs')
        parser.add_argument(
            '--quant-path',
            type=str,
            default=None,
            help='Path of the quantized model, which can be none')
        parser.add_argument(
            '--group-size',
            type=int,
            default=0,
            help='A parameter used in awq to quantize fp16 weights '
            'to 4 bits')

        parser.set_defaults(run=CLI.convert)

    @staticmethod
    def add_parser_list():
        """Add parser for list command."""
        parser = CLI.subparsers.add_parser(
            'list',
            formatter_class=DefaultsAndTypesHelpFormatter,
            description=CLI.list.__doc__,
            help=CLI.list.__doc__)
        parser.set_defaults(run=CLI.list)
        # define arguments
        ArgumentHelper.engine(parser)

    @staticmethod
    def add_parser_checkenv():
        """Add parser for check_env command."""
        parser = CLI.subparsers.add_parser(
            'check_env',
            formatter_class=DefaultsAndTypesHelpFormatter,
            description=CLI.check_env.__doc__,
            help=CLI.check_env.__doc__)
        parser.set_defaults(run=CLI.check_env)
        parser.add_argument('--dump-file',
                            type=str,
                            default=None,
                            help='The file path to save env info. Only '
                            'support file format in `json`, `yml`,'
                            ' `pkl`')

    @staticmethod
    def convert(args):
        """Convert LLMs to turbomind format."""
        from lmdeploy.turbomind.deploy.converter import main
        kwargs = convert_args(args)
        main(**kwargs)

    @staticmethod
    def list(args):
        """List the supported model names."""
        engine = args.engine
        assert engine in ['turbomind', 'pytorch']
        if engine == 'pytorch':
            model_names = [
                'llama', 'llama2', 'internlm', 'internlm2', 'baichuan2',
                'chatglm2', 'falcon', 'yi', 'mistral', 'mixtral', 'qwen1.5',
                'gemma'
            ]
        elif engine == 'turbomind':
            from lmdeploy.model import MODELS
            model_names = list(MODELS.module_dict.keys())
            model_names = [n for n in model_names if n.lower() not in ['base']]
        model_names.sort()
        print('Supported model names:')
        print('\n'.join(model_names))

    @staticmethod
    def check_env(args):
        """Check the environmental information."""
        import importlib

        import mmengine
        from mmengine.utils import get_git_hash
        from mmengine.utils.dl_utils import collect_env

        from lmdeploy.version import __version__

        env_info = collect_env()
        env_info['LMDeploy'] = __version__ + '+' + get_git_hash()[:7]

        # remove some unnecessary info
        remove_reqs = ['MMEngine', 'OpenCV']
        for req in remove_reqs:
            if req in env_info:
                env_info.pop(req)

        # extra important dependencies
        extra_reqs = ['transformers', 'gradio', 'fastapi', 'pydantic']

        for req in extra_reqs:
            try:
                env_info[req] = importlib.import_module(req).__version__
            except Exception:
                env_info[req] = 'Not Found'

        # print env info
        for k, v in env_info.items():
            print(f'{k}: {v}')

        # dump to local file
        dump_file = args.dump_file
        if dump_file is not None:
            work_dir, _ = os.path.split(dump_file)
            if work_dir:
                os.makedirs(work_dir, exist_ok=True)
            mmengine.dump(env_info, dump_file)

    @staticmethod
    def add_parsers():
        """Add all parsers."""
        CLI.add_parser_convert()
        CLI.add_parser_list()
        CLI.add_parser_checkenv()

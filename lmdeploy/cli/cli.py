# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

from ..version import __version__
from .utils import (ArgumentHelper, DefaultsAndTypesHelpFormatter,
                    convert_args, get_lora_adapters)


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

    @staticmethod
    def add_parser_chat():
        """Add parser for list command."""
        parser = CLI.subparsers.add_parser(
            'chat',
            formatter_class=DefaultsAndTypesHelpFormatter,
            description=CLI.chat.__doc__,
            help=CLI.chat.__doc__)
        parser.set_defaults(run=CLI.chat)
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
        ArgumentHelper.trut_remote_code(parser)
        # # chat template args
        ArgumentHelper.meta_instruction(parser)
        ArgumentHelper.cap(parser)
        #
        # pytorch engine args
        pt_group = parser.add_argument_group('PyTorch engine arguments')
        ArgumentHelper.adapters(pt_group)

        # common engine args
        tp_act = ArgumentHelper.tp(pt_group)
        model_name_act = ArgumentHelper.model_name(pt_group)
        session_len_act = ArgumentHelper.session_len(pt_group)
        max_batch_size_act = ArgumentHelper.max_batch_size(pt_group)
        cache_max_entry_act = ArgumentHelper.cache_max_entry_count(pt_group)

        # turbomind args
        tb_group = parser.add_argument_group('TurboMind engine arguments')
        # common engine args
        tb_group._group_actions.append(tp_act)
        tb_group._group_actions.append(model_name_act)
        tb_group._group_actions.append(session_len_act)
        tb_group._group_actions.append(max_batch_size_act)
        tb_group._group_actions.append(cache_max_entry_act)
        ArgumentHelper.model_format(tb_group)
        ArgumentHelper.quant_policy(tb_group)
        ArgumentHelper.rope_scaling_factor(tb_group)

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
        from lmdeploy.model import MODELS
        model_names = list(MODELS.module_dict.keys())
        deprecate_names = [
            'baichuan-7b', 'baichuan2-7b', 'chatglm2-6b', 'internlm-chat-20b',
            'internlm-chat-7b', 'internlm-chat-7b-8k', 'internlm2-1_8b',
            'internlm-20b', 'internlm2-20b', 'internlm2-7b', 'internlm2-chat',
            'internlm2-chat-1_8b', 'internlm2-chat-20b', 'internlm2-chat-7b',
            'llama-2-chat', 'llama-2', 'qwen-14b', 'qwen-7b', 'solar-70b',
            'yi-200k', 'yi-34b', 'yi-chat', 'Mistral-7B-Instruct',
            'Mixtral-8x7B-Instruct', 'baichuan-base', 'deepseek-chat',
            'internlm-chat'
        ]
        model_names = [
            n for n in model_names if n not in deprecate_names + ['base']
        ]
        deprecate_names.sort()
        model_names.sort()
        print('The older chat template name like "internlm2-7b", "qwen-7b"'
              ' and so on are deprecated and will be removed in the future.'
              ' The supported chat template names are:')
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
        extra_reqs = [
            'transformers', 'gradio', 'fastapi', 'pydantic', 'triton'
        ]

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
    def chat(args):
        """Chat with pytorch or turbomind engine."""
        from lmdeploy.archs import autoget_backend
        backend = args.backend
        if backend != 'pytorch':
            # set auto backend mode
            backend = autoget_backend(args.model_path)
        if backend == 'pytorch':
            from lmdeploy.messages import PytorchEngineConfig
            from lmdeploy.pytorch.chat import run_chat

            adapters = get_lora_adapters(args.adapters)
            engine_config = PytorchEngineConfig(
                model_name=args.model_name,
                tp=args.tp,
                session_len=args.session_len,
                cache_max_entry_count=args.cache_max_entry_count,
                adapters=adapters)
            run_chat(args.model_path,
                     engine_config,
                     trust_remote_code=args.trust_remote_code)
        else:
            from lmdeploy.turbomind.chat import main as run_chat
            kwargs = convert_args(args)
            run_chat(**kwargs)

    @staticmethod
    def add_parsers():
        """Add all parsers."""
        CLI.add_parser_convert()
        CLI.add_parser_list()
        CLI.add_parser_checkenv()

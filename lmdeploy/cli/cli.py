# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

from ..version import __version__
from .utils import (ArgumentHelper, DefaultsAndTypesHelpFormatter,
                    convert_args, get_chat_template, get_lora_adapters)


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
            help='deprecated and unused, '
            'it will be removed on 2024.12.31. It was originally used to '
            'specify the name of the built-in chat template, but now it '
            'is substituted with a clearer parameter `--chat-template`')
        parser.add_argument('model_path',
                            type=str,
                            help='The directory path of the model')
        ArgumentHelper.model_format(parser)
        ArgumentHelper.tp(parser)
        # other args
        ArgumentHelper.revision(parser)
        ArgumentHelper.download_dir(parser)
        parser.add_argument('--tokenizer-path',
                            type=str,
                            default=None,
                            help='The path of tokenizer model')
        parser.add_argument('--dst-path',
                            type=str,
                            default='workspace',
                            help='The destination path that saves outputs')
        parser.add_argument(
            '--group-size',
            type=int,
            default=0,
            help='A parameter used in awq to quantize fp16 weights '
            'to 4 bits')
        parser.add_argument(
            '--chat-template',
            type=str,
            default=None,
            help='the name of the built-in chat template, which can be '
            'overviewed by `lmdeploy list`')
        parser.add_argument(
            '--dtype',
            type=str,
            default='auto',
            choices=['auto', 'float16', 'bfloat16'],
            help='data type for model weights and activations. '
            'The "auto" option will use FP16 precision '
            'for FP32 and FP16 models, and BF16 precision '
            'for BF16 models. This option will be ignored if '
            'the model is a quantized model')
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
        # # chat template args
        ArgumentHelper.chat_template(parser)
        # model args
        ArgumentHelper.revision(parser)
        ArgumentHelper.download_dir(parser)
        #
        # pytorch engine args
        pt_group = parser.add_argument_group('PyTorch engine arguments')
        ArgumentHelper.adapters(pt_group)
        ArgumentHelper.device(pt_group)
        # common engine args
        dtype_act = ArgumentHelper.dtype(pt_group)
        tp_act = ArgumentHelper.tp(pt_group)
        session_len_act = ArgumentHelper.session_len(pt_group)
        cache_max_entry_act = ArgumentHelper.cache_max_entry_count(pt_group)
        prefix_caching_act = ArgumentHelper.enable_prefix_caching(pt_group)

        # turbomind args
        tb_group = parser.add_argument_group('TurboMind engine arguments')
        # common engine args
        tb_group._group_actions.append(dtype_act)
        tb_group._group_actions.append(tp_act)
        tb_group._group_actions.append(session_len_act)
        tb_group._group_actions.append(cache_max_entry_act)
        tb_group._group_actions.append(prefix_caching_act)
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
        model_names.sort()
        print('The supported chat template names are:')
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

        def get_gpu_topo():
            import subprocess
            import sys
            if sys.platform.startswith('linux'):
                try:
                    res = subprocess.run(['nvidia-smi', 'topo', '-m'],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         text=True,
                                         check=True)
                    if res.returncode == 0:
                        return '\n' + res.stdout
                    else:
                        return None
                except FileNotFoundError:
                    return None
            else:
                return None

        gpu_topo = get_gpu_topo()
        if gpu_topo is not None:
            env_info['NVIDIA Topology'] = gpu_topo

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

        chat_template_config = get_chat_template(args.chat_template)

        backend = args.backend
        if backend != 'pytorch':
            # set auto backend mode
            backend = autoget_backend(args.model_path)

        if backend == 'pytorch':
            from lmdeploy.messages import PytorchEngineConfig
            from lmdeploy.pytorch.chat import run_chat

            adapters = get_lora_adapters(args.adapters)
            engine_config = PytorchEngineConfig(
                dtype=args.dtype,
                tp=args.tp,
                session_len=args.session_len,
                cache_max_entry_count=args.cache_max_entry_count,
                adapters=adapters,
                enable_prefix_caching=args.enable_prefix_caching,
                device_type=args.device)
            run_chat(args.model_path,
                     engine_config,
                     chat_template_config=chat_template_config)
        else:
            from lmdeploy.turbomind.chat import main as run_chat
            kwargs = convert_args(args)
            kwargs.pop('chat_template')
            kwargs.pop('backend')
            kwargs.pop('device')
            kwargs['chat_template_config'] = chat_template_config
            run_chat(**kwargs)

    @staticmethod
    def add_parsers():
        """Add all parsers."""
        CLI.add_parser_convert()
        CLI.add_parser_list()
        CLI.add_parser_checkenv()
        CLI.add_parser_chat()

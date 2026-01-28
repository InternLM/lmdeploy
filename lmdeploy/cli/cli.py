# Copyright (c) OpenMMLab. All rights reserved.

import os

from ..version import __version__
from .utils import (ArgumentHelper, DefaultsAndTypesHelpFormatter, FlexibleArgumentParser, convert_args,
                    get_speculative_config)


class CLI(object):
    _desc = 'The CLI provides a unified API for converting, ' \
            'compressing and deploying large language models.'
    parser = FlexibleArgumentParser(prog='lmdeploy', description=_desc, add_help=True)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    subparsers = parser.add_subparsers(title='Commands', description='lmdeploy has following commands:', dest='command')

    @staticmethod
    def add_parser_chat():
        """Add parser for list command."""
        parser = CLI.subparsers.add_parser('chat',
                                           formatter_class=DefaultsAndTypesHelpFormatter,
                                           description=CLI.chat.__doc__,
                                           help=CLI.chat.__doc__)
        parser.set_defaults(run=CLI.chat)
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
        # common args
        ArgumentHelper.backend(parser)
        # chat template args
        ArgumentHelper.chat_template(parser)
        # model args
        ArgumentHelper.revision(parser)
        ArgumentHelper.download_dir(parser)

        # pytorch engine args
        pt_group = parser.add_argument_group('PyTorch engine arguments')
        ArgumentHelper.adapters(pt_group)
        ArgumentHelper.device(pt_group)
        ArgumentHelper.eager_mode(pt_group)
        ArgumentHelper.dllm_block_length(pt_group)
        # common engine args
        dtype_act = ArgumentHelper.dtype(pt_group)
        tp_act = ArgumentHelper.tp(pt_group)
        session_len_act = ArgumentHelper.session_len(pt_group)
        cache_max_entry_act = ArgumentHelper.cache_max_entry_count(pt_group)
        prefix_caching_act = ArgumentHelper.enable_prefix_caching(pt_group)
        quant_policy = ArgumentHelper.quant_policy(pt_group)

        # turbomind args
        tb_group = parser.add_argument_group('TurboMind engine arguments')
        # common engine args
        tb_group._group_actions.append(dtype_act)
        tb_group._group_actions.append(tp_act)
        tb_group._group_actions.append(session_len_act)
        tb_group._group_actions.append(cache_max_entry_act)
        tb_group._group_actions.append(prefix_caching_act)
        tb_group._group_actions.append(quant_policy)
        ArgumentHelper.model_format(tb_group)
        ArgumentHelper.rope_scaling_factor(tb_group)
        ArgumentHelper.communicator(tb_group)
        ArgumentHelper.cp(tb_group)
        ArgumentHelper.async_(tb_group)

        # speculative decoding
        ArgumentHelper.add_spec_group(parser)

    @staticmethod
    def add_parser_checkenv():
        """Add parser for check_env command."""
        parser = CLI.subparsers.add_parser('check_env',
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
        extra_reqs = ['transformers', 'fastapi', 'pydantic', 'triton']

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
        from .chat import main

        kwargs = convert_args(args)
        speculative_config = get_speculative_config(args)
        to_remove = ['speculative_algorithm', 'speculative_draft_model', 'speculative_num_draft_tokens']
        for key in to_remove:
            kwargs.pop(key)
        kwargs['speculative_config'] = speculative_config
        main(**kwargs)

    @staticmethod
    def add_parsers():
        """Add all parsers."""
        CLI.add_parser_checkenv()
        CLI.add_parser_chat()

# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
from copy import deepcopy

from mmengine.config import read_base
from opencompass.models import (
    HuggingFacewithChatTemplate,
    TurboMindModelwithChatTemplate,
)


#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################

with read_base():
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import bbh_datasets

    # Datasets, Summarizer

    # from .settings.chat_objective import datasets
    # from .settings.chat_objective import summarizer

    # # Runner
    # from ..configs_cluster.dlc_internal import eval as dlc_eval
    # from ..configs_cluster.dlc_internal import infer as dlc_infer
    # from ..configs_cluster.local import infer as local_infer
    # from ..configs_cluster.local import eval as local_eval

    # Models
    # from ..configs_models.chat_models import models


#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
datasets = bbh_datasets


#######################################################################
#                       PART 2  Dataset Summarizer                    #
#######################################################################
# summarizer = summarizer

#######################################################################
#                        PART 3  Models  List                         #
#######################################################################
# models = models

models = []
model_configs = [
    # Baseline
    # (abbr, path, num_gpus)
    (
        'rule-merge-fix_dense8B-merge-ratio_4_step_60_80_open_source_hf',
        '/mnt/141/internlm3/lvhaijun/develop_internlm3_open_source_hf_0110v1/rule-merge-fix_dense8B-merge-ratio_4_step_60_80_open_source_hf',
        1,
    ),
]

max_seq_len = 32768
max_out_len = 8192
max_batch_size = 128

for abbr, path, num_gpus in model_configs:
    if abbr is None:
        abbr = path.split('/')[-2] + '--' + path.split('/')[-1]

    base_model = dict(
        type=TurboMindModelwithChatTemplate,
        abbr=abbr,
        path=path,
        engine_config=dict(
            session_len=max_seq_len, max_batch_size=max_batch_size, tp=num_gpus
        ),
        gen_config=dict(
            top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=max_out_len, stop_token_ids=[128131]
        ),
        max_seq_len=max_seq_len,
        max_out_len=max_out_len,
        batch_size=max_batch_size,
        run_cfg=dict(num_gpus=num_gpus),
    )

    hf_base_model = dict(
        type=HuggingFacewithChatTemplate,
        abbr=abbr,
        path=path,
        max_out_len=max_out_len,
        batch_size=8,
        run_cfg=dict(num_gpus=num_gpus),
    )

    model = deepcopy(base_model)
    if 'TurboMindModelwithChatTemplate' in str(model['type']):
        model['gen_config']['top_k'] = 1  # greedy
        model['gen_config']['temperature'] = 1e-6
        models.append(model)
    else:
        models.append(model)

#######################################################################
#                 PART 4  Inference/Evaluation Configuration          #
#######################################################################

# Choose between dlc_infer and local_infer
# # Local Runner
# infer = local_infer
# eval = local_eval

# DLC Runner
# infer = dlc_infer
# eval = dlc_eval
# # infer["runner"]["aliyun_cfg"][
# #     "python_env_path"
# # ] = '/cpfs01/shared/public/public_hdd/shared_conda_envs/oc-v039postdev-ld-v063-internal'
# # eval["runner"]["aliyun_cfg"][
# #     "python_env_path"
# # ] = '/cpfs01/shared/public/public_hdd/shared_conda_envs/oc-v039postdev-ld-v063-internal'

# infer["runner"]["aliyun_cfg"][
#     "python_env_path"
# ] = '/cpfs01/shared/public/public_hdd/shared_conda_envs/oc-v039postdev-ld-v070a1-test'
# eval["runner"]["aliyun_cfg"][
#     "python_env_path"
# ] = '/cpfs01/shared/public/public_hdd/shared_conda_envs/oc-v039postdev-ld-v070a1-test'

#######################################################################
#                      PART 5  Utils Configuration                    #
#######################################################################
base_exp_dir = 'outputs/fullbench_internlm3/'
work_dir = osp.join(base_exp_dir, 'chat_objective')


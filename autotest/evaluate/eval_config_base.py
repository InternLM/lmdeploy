# flake8: noqa
"""API-based base-model OpenCompass config (PPL / LL / gen).

Uses ``TurboMindAPIModel`` against a running ``lmdeploy serve api_server``
(``/get_ppl``, ``/v1/encode``, completions).
"""

import os as _os

from mmengine.config import read_base
from opencompass.models.turbomind_api import TurboMindAPIModel
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferConcurrentTask

#######################################################################
#                          Import base configs                        #
#######################################################################

with read_base():
    from opencompass.configs.datasets.gpqa.gpqa_few_shot_ppl_4b5a83 import \
        gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import \
        mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_few_shot_ppl import \
        race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import \
        winogrande_datasets  # noqa: F401, E501

    from opencompass.configs.summarizers.groups.mmlu import \
        mmlu_summary_groups  # noqa: F401, E501

#######################################################################
#                     Dataset subset overrides                        #
#######################################################################

race_datasets = [race_datasets[1]]
mmlu_datasets = [
    x for x in mmlu_datasets if x['abbr'].replace('lukaemon_mmlu_', '') in [
        'business_ethics', 'clinical_knowledge', 'college_medicine',
        'global_facts', 'human_aging', 'management', 'marketing',
        'medical_genetics', 'miscellaneous', 'nutrition',
        'professional_accounting', 'professional_medicine', 'virology'
    ]
]

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

#######################################################################
#                         Model Configuration                         #
#######################################################################

MODEL_NAME = ''
MODEL_PATH = ''
API_BASE = ''

models = [
    dict(
        type=TurboMindAPIModel,
        abbr=f'{MODEL_NAME}',
        model_name=MODEL_NAME,
        api_addr=API_BASE,
        max_seq_len=7168,
        max_out_len=1024,
        batch_size=32,
        temperature=1e-6,
        top_p=0.9,
        top_k=1,
        gen_config=dict(max_new_tokens=1024),
        run_cfg=dict(num_gpus=0),
    )
]

#######################################################################
#                            Summarizer                               #
#######################################################################

summarizer = dict(
    dataset_abbrs=[
        ['race-high', 'accuracy'],
        ['GPQA_diamond', 'accuracy'],
        ['mmlu', 'naive_average'],
        ['gsm8k', 'accuracy'],
        ['winogrande', 'accuracy'],
        '',
        'mmlu',
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-humanities',
        'mmlu-other',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

#######################################################################
#                         Infer / Eval runners                        #
#######################################################################

_dataset_size_root = _os.path.dirname(
    _os.environ.get('REPORT_DIR', '.').rstrip('/') or '.') or '.'
_dataset_type = _os.environ.get('CHAT_TYPE', 'base').rstrip('/')
dataset_size_path = f'{_dataset_size_root}/dataset_size_{_dataset_type}.json'

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=1,
        dataset_size_path=dataset_size_path,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        retry=0,
        task=dict(type=OpenICLInferConcurrentTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
    ),
)

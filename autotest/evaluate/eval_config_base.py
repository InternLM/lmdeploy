import os as _os

from mmengine.config import read_base
from opencompass.models.turbomind_api import TurboMindAPIModel
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

# One MMLU-Pro subject (not CS+physics): extra knowledge signal for Qwen3 / 3.5
# base without a second gen task.
MMLU_PRO_KEEP = {
    'mmlu_pro_computer_science',
}

#######################################################################
#                          Import base configs                        #
#######################################################################

with read_base():
    from opencompass.configs.datasets.gpqa.gpqa_few_shot_ppl_4b5a83 import gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_new_gen import gsm8k_datasets  # noqa: F401, E501

    # from opencompass.configs.datasets.humaneval.internal_humaneval_v2_new_gen import (
    #   humaneval_datasets,  # noqa: F401, E501
    # )
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_few_shot_new_gen import mmlu_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_few_shot_ppl import race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import (
        winogrande_datasets,  # noqa: F401, E501
    )

#######################################################################
#                         Dataset list                                #
#######################################################################

mmlu_pro_datasets = [
    x for x in mmlu_pro_datasets if x['abbr'] in MMLU_PRO_KEEP
]
race_datasets = [race_datasets[1]]  # RACE-High only

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
#                         Infer / Eval runners                        #
#######################################################################

_dataset_size_root = _os.path.dirname(
    _os.environ.get('REPORT_DIR', '.').rstrip('/') or '.') or '.'
_dataset_type = _os.environ.get('CHAT_TYPE', 'base').rstrip('/')
dataset_size_path = f'{_dataset_size_root}/dataset_size_{_dataset_type}.json'

# PPLInferencer / LLInferencer are not supported by OpenICLInferConcurrentTask.
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
        task=dict(type=OpenICLInferTask),
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

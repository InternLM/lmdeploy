from mmengine.config import read_base
from opencompass.models import OpenAISDK
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.subjective.alignbench.alignbench_judgeby_critiquellm import \
        alignbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import \
        alpacav2_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare import \
        arenahard_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.compassarena.compassarena_compare import \
        compassarena_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.fofo.fofo_bilingual_judge import fofo_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.multiround.mtbench101_judge import \
        mtbench101_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge import \
        wildbench_datasets  # noqa: F401, E501

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets') and 'wildbench' not in k), [])
datasets += wildbench_datasets

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        abbr='lmdeploy-api-test',
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base='http://localhost:23344/v1',
        path='/nvme/qa_test_models/internlm/internlm2_5-20b-chat',
        tokenizer_path='/nvme/qa_test_models/internlm/internlm2_5-20b-chat',
        rpm_verbose=True,
        meta_template=api_meta_template,
        query_per_second=100,
        max_out_len=1024,
        max_seq_len=4096,
        temperature=0.01,
        batch_size=128,
        retry=3,
    )
]

judge_models = models

eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=SubjectiveEvalTask)),
)

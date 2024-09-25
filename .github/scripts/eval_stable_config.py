from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.bbh.bbh_gen_2879b0 import \
        bbh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import \
        ceval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmmlu.cmmlu_gen_c13365 import \
        cmmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_4c31db import \
        GaokaoBench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_gen_4baadb import \
        gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import \
        hellaswag_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import \
        humaneval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import \
        ifeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import \
        math_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_830460 import \
        sanitized_mbpp_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import \
        mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.nq.nq_open_1shot_gen_2e45e5 import \
        nq_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_gen_69ee4f import \
        race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.alignbench.alignbench_judgeby_critiquellm import \
        alignbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import \
        alpacav2_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare import \
        arenahard_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.compassarena.compassarena_compare import \
        compassarena_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.fofo.fofo_bilingual_judge import \
        fofo_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.followbench.followbench_llmeval import \
        followbench_llmeval_dataset  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.multiround.mtbench101_judge import \
        mtbench101_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge import \
        wildbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import \
        TheoremQA_datasets  # noqa: F401, E501 # noqa: F401, E501
    from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_bc5f21 import \
        triviaqa_datasets  # noqa: F401, E501 # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_gen_b36770 import \
        winogrande_datasets  # noqa: F401, E501

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

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
        openai_api_base='http://localhost:23333/v1',
        path='internlm2_20b_api',
        tokenizer_path='internlm/internlm2_5-20b-chat',
        rpm_verbose=True,
        meta_template=api_meta_template,
        query_per_second=50,
        max_out_len=1024,
        max_seq_len=4096,
        temperature=0.01,
        batch_size=128,
        retry=3,
    )
]

judge_models = models

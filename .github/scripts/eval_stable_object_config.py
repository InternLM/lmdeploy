from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import ARC_c_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import bbh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.CHARM.charm_reason_cot_only_gen_f7b7d3 import \
        charm_reason_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmmlu.cmmlu_0shot_cot_gen_305931 import cmmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.drop.drop_openai_simple_evals_gen_3857b0 import drop_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ds1000.ds1000_service_eval_gen_cbc84f import ds1000_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_a58960 import gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import \
        hellaswag_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_159614 import \
        humaneval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.humanevalx.humanevalx_gen_620cfa import humanevalx_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.LCBench.lcbench_gen_5ff288 import LCBench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MathBench.mathbench_2024_gen_50a320 import mathbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import \
        sanitized_mbpp_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_openai_simple_evals_gen_b618ea import mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import \
        mmlu_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_cot_gen_d95929 import race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.scicode.scicode_gen_085b98 import SciCode_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_cot_gen_1d56df import \
        BoolQ_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.teval.teval_en_gen_1ac254 import \
        teval_datasets as teval_en_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.teval.teval_zh_gen_1ac254 import \
        teval_datasets as teval_zh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import TheoremQA_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.wikibench.wikibench_gen_0978ad import wikibench_datasets  # noqa: F401, E501

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets') and 'scicode' not in k.lower() and 'teval' not in k), [])
datasets += teval_en_datasets
datasets += teval_zh_datasets
datasets += SciCode_datasets

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

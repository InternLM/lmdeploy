from copy import deepcopy

from mmengine.config import read_base
from opencompass.models import TurboMindModel

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_few_shot_ppl import ARC_c_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bbh.bbh_gen_98fba6 import bbh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ceval.ceval_ppl import ceval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmmlu.cmmlu_ppl_041cbf import cmmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.crowspairs.crowspairs_ppl import crowspairs_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.drop.drop_gen_a2697c import drop_datasets  # noqa: F401, E501
    # Corebench v1.7
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_d21e37 import \
        GaokaoBench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_few_shot_ppl_4b5a83 import gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_ppl_59c85e import \
        hellaswag_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.humaneval.internal_humaneval_gen_ce6b06 import \
        humaneval_datasets as humaneval_v2_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.humaneval.internal_humaneval_gen_d2537e import \
        humaneval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_4shot_base_gen_43d5b6 import math_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MathBench.mathbench_2024_few_shot_mixed_4a3fd4 import \
        mathbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_742f0c import sanitized_mbpp_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_few_shot_gen_bfaf90 import mmlu_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.nq.nq_open_1shot_gen_20a989 import nq_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_few_shot_ppl import race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_few_shot_ppl import \
        BoolQ_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import TheoremQA_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_20a989 import \
        triviaqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.wikibench.wikibench_few_shot_ppl_c23d79 import \
        wikibench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import \
        winogrande_datasets  # noqa: F401, E501
    from opencompass.configs.models.baichuan.hf_baichuan_7b import models as hf_baichuan_7b  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma_7b import models as hf_gemma_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_7b import models as hf_internlm2_5_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_7b import models as hf_internlm2_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_20b import models as hf_internlm2_20b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm_7b import models as hf_internlm_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm_20b import models as hf_internlm_20b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b import \
        models as lmdeploy_internlm2_5_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.hf_llama2_7b import models as hf_llama2_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.hf_llama3_8b import models as hf_llama3_8b  # noqa: F401, E501
    from opencompass.configs.models.mistral.hf_mistral_7b_v0_1 import models as hf_mistral_7b_v0_1  # noqa: F401, E501
    from opencompass.configs.models.mistral.hf_mixtral_8x7b_v0_1 import \
        models as hf_mixtral_8x7b_v0_1  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b import models as lmdeploy_qwen2_5_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen1_5_7b import models as hf_qwen1_5_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen2_7b import models as hf_qwen2_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen_7b import models as hf_qwen_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen1_5_7b import models as lmdeploy_qwen1_5_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen2_7b import models as lmdeploy_qwen2_7b  # noqa: F401, E501
    # Summary Groups
    from opencompass.configs.summarizers.groups.cmmlu import cmmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.GaokaoBench import GaokaoBench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mathbench_v1_2024 import \
        mathbench_2024_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu_pro import mmlu_pro_summary_groups  # noqa: F401, E501

    # read models
race_datasets = [race_datasets[1]]
summarizer = dict(
    dataset_abbrs=[
        ['race-high', 'accuracy'],
        ['ARC-c', 'accuracy'],
        ['BoolQ', 'accuracy'],
        ['mmlu_pro', 'naive_average'],
        ['GPQA_diamond', 'accuracy'],
        ['cmmlu', 'naive_average'],
        ['mmlu', 'naive_average'],
        ['drop', 'accuracy'],
        ['bbh', 'naive_average'],
        ['math', 'accuracy'],
        ['openai_humaneval', 'humaneval_pass@1'],
        ['openai_humaneval_v2', 'humaneval_pass@1'],
        ['sanitized_mbpp', 'score'],
        ['wikibench-wiki-single_choice_cncircular', 'perf_4'],
        ['gsm8k', 'accuracy'],
        ['GaokaoBench', 'weighted_average'],
        ['triviaqa_wiki_1shot', 'score'],
        ['nq_open_1shot', 'score'],
        ['winogrande', 'accuracy'],
        ['hellaswag', 'accuracy'],
        ['TheoremQA', 'score'],
        '###### MathBench-A: Application Part ######',
        'college',
        'high',
        'middle',
        'primary',
        'arithmetic',
        'mathbench-a (average)',
        '###### MathBench-T: Theory Part ######',
        'college_knowledge',
        'high_knowledge',
        'middle_knowledge',
        'primary_knowledge',
        'mathbench-t (average)',
        '###### Overall: Average between MathBench-A and MathBench-T ######',
        'Overall',
        '',
        'mmlu',
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-humanities',
        'mmlu-other',
        'cmmlu',
        'cmmlu-stem',
        'cmmlu-social-science',
        'cmmlu-humanities',
        'cmmlu-other',
        'cmmlu-china-specific',
        'mmlu_pro',
        'mmlu_pro_biology',
        'mmlu_pro_business',
        'mmlu_pro_chemistry',
        'mmlu_pro_computer_science',
        'mmlu_pro_economics',
        'mmlu_pro_engineering',
        'mmlu_pro_health',
        'mmlu_pro_history',
        'mmlu_pro_law',
        'mmlu_pro_math',
        'mmlu_pro_philosophy',
        'mmlu_pro_physics',
        'mmlu_pro_psychology',
        'mmlu_pro_other',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

turbomind_qwen1_5_7b = deepcopy(*lmdeploy_qwen1_5_7b)
turbomind_qwen2_7b = deepcopy(*lmdeploy_qwen2_7b)
turbomind_qwen2_5_7b = deepcopy(*lmdeploy_qwen2_5_7b)
turbomind_qwen2_5_14b = deepcopy(*lmdeploy_qwen2_5_7b)
turbomind_qwen2_5_14b['path'] = 'Qwen/Qwen2.5-14B'
turbomind_internlm2_5_7b = deepcopy(*lmdeploy_internlm2_5_7b)
turbomind_internlm2_5_7b_4bits = deepcopy(*lmdeploy_internlm2_5_7b)
turbomind_internlm2_5_7b_batch1 = deepcopy(*lmdeploy_internlm2_5_7b)
turbomind_internlm2_5_7b_batch1_4bits = deepcopy(*lmdeploy_internlm2_5_7b)

base_model = dict(
    type=TurboMindModel,
    engine_config=dict(session_len=7168, max_batch_size=128, tp=1),
    gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
    max_seq_len=7168,
    max_out_len=1024,
    batch_size=128,
    run_cfg=dict(num_gpus=1),
)

turbomind_qwen3_8b_base = deepcopy(base_model)
pytorch_qwen3_8b_base = deepcopy(base_model)
turbomind_qwen3_8b_base_4bits = deepcopy(base_model)
turbomind_qwen3_8b_base_kvint8 = deepcopy(base_model)
for model in [
        v for k, v in locals().items()
        if k.startswith('turbomind_qwen3_8b_base') or k.startswith('pytorch_qwen3_8b_base')
]:
    model['abbr'] = 'qwen3_8b_base_turbomind'
    model['path'] = 'Qwen/Qwen3-8B-Base'
    model['run_cfg']['num_gpus'] = 1
    model['engine_config']['tp'] = 1

for model in [v for k, v in locals().items() if k.endswith('_4bits')]:
    model['engine_config']['model_format'] = 'awq'
    model['abbr'] = model['abbr'] + '_4bits'
    model['path'] = model['path'] + '-inner-4bits'

for model in [v for k, v in locals().items() if '_batch1' in k]:
    model['abbr'] = model['abbr'] + '_batch1'
    model['engine_config']['max_batch_size'] = 1
    model['batch_size'] = 1

for model in [v for k, v in locals().items() if k.startswith('pytorch_')]:
    model['abbr'] = model['abbr'].replace('turbomind', 'pytorch')
    model['backend'] = 'pytorch'

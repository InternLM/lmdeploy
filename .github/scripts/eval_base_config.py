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
    # Summary Groups
    from opencompass.configs.summarizers.groups.cmmlu import cmmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.GaokaoBench import GaokaoBench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mathbench_v1_2024 import \
        mathbench_2024_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu_pro import mmlu_pro_summary_groups  # noqa: F401, E501

    # read models
race_datasets = [race_datasets[1]]
mmlu_datasets = [
    x for x in mmlu_datasets if x['abbr'].replace('lukaemon_mmlu_', '') in [
        'business_ethics', 'clinical_knowledge', 'college_medicine', 'global_facts', 'human_aging', 'management',
        'marketing', 'medical_genetics', 'miscellaneous', 'nutrition', 'professional_accounting',
        'professional_medicine', 'virology'
    ]
]

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

base_model = dict(
    type=TurboMindModel,
    engine_config=dict(session_len=7168, tp=1),
    gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
    max_seq_len=7168,
    max_out_len=1024,
    batch_size=32,
    run_cfg=dict(num_gpus=1),
)

turbomind_qwen2_5_1_5b = deepcopy(base_model)
turbomind_qwen2_5_1_5b['path'] = 'Qwen/Qwen2.5-1.5B'
turbomind_qwen2_5_1_5b['abbr'] = 'turbomind_qwen2_5_1_5b'
turbomind_qwen2_5_7b = deepcopy(base_model)
turbomind_qwen2_5_7b['path'] = 'Qwen/Qwen2.5-7B'
turbomind_qwen2_5_7b['abbr'] = 'turbomind_qwen2_5_7b'
turbomind_qwen2_5_32b = deepcopy(base_model)
turbomind_qwen2_5_32b['path'] = 'Qwen/Qwen2.5-32B'
turbomind_qwen2_5_32b['abbr'] = 'turbomind_qwen2_5_32b'
turbomind_qwen2_5_32b['run_cfg']['num_gpus'] = 2
turbomind_qwen2_5_32b['engine_config']['tp'] = 2
turbomind_internlm2_5_7b = deepcopy(base_model)
turbomind_internlm2_5_7b['path'] = 'internlm/internlm2_5-7b-chat'
turbomind_internlm2_5_7b['abbr'] = 'turbomind_internlm2_5_7b'
turbomind_glm_4_9b = deepcopy(base_model)
turbomind_glm_4_9b['path'] = 'THUDM/glm-4-9b'
turbomind_glm_4_9b['abbr'] = 'turbomind_glm_4_9b'
turbomind_llama_3_70b = deepcopy(base_model)
turbomind_llama_3_70b['path'] = 'meta-llama/Meta-Llama-3-70B'
turbomind_llama_3_70b['abbr'] = 'turbomind_llama_3_70b'
turbomind_llama_3_70b['run_cfg']['num_gpus'] = 4
turbomind_llama_3_70b['engine_config']['tp'] = 4
turbomind_llama_3_1_8b = deepcopy(base_model)
turbomind_llama_3_1_8b['path'] = 'meta-llama/Llama-3.1-8B'
turbomind_llama_3_1_8b['abbr'] = 'turbomind_llama_3_1_8b'
turbomind_qwen3_0_6b_base = deepcopy(base_model)
turbomind_qwen3_0_6b_base['path'] = 'Qwen/Qwen3-0.6B-Base'
turbomind_qwen3_0_6b_base['abbr'] = 'turbomind_qwen3_0_6b_base'
turbomind_qwen3_8b_base = deepcopy(base_model)
turbomind_qwen3_8b_base['path'] = 'Qwen/Qwen3-8B-Base'
turbomind_qwen3_8b_base['abbr'] = 'turbomind_qwen3_8b_base'
turbomind_qwen3_30b_A3B_base = deepcopy(base_model)
turbomind_qwen3_30b_A3B_base['path'] = 'Qwen/Qwen3-30B-A3B-Base'
turbomind_qwen3_30b_A3B_base['abbr'] = 'turbomind_qwen3_30b_A3B_base'
turbomind_qwen3_30b_A3B_base['run_cfg']['num_gpus'] = 2
turbomind_qwen3_30b_A3B_base['engine_config']['tp'] = 2

pytorch_qwen2_5_1_5b = deepcopy(base_model)
pytorch_qwen2_5_1_5b['path'] = 'Qwen/Qwen2.5-1.5B'
pytorch_qwen2_5_1_5b['abbr'] = 'pytorch_qwen2_5_1_5b'
pytorch_qwen2_5_7b = deepcopy(base_model)
pytorch_qwen2_5_7b['path'] = 'Qwen/Qwen2.5-7B'
pytorch_qwen2_5_7b['abbr'] = 'pytorch_qwen2_5_7b'
pytorch_qwen2_5_32b = deepcopy(base_model)
pytorch_qwen2_5_32b['path'] = 'Qwen/Qwen2.5-32B'
pytorch_qwen2_5_32b['abbr'] = 'pytorch_qwen2_5_32b'
pytorch_qwen2_5_32b['run_cfg']['num_gpus'] = 2
pytorch_qwen2_5_32b['engine_config']['tp'] = 2
pytorch_internlm2_5_7b = deepcopy(base_model)
pytorch_internlm2_5_7b['path'] = 'internlm/internlm2_5-7b-chat'
pytorch_internlm2_5_7b['abbr'] = 'pytorch_internlm2_5_7b'
pytorch_gemma_2_9b = deepcopy(base_model)
pytorch_gemma_2_9b['path'] = 'google/gemma-2-9b'
pytorch_gemma_2_9b['abbr'] = 'pytorch_gemma_2_9b'
pytorch_llama_3_70b = deepcopy(base_model)
pytorch_llama_3_70b['path'] = 'meta-llama/Meta-Llama-3-70B'
pytorch_llama_3_70b['abbr'] = 'pytorch_llama_3_70b'
pytorch_llama_3_70b['run_cfg']['num_gpus'] = 4
pytorch_llama_3_70b['engine_config']['tp'] = 4
pytorch_llama_3_1_8b = deepcopy(base_model)
pytorch_llama_3_1_8b['path'] = 'meta-llama/Llama-3.1-8B'
pytorch_llama_3_1_8b['abbr'] = 'pytorch_llama_3_1_8b'
pytorch_qwen3_0_6b_base = deepcopy(base_model)
pytorch_qwen3_0_6b_base['path'] = 'Qwen/Qwen3-0.6B-Base'
pytorch_qwen3_0_6b_base['abbr'] = 'pytorch_qwen3_0_6b_base'
pytorch_qwen3_8b_base = deepcopy(base_model)
pytorch_qwen3_8b_base['path'] = 'Qwen/Qwen3-8B-Base'
pytorch_qwen3_8b_base['abbr'] = 'pytorch_qwen3_8b_base'
pytorch_qwen3_30b_A3B_base = deepcopy(base_model)
pytorch_qwen3_30b_A3B_base['path'] = 'Qwen/Qwen3-30B-A3B-Base'
pytorch_qwen3_30b_A3B_base['abbr'] = 'pytorch_qwen3_30b_A3B_base'
pytorch_qwen3_30b_A3B_base['run_cfg']['num_gpus'] = 2
pytorch_qwen3_30b_A3B_base['engine_config']['tp'] = 2

for model in [v for k, v in locals().items() if k.startswith('pytorch_')]:
    model['backend'] = 'pytorch'

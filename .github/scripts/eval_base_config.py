from copy import deepcopy

from mmengine.config import read_base
from opencompass.models import TurboMindModel

with read_base():
    # choose a list of datasets
    # Corebench v1.7
    from opencompass.configs.datasets.gpqa.gpqa_few_shot_ppl_4b5a83 import gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_few_shot_ppl import race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import (
        winogrande_datasets,  # noqa: F401, E501
    )

    # Summary Groups
    from opencompass.configs.summarizers.groups.cmmlu import cmmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.GaokaoBench import GaokaoBench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mathbench_v1_2024 import (
        mathbench_2024_summary_groups,  # noqa: F401, E501
    )
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
        ['GPQA_diamond', 'accuracy'],
        ['mmlu', 'naive_average'],
        ['gsm8k', 'accuracy'],
        ['winogrande', 'accuracy'],
        '###### Overall: Average between MathBench-A and MathBench-T ######',
        'Overall',
        '',
        'mmlu',
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-humanities',
        'mmlu-other',
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

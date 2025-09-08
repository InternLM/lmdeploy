from copy import deepcopy

from mmengine.config import read_base

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_ppl import race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import \
        winogrande_datasets  # noqa: F401, E501
    # read hf models - chat models
    from opencompass.configs.models.chatglm.lmdeploy_glm4_9b import models as lmdeploy_glm4_9b_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_7b_base import \
        models as lmdeploy_deepseek_7b_base_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_67b_base import \
        models as lmdeploy_deepseek_67b_base_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_v2 import lmdeploy_deepseek_v2_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.lmdeploy_gemma_9b import models as pytorch_gemma_9b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_1_8b import \
        models as lmdeploy_internlm2_1_8b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b import \
        models as lmdeploy_internlm2_5_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_20b import \
        models as lmdeploy_internlm2_20b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_base_7b import \
        models as lmdeploy_internlm2_base_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b import \
        models as lmdeploy_llama3_1_8b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_8b import \
        models as lmdeploy_llama3_8b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_70b import \
        models as lmdeploy_llama3_70b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_1_5b import \
        models as lmdeploy_qwen2_5_1_5b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b import \
        models as lmdeploy_qwen2_5_7b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_32b import \
        models as lmdeploy_qwen2_5_32b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_72b import \
        models as lmdeploy_qwen2_5_72b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen2_1_5b import \
        models as lmdeploy_qwen2_1_5b_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen2_7b import models as lmdeploy_qwen2_7b_model  # noqa: F401, E501
    from opencompass.configs.models.yi.lmdeploy_yi_1_5_9b import models as lmdeploy_yi_1_5_9b_model  # noqa: F401, E501

    from .volc import infer as volc_infer  # noqa: F401, E501

race_datasets = [race_datasets[1]]
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

pytorch_glm4_9b_model = deepcopy(lmdeploy_glm4_9b_model)
pytorch_deepseek_7b_base_model = deepcopy(lmdeploy_deepseek_7b_base_model)
pytorch_deepseek_67b_base_model = deepcopy(lmdeploy_deepseek_67b_base_model)
pytorch_deepseek_v2_model = deepcopy(lmdeploy_deepseek_v2_model)
pytorch_internlm2_5_7b_model = deepcopy(lmdeploy_internlm2_5_7b_model)
pytorch_internlm2_20b_model = deepcopy(lmdeploy_internlm2_20b_model)
pytorch_internlm2_base_7b_model = deepcopy(lmdeploy_internlm2_base_7b_model)
pytorch_llama3_1_8b_model = deepcopy(lmdeploy_llama3_1_8b_model)
pytorch_llama3_70b_model = deepcopy(lmdeploy_llama3_70b_model)
pytorch_qwen2_5_1_5b_model = deepcopy(lmdeploy_qwen2_5_1_5b_model)
pytorch_qwen2_5_72b_model = deepcopy(lmdeploy_qwen2_5_72b_model)
pytorch_qwen2_7b_model = deepcopy(lmdeploy_qwen2_7b_model)
pytorch_yi_1_5_9b_model = deepcopy(lmdeploy_yi_1_5_9b_model)
pytorch_deepseek_v2_model['engine_config']['cache_max_entry_count'] = 0.7

lmdeploy_glm4_9b_model_native = deepcopy(lmdeploy_glm4_9b_model)
lmdeploy_deepseek_7b_base_model_native = deepcopy(lmdeploy_deepseek_7b_base_model)
lmdeploy_deepseek_67b_base_model_native = deepcopy(lmdeploy_deepseek_67b_base_model)
lmdeploy_deepseek_v2_model_native = deepcopy(lmdeploy_deepseek_v2_model)
lmdeploy_internlm2_5_7b_model_native = deepcopy(lmdeploy_internlm2_5_7b_model)
lmdeploy_internlm2_20b_model_native = deepcopy(lmdeploy_internlm2_20b_model)
lmdeploy_internlm2_base_7b_model_native = deepcopy(lmdeploy_internlm2_base_7b_model)
lmdeploy_llama3_1_8b_model_native = deepcopy(lmdeploy_llama3_1_8b_model)
lmdeploy_llama3_70b_model_native = deepcopy(lmdeploy_llama3_70b_model)
lmdeploy_qwen2_5_1_5b_model_native = deepcopy(lmdeploy_qwen2_5_1_5b_model)
lmdeploy_qwen2_5_72b_model_native = deepcopy(lmdeploy_qwen2_5_72b_model)
lmdeploy_qwen2_7b_model_native = deepcopy(lmdeploy_qwen2_7b_model)
lmdeploy_yi_1_5_9b_model_native = deepcopy(lmdeploy_yi_1_5_9b_model)

for model in [v for k, v in locals().items() if k.startswith('lmdeploy_') or k.startswith('pytorch_')]:
    for m in model:
        m['engine_config']['max_batch_size'] = 512
        m['gen_config']['do_sample'] = False
        m['batch_size'] = 5000

for model in [v for k, v in locals().items() if k.startswith('lmdeploy_')]:
    for m in model:
        m['backend'] = 'turbomind'

for model in [v for k, v in locals().items() if k.startswith('pytorch_')]:
    for m in model:
        m['abbr'] = m['abbr'].replace('turbomind', 'pytorch').replace('lmdeploy', 'pytorch')
        m['backend'] = 'pytorch'

for model in [v for k, v in locals().items() if k.endswith('_native')]:
    for m in model:
        m['abbr'] = m['abbr'] + '_native'
        m['engine_config']['communicator'] = 'native'

# models = sum([v for k, v in locals().items() if  k.startswith('lmdeploy_') or k.startswith('pytorch_')], [])
# models = sorted(models, key=lambda x: x['run_cfg']['num_gpus'])

summarizer = dict(
    dataset_abbrs=[
        ['gsm8k', 'accuracy'],
        ['GPQA_diamond', 'accuracy'],
        ['race-high', 'accuracy'],
        ['winogrande', 'accuracy'],
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

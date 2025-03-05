import os
from copy import deepcopy

from mmengine.config import read_base

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_gen import race_datasets  # noqa: F401, E501
    # read hf models - chat models
    from opencompass.configs.models.chatglm.lmdeploy_glm4_9b_chat import \
        models as lmdeploy_glm4_9b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_r1_distill_qwen_32b import \
        models as lmdeploy_deepseek_r1_distill_qwen_32b_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_v2_5_1210 import \
        models as lmdeploy_deepseek_v2_5_1210_model  # noqa: F401, E501
    from opencompass.configs.models.deepseek.lmdeploy_deepseek_v2_lite import \
        models as lmdeploy_deepseek_v2_lite_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.lmdeploy_gemma_9b_it import \
        models as pytorch_gemma_9b_it_model  # noqa: F401, E501
    from opencompass.configs.models.gemma.lmdeploy_gemma_27b_it import \
        models as pytorch_gemma_27b_it_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
        models as lmdeploy_internlm2_5_7b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_20b_chat import \
        models as lmdeploy_internlm2_5_20b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_chat_1_8b import \
        models as lmdeploy_internlm2_chat_1_8b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_chat_1_8b_sft import \
        models as lmdeploy_internlm2_chat_1_8b_sft_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_chat_7b import \
        models as lmdeploy_internlm2_chat_7b_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_chat_7b_sft import \
        models as lmdeploy_internlm2_chat_7b_sft_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm3_8b_instruct import \
        models as lmdeploy_internlm3_8b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama2_7b_chat import \
        models as lmdeploy_llama2_7b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b_instruct import \
        models as lmdeploy_llama3_1_8b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_2_3b_instruct import \
        models as lmdeploy_llama3_2_3b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_3_70b_instruct import \
        models as lmdeploy_llama3_3_70b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_8b_instruct import \
        models as lmdeploy_llama3_8b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.mistral.lmdeploy_mistral_large_instruct_2411 import \
        models as lmdeploy_mistral_large_instruct_2411_model  # noqa: F401, E501
    from opencompass.configs.models.mistral.lmdeploy_mistral_nemo_instruct_2407 import \
        models as lmdeploy_mistral_nemo_instruct_2407_model  # noqa: F401, E501
    from opencompass.configs.models.mistral.lmdeploy_mistral_small_instruct_2409 import \
        models as lmdeploy_mistral_small_instruct_2409_model  # noqa: F401, E501
    from opencompass.configs.models.nvidia.lmdeploy_nemotron_70b_instruct_hf import \
        models as lmdeploy_nemotron_70b_instruct_hf_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_0_5b_instruct import \
        models as lmdeploy_qwen2_5_0_5b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_3b_instruct import \
        models as lmdeploy_qwen2_5_3b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_14b_instruct import \
        models as lmdeploy_qwen2_5_14b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_72b_instruct import \
        models as lmdeploy_qwen2_5_72b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen2_1_5b_instruct import \
        models as lmdeploy_qwen2_1_5b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen2_7b_instruct import \
        models as lmdeploy_qwen2_7b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.yi.lmdeploy_yi_1_5_6b_chat import \
        models as lmdeploy_yi_1_5_6b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.yi.lmdeploy_yi_1_5_9b_chat import \
        models as lmdeploy_yi_1_5_9b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.yi.lmdeploy_yi_1_5_34b_chat import \
        models as lmdeploy_yi_1_5_34b_chat_model  # noqa: F401, E501

    from ...volc import infer as volc_infer  # noqa: F401, E501

race_datasets = [race_datasets[1]]
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

pytorch_glm4_9b_chat_model = deepcopy(*lmdeploy_glm4_9b_chat_model)
pytorch_deepseek_v2_lite_model = deepcopy(*lmdeploy_deepseek_v2_lite_model)
pytorch_deepseek_v2_5_1210_model = deepcopy(*lmdeploy_deepseek_v2_5_1210_model)
pytorch_internlm3_8b_instruct_model = deepcopy(*lmdeploy_internlm3_8b_instruct_model)
pytorch_internlm2_5_7b_chat_model = deepcopy(*lmdeploy_internlm2_5_7b_chat_model)
pytorch_internlm2_5_20b_chat_model = deepcopy(*lmdeploy_internlm2_5_20b_chat_model)
pytorch_llama3_2_3b_instruct_model = deepcopy(*lmdeploy_llama3_2_3b_instruct_model)
pytorch_llama3_3_70b_instruct_model = deepcopy(*lmdeploy_llama3_3_70b_instruct_model)
pytorch_mistral_large_instruct_2411_model = deepcopy(*lmdeploy_mistral_large_instruct_2411_model)
pytorch_mistral_nemo_instruct_2407_model = deepcopy(*lmdeploy_mistral_nemo_instruct_2407_model)
pytorch_mistral_small_instruct_2409_model = deepcopy(*lmdeploy_mistral_small_instruct_2409_model)
pytorch_qwen2_5_72b_instruct_model = deepcopy(*lmdeploy_qwen2_5_72b_instruct_model)
pytorch_qwen2_7b_instruct_model = deepcopy(*lmdeploy_qwen2_7b_instruct_model)
pytorch_yi_1_5_34b_chat_model = deepcopy(*lmdeploy_yi_1_5_34b_chat_model)

lmdeploy_glm4_9b_chat_model_native = deepcopy(*lmdeploy_glm4_9b_chat_model)
lmdeploy_deepseek_r1_distill_qwen_32b_model_native = deepcopy(*lmdeploy_deepseek_r1_distill_qwen_32b_model)
lmdeploy_deepseek_v2_lite_model_native = deepcopy(*lmdeploy_deepseek_v2_lite_model)
lmdeploy_deepseek_v2_5_1210_model_native = deepcopy(*lmdeploy_deepseek_v2_5_1210_model)
lmdeploy_internlm3_8b_instruct_model_native = deepcopy(*lmdeploy_internlm3_8b_instruct_model)
lmdeploy_internlm2_5_7b_chat_model_native = deepcopy(*lmdeploy_internlm2_5_7b_chat_model)
lmdeploy_internlm2_5_20b_chat_model_native = deepcopy(*lmdeploy_internlm2_5_20b_chat_model)
lmdeploy_llama3_1_8b_instruct_model_native = deepcopy(*lmdeploy_llama3_1_8b_instruct_model)
lmdeploy_llama3_2_3b_instruct_model_native = deepcopy(*lmdeploy_llama3_2_3b_instruct_model)
lmdeploy_llama3_8b_instruct_model_native = deepcopy(*lmdeploy_llama3_8b_instruct_model)
lmdeploy_mistral_large_instruct_2411_model_native = deepcopy(*lmdeploy_mistral_large_instruct_2411_model)
lmdeploy_mistral_nemo_instruct_2407_model_native = deepcopy(*lmdeploy_mistral_nemo_instruct_2407_model)
lmdeploy_mistral_small_instruct_2409_model_native = deepcopy(*lmdeploy_mistral_small_instruct_2409_model)
lmdeploy_nemotron_70b_instruct_hf_model_native = deepcopy(*lmdeploy_nemotron_70b_instruct_hf_model)
lmdeploy_qwen2_5_0_5b_instruct_model_native = deepcopy(*lmdeploy_qwen2_5_0_5b_instruct_model)
lmdeploy_qwen2_5_14b_instruct_model_native = deepcopy(*lmdeploy_qwen2_5_14b_instruct_model)
lmdeploy_qwen2_5_72b_instruct_model_native = deepcopy(*lmdeploy_qwen2_5_72b_instruct_model)
lmdeploy_qwen2_7b_instruct_model_native = deepcopy(*lmdeploy_qwen2_7b_instruct_model)
lmdeploy_yi_1_5_6b_chat_model_native = deepcopy(*lmdeploy_yi_1_5_6b_chat_model)
lmdeploy_yi_1_5_34b_chat_model_native = deepcopy(*lmdeploy_yi_1_5_34b_chat_model)

for model in [v for k, v in locals().items() if k.startswith('lmdeploy_')]:
    model['engine_config']['max_batch_size'] = 512
    model['backend'] = 'turbomind'
    model['gen_config']['do_sample'] = False
    model['batch_size'] = 5000

for model in [v for k, v in locals().items() if k.startswith('pytorch_')]:
    model['engine_config']['max_batch_size'] = 512
    model['backend'] = 'pytorch'
    model['batch_size'] = 5000

for model in [v for k, v in locals().items() if k.endswith('_native')]:
    model['engine_config']['communicator'] = 'native'

if os['TEST_BACKEND'] is not None or os['TEST_BACKEND'] == 'pytorch':
    models = [v for k, v in locals().items() if k.startswith('pytorch_')]
else:
    models = [v for k, v in locals().items() if k.startswith('lmdeploy_')]

models = sorted(models, key=lambda x: x['run_cfg']['num_gpus'])

summarizer = dict(
    dataset_abbrs=[
        'gsm8k',
        'race-middle',
        'race-high',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

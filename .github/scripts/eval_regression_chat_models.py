from copy import deepcopy

from mmengine.config import read_base

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_0shot_gen_11c4b5 import math_datasets  # noqa: F401, E501
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
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_32b_instruct import \
        models as lmdeploy_qwen2_5_32b_instruct_model  # noqa: F401, E501
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

    from .volc import infer as volc_infer  # noqa: F401, E501

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])

pytorch_glm4_9b_chat_model = deepcopy(lmdeploy_glm4_9b_chat_model)
pytorch_deepseek_v2_lite_model = deepcopy(lmdeploy_deepseek_v2_lite_model)
pytorch_deepseek_v2_5_1210_model = deepcopy(lmdeploy_deepseek_v2_5_1210_model)
pytorch_internlm3_8b_instruct_model = deepcopy(lmdeploy_internlm3_8b_instruct_model)
pytorch_internlm2_5_7b_chat_model = deepcopy(lmdeploy_internlm2_5_7b_chat_model)
pytorch_internlm2_5_20b_chat_model = deepcopy(lmdeploy_internlm2_5_20b_chat_model)
pytorch_llama3_2_3b_instruct_model = deepcopy(lmdeploy_llama3_2_3b_instruct_model)
pytorch_llama3_3_70b_instruct_model = deepcopy(lmdeploy_llama3_3_70b_instruct_model)
pytorch_mistral_nemo_instruct_2407_model = deepcopy(lmdeploy_mistral_nemo_instruct_2407_model)
pytorch_mistral_small_instruct_2409_model = deepcopy(lmdeploy_mistral_small_instruct_2409_model)
pytorch_qwen2_5_72b_instruct_model = deepcopy(lmdeploy_qwen2_5_72b_instruct_model)
pytorch_qwen2_5_32b_instruct_model = deepcopy(lmdeploy_qwen2_5_32b_instruct_model)
pytorch_qwen2_7b_instruct_model = deepcopy(lmdeploy_qwen2_7b_instruct_model)
pytorch_yi_1_5_34b_chat_model = deepcopy(lmdeploy_yi_1_5_34b_chat_model)

lmdeploy_glm4_9b_chat_model_native = deepcopy(lmdeploy_glm4_9b_chat_model)
lmdeploy_deepseek_r1_distill_qwen_32b_model_native = deepcopy(lmdeploy_deepseek_r1_distill_qwen_32b_model)
lmdeploy_deepseek_v2_lite_model_native = deepcopy(lmdeploy_deepseek_v2_lite_model)
lmdeploy_deepseek_v2_5_1210_model_native = deepcopy(lmdeploy_deepseek_v2_5_1210_model)
lmdeploy_internlm3_8b_instruct_model_native = deepcopy(lmdeploy_internlm3_8b_instruct_model)
lmdeploy_internlm2_5_7b_chat_model_native = deepcopy(lmdeploy_internlm2_5_7b_chat_model)
lmdeploy_internlm2_5_20b_chat_model_native = deepcopy(lmdeploy_internlm2_5_20b_chat_model)
lmdeploy_llama3_1_8b_instruct_model_native = deepcopy(lmdeploy_llama3_1_8b_instruct_model)
lmdeploy_llama3_2_3b_instruct_model_native = deepcopy(lmdeploy_llama3_2_3b_instruct_model)
lmdeploy_llama3_8b_instruct_model_native = deepcopy(lmdeploy_llama3_8b_instruct_model)
lmdeploy_llama3_3_70b_instruct_model_native = deepcopy(lmdeploy_llama3_3_70b_instruct_model)
lmdeploy_mistral_large_instruct_2411_model_native = deepcopy(lmdeploy_mistral_large_instruct_2411_model)
lmdeploy_mistral_nemo_instruct_2407_model_native = deepcopy(lmdeploy_mistral_nemo_instruct_2407_model)
lmdeploy_mistral_small_instruct_2409_model_native = deepcopy(lmdeploy_mistral_small_instruct_2409_model)
lmdeploy_nemotron_70b_instruct_hf_model_native = deepcopy(lmdeploy_nemotron_70b_instruct_hf_model)
lmdeploy_qwen2_5_0_5b_instruct_model_native = deepcopy(lmdeploy_qwen2_5_0_5b_instruct_model)
lmdeploy_qwen2_5_14b_instruct_model_native = deepcopy(lmdeploy_qwen2_5_14b_instruct_model)
lmdeploy_qwen2_5_32b_instruct_model_native = deepcopy(lmdeploy_qwen2_5_32b_instruct_model)
lmdeploy_qwen2_5_72b_instruct_model_native = deepcopy(lmdeploy_qwen2_5_72b_instruct_model)
lmdeploy_qwen2_7b_instruct_model_native = deepcopy(lmdeploy_qwen2_7b_instruct_model)
lmdeploy_yi_1_5_6b_chat_model_native = deepcopy(lmdeploy_yi_1_5_6b_chat_model)
lmdeploy_yi_1_5_34b_chat_model_native = deepcopy(lmdeploy_yi_1_5_34b_chat_model)

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
        ['GPQA_diamond', 'accuracy'],
        ['math', 'accuracy'],
        ['IFEval', 'Prompt-level-strict-accuracy'],
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

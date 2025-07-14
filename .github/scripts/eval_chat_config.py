from copy import deepcopy

from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import bbh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ceval.ceval_gen_2daf24 import ceval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.crowspairs.crowspairs_gen_381af0 import crowspairs_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_4c31db import \
        GaokaoBench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import \
        hellaswag_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_a0fc46 import sanitized_mbpp_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import \
        mmlu_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.nq.nq_open_1shot_gen_01cf41 import nq_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_gen_69ee4f import race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import TheoremQA_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_eaf81e import \
        triviaqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_gen_b36770 import \
        winogrande_datasets  # noqa: F401, E501
    # read models
    from opencompass.configs.models.baichuan.hf_baichuan2_7b_chat import \
        models as hf_baichuan2_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma2_9b_it import models as hf_gemma2_9b_it  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_7b_chat import \
        models as hf_internlm2_5_7b_chat  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_20b_chat import \
        models as hf_internlm2_5_20b_chat  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_chat_7b import \
        models as hf_internlm2_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_chat_20b import \
        models as hf_internlm2_chat_20b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
        models as lmdeploy_internlm2_5_7b_chat  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_20b_chat import \
        models as lmdeploy_internlm2_5_20b_chat  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_chat_7b import \
        models as lmdeploy_internlm2_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_chat_20b import \
        models as lmdeploy_internlm2_chat_20b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm3_8b_instruct import \
        models as lmdeploy_internlm3_8b_instruct  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm_chat_7b import \
        models as lmdeploy_internlm_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.hf_llama2_7b_chat import models as hf_llama2_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.hf_llama3_1_8b_instruct import \
        models as hf_llama3_1_8b_instruct  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.hf_llama3_8b_instruct import \
        models as hf_llama_3_8b_instruct  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama2_7b_chat import \
        models as lmdeploy_llama2_7b_chat  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b_instruct import \
        models as lmdeploy_llama3_1_8b_instruct  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_8b_instruct import \
        models as lmdeploy_llama3_8b_instruct  # noqa: F401, E501
    from opencompass.configs.models.mistral.hf_mistral_7b_instruct_v0_1 import \
        models as hf_mistral_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.mistral.hf_mixtral_8x7b_instruct_v0_1 import \
        models as hf_mixtral_chat_8x7b  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import \
        models as lmdeploy_qwen2_5_7b_instruct  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_32b_instruct import \
        models as lmdeploy_qwen2_5_32b_instruct  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen1_5_7b_chat import models as hf_qwen1_5_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen1_5_moe_a2_7b_chat import \
        models as hf_qwen1_5_moe_a2_7b_chat  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen2_7b_instruct import models as hf_qwen2_7b_instruct  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen_7b_chat import models as hf_qwen_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen1_5_7b_chat import \
        models as lmdeploy_qwen1_5_7b_chat  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen2_7b_instruct import \
        models as lmdeploy_qwen2_7b_instruct  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen_7b_chat import \
        models as lmdeploy_qwen_7b_chat  # noqa: F401, E501
    # Summary Groups
    from opencompass.configs.summarizers.groups.bbh import bbh_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.cmmlu import cmmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.ds1000 import ds1000_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.GaokaoBench import GaokaoBench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.humanevalx import humanevalx_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mathbench_v1_2024 import \
        mathbench_2024_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu_pro import mmlu_pro_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.scicode import scicode_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.teval import teval_summary_groups  # noqa: F401, E501

llama2_meta_template = dict(round=[
    dict(role='HUMAN', begin='[INST] ', end=' [/INST]'),
    dict(role='BOT', begin='', end='', generate=True),
],
                            eos_token_id=2)

MAX_SESSION_LEN = 2048
MAX_NEW_TOKENS = 1024

# ===== Configs for internlm/internlm2-chat-7b =====
turbomind_internlm2_chat_7b = deepcopy(*lmdeploy_internlm2_chat_7b)
turbomind_internlm2_chat_7b_4bits = deepcopy(*lmdeploy_internlm2_chat_7b)
turbomind_internlm2_chat_7b_kvint4 = deepcopy(*lmdeploy_internlm2_chat_7b)
turbomind_internlm2_chat_7b_kvint8 = deepcopy(*lmdeploy_internlm2_chat_7b)
pytorch_internlm2_chat_7b = deepcopy(*lmdeploy_internlm2_chat_7b)

# ===== Configs for internlm/internlm2_5_7b_chat =====
turbomind_internlm2_5_7b_chat = deepcopy(*lmdeploy_internlm2_5_7b_chat)
turbomind_internlm2_5_7b_chat_4bits = deepcopy(*lmdeploy_internlm2_5_7b_chat)
turbomind_internlm2_5_7b_chat_kvint4 = deepcopy(*lmdeploy_internlm2_5_7b_chat)
turbomind_internlm2_5_7b_chat_kvint8 = deepcopy(*lmdeploy_internlm2_5_7b_chat)
pytorch_internlm2_5_7b_chat = deepcopy(*lmdeploy_internlm2_5_7b_chat)
pytorch_internlm2_5_7b_chat_w8a8 = deepcopy(*lmdeploy_internlm2_5_7b_chat)
turbomind_internlm2_5_7b_chat_batch1 = deepcopy(*lmdeploy_internlm2_5_7b_chat)
turbomind_internlm2_5_7b_chat_batch1_4bits = deepcopy(*lmdeploy_internlm2_5_7b_chat)

turbomind_internlm3_8b_instruct = deepcopy(*lmdeploy_internlm3_8b_instruct)
turbomind_internlm3_8b_instruct_4bits = deepcopy(*lmdeploy_internlm3_8b_instruct)
turbomind_internlm3_8b_instruct_kvint4 = deepcopy(*lmdeploy_internlm3_8b_instruct)
turbomind_internlm3_8b_instruct_kvint8 = deepcopy(*lmdeploy_internlm3_8b_instruct)
pytorch_internlm3_8b_instruct = deepcopy(*lmdeploy_internlm3_8b_instruct)
pytorch_internlm3_8b_instruct_w8a8 = deepcopy(*lmdeploy_internlm3_8b_instruct)

# ===== Configs for internlm/internlm2_5_20b_chat =====
turbomind_internlm2_5_20b_chat = deepcopy(*lmdeploy_internlm2_5_20b_chat)
turbomind_internlm2_5_20b_chat_4bits = deepcopy(*lmdeploy_internlm2_5_20b_chat)
turbomind_internlm2_5_20b_chat_kvint4 = deepcopy(*lmdeploy_internlm2_5_20b_chat)
turbomind_internlm2_5_20b_chat_kvint8 = deepcopy(*lmdeploy_internlm2_5_20b_chat)
pytorch_internlm2_5_20b_chat = deepcopy(*lmdeploy_internlm2_5_20b_chat)

# ===== Configs for internlm/internlm2_chat_20b =====
turbomind_internlm2_chat_20b = deepcopy(*lmdeploy_internlm2_chat_20b)
turbomind_internlm2_chat_20b_4bits = deepcopy(*lmdeploy_internlm2_chat_20b)
turbomind_internlm2_chat_20b_kvint4 = deepcopy(*lmdeploy_internlm2_chat_20b)
turbomind_internlm2_chat_20b_kvint8 = deepcopy(*lmdeploy_internlm2_chat_20b)
pytorch_internlm2_chat_20b = deepcopy(*lmdeploy_internlm2_chat_20b)

# ===== Configs for Qwen/Qwen1.5-7B-Chat =====
turbomind_qwen1_5_7b_chat = deepcopy(*lmdeploy_qwen1_5_7b_chat)
turbomind_qwen1_5_7b_chat_4bits = deepcopy(*lmdeploy_qwen1_5_7b_chat)
turbomind_qwen1_5_7b_chat_kvint4 = deepcopy(*lmdeploy_qwen1_5_7b_chat)
turbomind_qwen1_5_7b_chat_kvint8 = deepcopy(*lmdeploy_qwen1_5_7b_chat)
pytorch_qwen1_5_7b_chat = deepcopy(*lmdeploy_qwen1_5_7b_chat)

# ===== Configs for Qwen/Qwen-7B-Chat =====
turbomind_qwen_7b_chat = deepcopy(*lmdeploy_qwen_7b_chat)
turbomind_qwen_7b_chat_4bits = deepcopy(*lmdeploy_qwen_7b_chat)
turbomind_qwen_7b_chat_kvint4 = deepcopy(*lmdeploy_qwen_7b_chat)
turbomind_qwen_7b_chat_kvint8 = deepcopy(*lmdeploy_qwen_7b_chat)
pytorch_qwen_7b_chat = deepcopy(*lmdeploy_qwen_7b_chat)

# ===== Configs for meta-llama/Meta-Llama-3-8B-Instruct =====
turbomind_llama3_8b_instruct = deepcopy(*lmdeploy_llama3_8b_instruct)
turbomind_llama3_8b_instruct_4bits = deepcopy(*lmdeploy_llama3_8b_instruct)
turbomind_llama3_8b_instruct_kvint4 = deepcopy(*lmdeploy_llama3_8b_instruct)
turbomind_llama3_8b_instruct_kvint8 = deepcopy(*lmdeploy_llama3_8b_instruct)
pytorch_llama3_8b_instruct = deepcopy(*lmdeploy_llama3_8b_instruct)

# ===== Configs for meta-llama/Meta-Llama-3.1-8B-Instruct =====
turbomind_llama3_1_8b_instruct = deepcopy(*lmdeploy_llama3_1_8b_instruct)
turbomind_llama3_1_8b_instruct['path'] = 'meta-llama/Meta-Llama-3-1-8B-Instruct'
turbomind_llama3_1_8b_instruct_4bits = deepcopy(turbomind_llama3_1_8b_instruct)
turbomind_llama3_1_8b_instruct_kvint4 = deepcopy(turbomind_llama3_1_8b_instruct)
turbomind_llama3_1_8b_instruct_kvint8 = deepcopy(turbomind_llama3_1_8b_instruct)
pytorch_llama3_1_8b_instruct = deepcopy(turbomind_llama3_1_8b_instruct)
pytorch_llama3_1_8b_instruct_w8a8 = deepcopy(turbomind_llama3_1_8b_instruct)

# ===== Configs for Qwen/Qwen2-7B-Instruct =====
turbomind_qwen2_7b_instruct = deepcopy(*lmdeploy_qwen2_7b_instruct)
turbomind_qwen2_7b_instruct_4bits = deepcopy(*lmdeploy_qwen2_7b_instruct)
turbomind_qwen2_7b_instruct_kvint4 = deepcopy(*lmdeploy_qwen2_7b_instruct)
turbomind_qwen2_7b_instruct_kvint8 = deepcopy(*lmdeploy_qwen2_7b_instruct)
pytorch_qwen2_7b_instruct = deepcopy(*lmdeploy_qwen2_7b_instruct)
pytorch_qwen2_7b_instruct_w8a8 = deepcopy(*lmdeploy_qwen2_7b_instruct)

# ===== Configs for Qwen/Qwen25-7B-Instruct =====
turbomind_qwen2_5_7b_instruct = deepcopy(*lmdeploy_qwen2_5_7b_instruct)
turbomind_qwen2_5_7b_instruct_4bits = deepcopy(*lmdeploy_qwen2_5_7b_instruct)
turbomind_qwen2_5_7b_instruct_kvint4 = deepcopy(*lmdeploy_qwen2_5_7b_instruct)
turbomind_qwen2_5_7b_instruct_kvint8 = deepcopy(*lmdeploy_qwen2_5_7b_instruct)
pytorch_qwen2_5_7b_instruct = deepcopy(*lmdeploy_qwen2_5_7b_instruct)
pytorch_qwen2_5_7b_instruct_w8a8 = deepcopy(*lmdeploy_qwen2_5_7b_instruct)

# ===== Configs for Qwen/Qwen25-32B-Instruct =====
turbomind_qwen2_5_32b_instruct = deepcopy(*lmdeploy_qwen2_5_32b_instruct)
turbomind_qwen2_5_32b_instruct_4bits = deepcopy(*lmdeploy_qwen2_5_32b_instruct)
turbomind_qwen2_5_32b_instruct_kvint4 = deepcopy(*lmdeploy_qwen2_5_32b_instruct)
turbomind_qwen2_5_32b_instruct_kvint8 = deepcopy(*lmdeploy_qwen2_5_32b_instruct)
pytorch_qwen2_5_32b_instruct = deepcopy(*lmdeploy_qwen2_5_32b_instruct)
pytorch_qwen2_5_32b_instruct_w8a8 = deepcopy(*lmdeploy_qwen2_5_32b_instruct)

# ===== Configs for meta-llama/Llama-2-7b-chat-hf =====
turbomind_llama2_7b_chat = deepcopy(*lmdeploy_llama2_7b_chat)
turbomind_llama2_7b_chat_4bits = deepcopy(*lmdeploy_llama2_7b_chat)
turbomind_llama2_7b_chat_kvint4 = deepcopy(*lmdeploy_llama2_7b_chat)
turbomind_llama2_7b_chat_kvint8 = deepcopy(*lmdeploy_llama2_7b_chat)

base_model = dict(type=TurboMindModelwithChatTemplate,
                  engine_config=dict(session_len=32768, max_batch_size=256),
                  gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=32768),
                  max_seq_len=32768,
                  max_out_len=32768,
                  batch_size=500,
                  pred_postprocessor=dict(type=extract_non_reasoning_content),
                  run_cfg=dict(num_gpus=1))

turbomind_qwen3_32b = deepcopy(base_model)
pytorch_qwen3_32b = deepcopy(base_model)
turbomind_qwen3_32b_4bits = deepcopy(base_model)
turbomind_qwen3_32b_kvint8 = deepcopy(base_model)

turbomind_qwen3_30b_a3b = deepcopy(base_model)
pytorch_qwen3_30b_a3b = deepcopy(base_model)
turbomind_qwen3_30b_a3b_4bits = deepcopy(base_model)
turbomind_qwen3_30b_a3b_kvint8 = deepcopy(base_model)
turbomind_qwen3_30b_a3b_fp8 = deepcopy(base_model)
pytorch_qwen3_30b_a3b_fp8 = deepcopy(base_model)
turbomind_qwen3_30b_a3b_fp8['engine_config']['cache_max_entry_count'] = 0.6

turbomind_qwen3_235b_a22b = deepcopy(base_model)
pytorch_qwen3_235b_a22b = deepcopy(base_model)
turbomind_qwen3_235b_a22b_4bits = deepcopy(base_model)
turbomind_qwen3_235b_a22b_kvint8 = deepcopy(base_model)
turbomind_qwen3_235b_a22b_fp8 = deepcopy(base_model)
pytorch_qwen3_235b_a22b_fp8 = deepcopy(base_model)

for model in [v for k, v in locals().items() if k.startswith('turbomind_')]:
    model['engine_config']['max_batch_size'] = 512
    model['gen_config']['do_sample'] = False
    model['batch_size'] = 1000

for model in [v for k, v in locals().items() if k.endswith('_4bits')]:
    model['engine_config']['model_format'] = 'awq'
    model['abbr'] = model['abbr'] + '_4bits'
    model['path'] = model['path'] + '-inner-4bits'

for model in [v for k, v in locals().items() if k.endswith('_w8a8')]:
    model['abbr'] = model['abbr'] + '_w8a8'
    model['path'] = model['path'] + '-inner-w8a8'

for model in [v for k, v in locals().items() if k.endswith('_kvint4')]:
    model['engine_config']['quant_policy'] = 4
    model['abbr'] = model['abbr'] + '_kvint4'

for model in [v for k, v in locals().items() if k.endswith('_kvint8')]:
    model['engine_config']['quant_policy'] = 8
    model['abbr'] = model['abbr'] + '_kvint8'

for model in [v for k, v in locals().items() if k.startswith('pytorch_')]:
    model['abbr'] = model['abbr'].replace('turbomind', 'pytorch')
    model['backend'] = 'pytorch'
    model['engine_config']['max_batch_size'] = 512
    model['gen_config']['do_sample'] = False
    model['batch_size'] = 1000

for model in [v for k, v in locals().items() if '_batch1' in k]:
    model['abbr'] = model['abbr'] + '_batch1'
    model['engine_config']['max_batch_size'] = 1
    model['batch_size'] = 1

# update config for Qwen3-32B, Qwen3-30B-A3B, Qwen3-235B-A22B
for model in [
        v for k, v in locals().items() if k.startswith('turbomind_qwen3_32b') or k.startswith('pytorch_qwen3_32b')
]:
    model['abbr'] = 'qwen3_32b_turbomind'
    model['path'] = 'Qwen/Qwen3-32B'
    model['run_cfg']['num_gpus'] = 2
    model['engine_config']['tp'] = 2
    model['engine_config']['max_batch_size'] = 1024
    model['batch_size'] = 2048

for model in [
        v for k, v in locals().items()
        if k.startswith('turbomind_qwen3_30b_a3b') or k.startswith('pytorch_qwen3_30b_a3b')
]:
    model['abbr'] = 'qwen3_30b_a3b_turbomind'
    model['path'] = 'Qwen/Qwen3-30B-A3B'
    model['run_cfg']['num_gpus'] = 2
    model['engine_config']['tp'] = 2
    model['engine_config']['max_batch_size'] = 1024
    model['batch_size'] = 2048

for model in [
        v for k, v in locals().items()
        if k.startswith('turbomind_qwen3_30b_a3b_fp8') or k.startswith('pytorch_qwen3_30b_a3b_fp8')
]:
    model['abbr'] = 'qwen3_30b_a3b_fp8_turbomind'
    model['path'] = 'Qwen/Qwen3-30B-A3B-FP8'

for model in [
        v for k, v in locals().items()
        if k.startswith('turbomind_qwen3_235b_a22b') or k.startswith('pytorch_qwen3_235b_a22b')
]:
    model['abbr'] = 'qwen3_235b_a22b_turbomind'
    model['path'] = 'Qwen/Qwen3-235B-A22B'
    model['run_cfg']['num_gpus'] = 8
    model['engine_config']['tp'] = 8
    model['engine_config']['max_batch_size'] = 1024
    model['batch_size'] = 2048

for model in [
        v for k, v in locals().items()
        if k.startswith('turbomind_qwen3_235b_a22b_fp8') or k.startswith('pytorch_qwen3_235b_a22b_fp8')
]:
    model['abbr'] = 'qwen3_235b_a22b_fp8_turbomind'
    model['path'] = 'Qwen/Qwen3-235B-A22B-FP8'

turbomind_qwen3_235b_a22b_fp8['engine_config']['cache_max_entry_count'] = 0.6
turbomind_qwen3_235b_a22b_fp8['engine_config']['tp'] = 4
turbomind_qwen3_235b_a22b_fp8['run_cfg']['num_gpus'] = 4
pytorch_qwen3_235b_a22b_fp8['engine_config']['tp'] = 4
pytorch_qwen3_235b_a22b_fp8['run_cfg']['num_gpus'] = 4

basic_pytorch_chat_tp1 = dict(type=TurboMindModelwithChatTemplate,
                              engine_config=dict(session_len=MAX_SESSION_LEN, max_batch_size=512, tp=1),
                              gen_config=dict(do_sample=False, max_new_tokens=MAX_NEW_TOKENS),
                              max_out_len=MAX_NEW_TOKENS,
                              max_seq_len=MAX_SESSION_LEN,
                              batch_size=1000,
                              run_cfg=dict(num_gpus=1))

# ===== Configs for Qwen/Qwen1.5-MoE-A2.7B-Chat =====
pytorch_qwen1_5_moe_2_7b_chat = deepcopy(basic_pytorch_chat_tp1)
pytorch_qwen1_5_moe_2_7b_chat['abbr'] = 'pytorch_qwen1_5_moe_2_7b_chat'
pytorch_qwen1_5_moe_2_7b_chat['path'] = 'Qwen/Qwen1.5-MoE-A2.7B-Chat'

# ===== Configs for google/gemma2-7b-it =====
pytorch_gemma_2_9b_it = deepcopy(basic_pytorch_chat_tp1)
pytorch_gemma_2_9b_it['abbr'] = 'pytorch_gemma_2_9b_it'
pytorch_gemma_2_9b_it['path'] = 'google/gemma-2-9b-it'

# ===== Configs for google/gemma2-27b-it =====
pytorch_gemma_2_27b_it = deepcopy(basic_pytorch_chat_tp1)
pytorch_gemma_2_27b_it['abbr'] = 'pytorch_gemma_2_27b_it'
pytorch_gemma_2_27b_it['path'] = 'google/gemma-2-27b-it'
pytorch_gemma_2_27b_it['run_cfg']['num_gpus'] = 2
pytorch_gemma_2_27b_it['engine_config']['tp'] = 2

race_datasets = [race_datasets[1]]

# Summarizer
summarizer = dict(
    dataset_abbrs=[
        ['race-high', 'accuracy'],
        ['ARC-c', 'accuracy'],
        ['BoolQ', 'accuracy'],
        ['mmlu_pro', 'naive_average'],
        ['drop', 'accuracy'],
        ['bbh', 'naive_average'],
        ['GPQA_diamond', 'accuracy'],
        ['math', 'accuracy'],
        ['wikibench-wiki-single_choice_cncircular', 'perf_4'],
        ['openai_humaneval', 'humaneval_pass@1'],
        ['sanitized_mbpp', 'score'],
        ['cmmlu', 'naive_average'],
        ['mmlu', 'naive_average'],
        ['teval', 'naive_average'],
        ['SciCode', 'accuracy'],
        ['SciCode', 'sub_accuracy'],
        ['humanevalx', 'naive_average'],
        ['ds1000', 'naive_average'],
        ['IFEval', 'Prompt-level-strict-accuracy'],
        ['gsm8k', 'accuracy'],
        ['GaokaoBench', 'weighted_average'],
        ['triviaqa_wiki_1shot', 'score'],
        ['nq_open_1shot', 'score'],
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
        ''
        'mmlu',
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-humanities',
        'mmlu-other',
        '',
        'cmmlu',
        'cmmlu-stem',
        'cmmlu-social-science',
        'cmmlu-humanities',
        'cmmlu-other',
        'cmmlu-china-specific',
        '',
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
        '',
        'humanevalx-python',
        'humanevalx-cpp',
        'humanevalx-go',
        'humanevalx-java',
        'humanevalx-js',
        '',
        'ds1000_Pandas',
        'ds1000_Numpy',
        'ds1000_Tensorflow',
        'ds1000_Scipy',
        'ds1000_Sklearn',
        'ds1000_Pytorch',
        'ds1000_Matplotlib',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

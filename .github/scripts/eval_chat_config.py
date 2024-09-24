from copy import deepcopy

from mmengine.config import read_base
from opencompass.models import TurboMindModel, TurboMindModelwithChatTemplate

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import \
        bbh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ceval.ceval_gen_2daf24 import \
        ceval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmmlu.cmmlu_gen_c13365 import \
        cmmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.crowspairs.crowspairs_gen_381af0 import \
        crowspairs_datasets  # noqa: F401, E501
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
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_a0fc46 import \
        sanitized_mbpp_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import \
        mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.nq.nq_open_1shot_gen_01cf41 import \
        nq_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_gen_69ee4f import \
        race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import \
        TheoremQA_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_eaf81e import \
        triviaqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_gen_b36770 import \
        winogrande_datasets  # noqa: F401, E501
    # read models
    from opencompass.configs.models.baichuan.hf_baichuan2_7b_chat import \
        models as hf_baichuan2_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma2_9b_it import \
        models as hf_gemma2_9b_it  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_7b_chat import \
        models as hf_internlm2_5_7b_chat  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_20b_chat import \
        models as hf_internlm2_5_20b_chat  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_chat_7b import \
        models as hf_internlm2_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_chat_20b import \
        models as hf_internlm2_chat_20b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm_chat_7b import \
        models as hf_internlm_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm_chat_20b import \
        models as hf_internlm_chat_20b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
        models as lmdeploy_internlm2_5_7b_chat  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_20b_chat import \
        models as lmdeploy_internlm2_5_20b_chat  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_chat_7b import \
        models as lmdeploy_internlm2_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_chat_20b import \
        models as lmdeploy_internlm2_chat_20b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm_chat_7b import \
        models as lmdeploy_internlm_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.hf_llama2_7b_chat import \
        models as hf_llama2_chat_7b  # noqa: F401, E501
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
    from opencompass.configs.models.qwen.hf_qwen1_5_7b_chat import \
        models as hf_qwen1_5_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen1_5_moe_a2_7b_chat import \
        models as hf_qwen1_5_moe_a2_7b_chat  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen2_7b_instruct import \
        models as hf_qwen2_7b_instruct  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen_7b_chat import \
        models as hf_qwen_chat_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen1_5_7b_chat import \
        models as lmdeploy_qwen1_5_7b_chat  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen2_7b_instruct import \
        models as lmdeploy_qwen2_7b_instruct  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen_7b_chat import \
        models as lmdeploy_qwen_7b_chat  # noqa: F401, E501
    # and output the results in a chosen format
    from opencompass.configs.summarizers.medium import \
        summarizer  # noqa: F401, E501

llama2_meta_template = dict(round=[
    dict(role='HUMAN', begin='[INST] ', end=' [/INST]'),
    dict(role='BOT', begin='', end='', generate=True),
],
                            eos_token_id=2)

MAX_SESSION_LEN = 2048
MAX_NEW_TOKENS = 1024

# ===== Configs for internlm/internlm-chat-7b =====
turbomind_internlm_chat_7b = deepcopy(*lmdeploy_internlm_chat_7b)
turbomind_internlm_chat_7b_4bits = deepcopy(*lmdeploy_internlm_chat_7b)
pytorch_internlm_chat_7b = deepcopy(*lmdeploy_internlm_chat_7b)

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
turbomind_internlm2_5_7b_chat_batch1 = deepcopy(*lmdeploy_internlm2_5_7b_chat)
pytorch_internlm2_5_7b_chat = deepcopy(*lmdeploy_internlm2_5_7b_chat)

# ===== Configs for internlm/internlm2_5_20b_chat =====
turbomind_internlm2_5_20b_chat = deepcopy(*lmdeploy_internlm2_5_20b_chat)
turbomind_internlm2_5_20b_chat_4bits = deepcopy(*lmdeploy_internlm2_5_20b_chat)
turbomind_internlm2_5_20b_chat_kvint4 = deepcopy(
    *lmdeploy_internlm2_5_20b_chat)
turbomind_internlm2_5_20b_chat_kvint8 = deepcopy(
    *lmdeploy_internlm2_5_20b_chat)
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

# ===== Configs for meta-llama/Llama-2-7b-chat-hf =====
turbomind_llama2_7b_chat = dict(type=TurboMindModel,
                                abbr='tb_llama2_chat_7b',
                                path='meta-llama/Llama-2-7b-chat-hf',
                                engine_config=dict(session_len=MAX_SESSION_LEN,
                                                   max_batch_size=128),
                                gen_config=dict(top_k=1,
                                                top_p=0.8,
                                                temperature=1.0,
                                                max_new_tokens=MAX_NEW_TOKENS),
                                max_out_len=MAX_NEW_TOKENS,
                                max_seq_len=MAX_SESSION_LEN,
                                batch_size=128,
                                concurrency=128,
                                meta_template=llama2_meta_template,
                                run_cfg=dict(num_gpus=1),
                                end_str='[INST]')
turbomind_llama2_7b_chat_4bits = deepcopy(turbomind_llama2_7b_chat)
turbomind_llama2_7b_chat_kvint4 = deepcopy(turbomind_llama2_7b_chat)
turbomind_llama2_7b_chat_kvint8 = deepcopy(turbomind_llama2_7b_chat)

# ===== Configs for meta-llama/Meta-Llama-3-8B-Instruct =====
turbomind_llama3_8b_instruct = deepcopy(*lmdeploy_llama3_8b_instruct)
turbomind_llama3_8b_instruct_4bits = deepcopy(*lmdeploy_llama3_8b_instruct)
turbomind_llama3_8b_instruct_kvint4 = deepcopy(*lmdeploy_llama3_8b_instruct)
turbomind_llama3_8b_instruct_kvint8 = deepcopy(*lmdeploy_llama3_8b_instruct)
pytorch_llama3_8b_instruct = deepcopy(*lmdeploy_llama3_8b_instruct)

# ===== Configs for meta-llama/Meta-Llama-3.1-8B-Instruct =====
turbomind_llama3_1_8b_instruct = deepcopy(*lmdeploy_llama3_1_8b_instruct)
turbomind_llama3_1_8b_instruct[
    'path'] = 'meta-llama/Meta-Llama-3-1-8B-Instruct'
turbomind_llama3_1_8b_instruct_4bits = deepcopy(turbomind_llama3_1_8b_instruct)
turbomind_llama3_1_8b_instruct_kvint4 = deepcopy(
    turbomind_llama3_1_8b_instruct)
turbomind_llama3_1_8b_instruct_kvint8 = deepcopy(
    turbomind_llama3_1_8b_instruct)
pytorch_llama3_1_8b_instruct = deepcopy(turbomind_llama3_1_8b_instruct)

# ===== Configs for Qwen/Qwen2-7B-Instruct =====
turbomind_qwen2_7b_instruct = deepcopy(*lmdeploy_qwen2_7b_instruct)
turbomind_qwen2_7b_instruct_4bits = deepcopy(*lmdeploy_qwen2_7b_instruct)
turbomind_qwen2_7b_instruct_kvint4 = deepcopy(*lmdeploy_qwen2_7b_instruct)
turbomind_qwen2_7b_instruct_kvint8 = deepcopy(*lmdeploy_qwen2_7b_instruct)
pytorch_qwen2_7b_instruct = deepcopy(*lmdeploy_qwen2_7b_instruct)

for model in [v for k, v in locals().items() if k.startswith('turbomind_')]:
    model['engine_config']['max_batch_size'] = 128
    model['gen_config']['do_sample'] = False
    model['batch_size'] = 128

for model in [v for k, v in locals().items() if k.endswith('_4bits')]:
    model['engine_config']['model_format'] = 'awq'
    model['abbr'] = model['abbr'] + '_4bits'
    model['path'] = model['path'] + '-inner-4bits'

for model in [v for k, v in locals().items() if k.endswith('_kvint4')]:
    model['engine_config']['quant_policy'] = 4
    model['abbr'] = model['abbr'] + '_kvint4'

for model in [v for k, v in locals().items() if k.endswith('_kvint8')]:
    model['engine_config']['quant_policy'] = 8
    model['abbr'] = model['abbr'] + '_kvint8'

for model in [v for k, v in locals().items() if k.startswith('pytorch_')]:
    model['abbr'] = model['abbr'].replace('turbomind', 'pytorch')
    model['backend'] = 'pytorch'
    model['engine_config']['max_batch_size'] = 64
    model['gen_config']['do_sample'] = False
    model['batch_size'] = 64

turbomind_internlm2_5_7b_chat_batch1[
    'abbr'] = turbomind_internlm2_5_7b_chat_batch1['abbr'] + '_batch1'
turbomind_internlm2_5_7b_chat_batch1['engine_config']['max_batch_size'] = 1
turbomind_internlm2_5_7b_chat_batch1['batch_size'] = 1

basic_pytorch_chat_tp1 = dict(type=TurboMindModelwithChatTemplate,
                              engine_config=dict(session_len=MAX_SESSION_LEN,
                                                 max_batch_size=64,
                                                 tp=1),
                              gen_config=dict(do_sample=False,
                                              max_new_tokens=MAX_NEW_TOKENS),
                              max_out_len=MAX_NEW_TOKENS,
                              max_seq_len=MAX_SESSION_LEN,
                              batch_size=64,
                              run_cfg=dict(num_gpus=1))

# ===== Configs for Qwen/Qwen1.5-MoE-A2.7B-Chat =====
pytorch_qwen1_5_moe_2_7b_chat = deepcopy(basic_pytorch_chat_tp1)
pytorch_qwen1_5_moe_2_7b_chat['abbr'] = 'pytorch_qwen1_5_moe_2_7b_chat'
pytorch_qwen1_5_moe_2_7b_chat['path'] = 'Qwen/Qwen1.5-MoE-A2.7B-Chat'

# ===== Configs for google/gemma2-7b-it =====
pytorch_gemma_2_9b_it = deepcopy(basic_pytorch_chat_tp1)
pytorch_gemma_2_9b_it['abbr'] = 'pytorch_gemma_2_9b_it'
pytorch_gemma_2_9b_it['path'] = 'google/gemma-2-9b-it'

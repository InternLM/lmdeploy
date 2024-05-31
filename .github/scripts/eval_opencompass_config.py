from mmengine.config import read_base
from opencompass.models import LmdeployPytorchModel, TurboMindModel

with read_base():
    # choose a list of datasets
    from .datasets.ceval.ceval_gen_5f30c7 import \
        ceval_datasets  # noqa: F401, E501
    from .datasets.crowspairs.crowspairs_gen_381af0 import \
        crowspairs_datasets  # noqa: F401, E501
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import \
        gsm8k_datasets  # noqa: F401, E501
    from .datasets.mmlu.mmlu_gen_a484b3 import \
        mmlu_datasets  # noqa: F401, E501
    from .datasets.race.race_gen_69ee4f import \
        race_datasets  # noqa: F401, E501
    from .datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import \
        WiC_datasets  # noqa: F401, E501
    from .datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_7902a7 import \
        WSC_datasets  # noqa: F401, E501
    from .datasets.triviaqa.triviaqa_gen_2121ce import \
        triviaqa_datasets  # noqa: F401, E501
    # read hf models
    from .models.baichuan.hf_baichuan2_7b_chat import \
        models as hf_baichuan2_chat_7b  # noqa: F401, E501
    from .models.gemma.hf_gemma_7b_it import \
        models as hf_gemma_chat_7b  # noqa: F401, E501
    from .models.hf_internlm.hf_internlm2_chat_7b import \
        models as hf_internlm2_chat_7b  # noqa: F401, E501
    from .models.hf_internlm.hf_internlm2_chat_20b import \
        models as hf_internlm2_chat_20b  # noqa: F401, E501
    from .models.hf_internlm.hf_internlm_chat_7b import \
        models as hf_internlm_chat_7b  # noqa: F401, E501
    from .models.hf_internlm.hf_internlm_chat_20b import \
        models as hf_internlm_chat_20b  # noqa: F401, E501
    from .models.hf_llama.hf_llama2_7b_chat import \
        models as hf_llama2_chat_7b  # noqa: F401, E501
    from .models.hf_llama.hf_llama3_8b_instruct import \
        models as hf_llama_3_8b_instruct  # noqa: F401, E501
    from .models.mistral.hf_mistral_7b_instruct_v0_1 import \
        models as hf_mistral_chat_7b  # noqa: F401, E501
    from .models.mistral.hf_mixtral_8x7b_instruct_v0_1 import \
        models as hf_mixtral_chat_8x7b  # noqa: F401, E501
    from .models.qwen.hf_qwen1_5_7b_chat import \
        models as hf_qwen1_5_chat_7b  # noqa: F401, E501
    from .models.qwen.hf_qwen_7b_chat import \
        models as hf_qwen_chat_7b  # noqa: F401, E501
    # and output the results in a chosen format
    from .summarizers.medium import summarizer  # noqa: F401, E501

internlm_meta_template = dict(round=[
    dict(role='HUMAN', begin='<|User|>:', end='\n'),
    dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
],
                              eos_token_id=103028)

internlm2_meta_template = dict(round=[
    dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
    dict(role='BOT',
         begin='<|im_start|>assistant\n',
         end='<|im_end|>\n',
         generate=True),
],
                               eos_token_id=92542)

llama2_meta_template = dict(round=[
    dict(role='HUMAN', begin='[INST] ', end=' [/INST]'),
    dict(role='BOT', begin='', end='', generate=True),
],
                            eos_token_id=2)

llama3_meta_template = dict(round=[
    dict(role='HUMAN',
         begin='<|start_header_id|>user<|end_header_id|>\n\n',
         end='<|eot_id|>'),
    dict(role='BOT',
         begin='<|start_header_id|>assistant<|end_header_id|>\n\n',
         end='<|eot_id|>',
         generate=True),
],
                            eos_token_id=[128001, 128009])

qwen_meta_template = dict(round=[
    dict(role='HUMAN', begin='\n<|im_start|>user\n', end='<|im_end|>'),
    dict(role='BOT',
         begin='\n<|im_start|>assistant\n',
         end='<|im_end|>',
         generate=True),
], )

qwen1_5_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT',
             begin='<|im_start|>assistant\n',
             end='<|im_end|>\n',
             generate=True),
    ],
    eos_token_id=151645,
)

baichuan2_meta_template = dict(round=[
    dict(role='HUMAN', begin='<reserved_106>'),
    dict(role='BOT', begin='<reserved_107>', generate=True),
], )

mistral_meta_template = dict(begin='<s>',
                             round=[
                                 dict(role='HUMAN',
                                      begin='[INST]',
                                      end='[/INST]'),
                                 dict(role='BOT',
                                      begin='',
                                      end='</s>',
                                      generate=True),
                             ],
                             eos_token_id=2)

gemma_meta_template = dict(round=[
    dict(role='HUMAN', begin='<start_of_turn>user\n', end='<end_of_turn>\n'),
    dict(role='BOT',
         begin='<start_of_turn>model',
         end='<end_of_turn>\n',
         generate=True),
],
                           eos_token_id=1)

MAX_SESSION_LEN = 2048
MAX_NEW_TOKENS = 1024

tb_engine_config_template_max_bs_8 = dict(session_len=MAX_SESSION_LEN,
                                          max_batch_size=8,
                                          rope_scaling_factor=1.0)
tb_engine_config_template_max_bs_16 = dict(session_len=MAX_SESSION_LEN,
                                           max_batch_size=16,
                                           rope_scaling_factor=1.0)
tb_engine_config_template_max_bs_32 = dict(session_len=MAX_SESSION_LEN,
                                           max_batch_size=32,
                                           rope_scaling_factor=1.0)
tb_engine_config_template_max_bs_128 = dict(session_len=MAX_SESSION_LEN,
                                            max_batch_size=128,
                                            rope_scaling_factor=1.0)
tb_engine_config_template_max_bs_128_tp2 = dict(session_len=MAX_SESSION_LEN,
                                                max_batch_size=128,
                                                tp=2,
                                                rope_scaling_factor=1.0)

pt_engine_config_template_max_bs_8 = dict(session_len=MAX_SESSION_LEN,
                                          max_batch_size=8)
pt_engine_config_template_max_bs_16 = dict(session_len=MAX_SESSION_LEN,
                                           max_batch_size=16)
pt_engine_config_template_max_bs_32 = dict(session_len=MAX_SESSION_LEN,
                                           max_batch_size=32)
pt_engine_config_template_max_bs_64 = dict(session_len=MAX_SESSION_LEN,
                                           max_batch_size=64)
pt_engine_config_template_max_bs_128 = dict(session_len=MAX_SESSION_LEN,
                                            max_batch_size=128)
pt_engine_config_template_max_bs_128_tp2 = dict(session_len=MAX_SESSION_LEN,
                                                tp=2,
                                                max_batch_size=128)
pt_engine_config_template_max_bs_64_tp2 = dict(session_len=MAX_SESSION_LEN,
                                               tp=2,
                                               max_batch_size=64)

pt_engine_config_template_max_bs_8_prefill = dict(session_len=MAX_SESSION_LEN,
                                                  cache_max_entry_count=0.5,
                                                  max_prefill_token_num=4096,
                                                  max_batch_size=8)
pt_engine_config_template_max_bs_16_prefill = dict(session_len=MAX_SESSION_LEN,
                                                   cache_max_entry_count=0.5,
                                                   max_prefill_token_num=4096,
                                                   max_batch_size=16)
pt_engine_config_template_max_bs_64_prefill = dict(session_len=MAX_SESSION_LEN,
                                                   cache_max_entry_count=0.5,
                                                   max_prefill_token_num=4096,
                                                   max_batch_size=64)
pt_engine_config_template_max_bs_128_prefill = dict(
    session_len=MAX_SESSION_LEN,
    cache_max_entry_count=0.5,
    max_prefill_token_num=4096,
    max_batch_size=128)
pt_engine_config_template_max_bs_8_prefill_tp2 = dict(
    session_len=MAX_SESSION_LEN,
    cache_max_entry_count=0.5,
    max_prefill_token_num=4096,
    max_batch_size=8,
    tp=2)
pt_engine_config_template_max_bs_64_prefill_tp2 = dict(
    session_len=MAX_SESSION_LEN,
    cache_max_entry_count=0.5,
    max_prefill_token_num=4096,
    max_batch_size=64,
    tp=2)
pt_engine_config_template_max_bs_128_prefill_tp2 = dict(
    session_len=MAX_SESSION_LEN,
    cache_max_entry_count=0.5,
    max_prefill_token_num=4096,
    max_batch_size=128,
    tp=2)
tb_awq_engine_config_template_max_bs_8 = dict(session_len=MAX_SESSION_LEN,
                                              max_batch_size=8,
                                              model_format='awq',
                                              rope_scaling_factor=1.0)
tb_awq_engine_config_template_max_bs_32 = dict(session_len=MAX_SESSION_LEN,
                                               max_batch_size=32,
                                               model_format='awq',
                                               rope_scaling_factor=1.0)
tb_awq_engine_config_template_max_bs_128 = dict(session_len=MAX_SESSION_LEN,
                                                max_batch_size=128,
                                                model_format='awq',
                                                rope_scaling_factor=1.0)

tb_awq_engine_config_template_max_bs_128_tp2 = dict(
    session_len=MAX_SESSION_LEN,
    max_batch_size=128,
    model_format='awq',
    tp=2,
    rope_scaling_factor=1.0)

tb_kvint4_engine_config_template_max_bs_128 = dict(session_len=MAX_SESSION_LEN,
                                                   max_batch_size=128,
                                                   quant_policy=4,
                                                   rope_scaling_factor=1.0)

tb_kvint4_engine_config_template_max_bs_128_tp2 = dict(
    session_len=MAX_SESSION_LEN,
    max_batch_size=128,
    quant_policy=4,
    tp=2,
    rope_scaling_factor=1.0)

gen_config_template = dict(top_k=1,
                           top_p=0.8,
                           temperature=1.0,
                           max_new_tokens=MAX_NEW_TOKENS)
qwen_gen_config_template = dict(top_k=1,
                                top_p=0.8,
                                temperature=1.0,
                                stop_words=[151645],
                                max_new_tokens=MAX_NEW_TOKENS)

tokenizer_kwargs_template = dict(padding_side='left',
                                 truncation_side='left',
                                 use_fast=False,
                                 trust_remote_code=True)
model_kwargs_template = dict(device_map='auto', trust_remote_code=True)

run_cfg_tp1_template = dict(num_gpus=1, num_procs=1)
run_cfg_tp2_template = dict(num_gpus=2, num_procs=1)

# ===== Configs for internlm/internlm-chat-7b =====
# config for internlm-chat-7b
tb_internlm_chat_7b = dict(type=TurboMindModel,
                           abbr='internlm-chat-7b-turbomind',
                           path='internlm/internlm-chat-7b',
                           engine_config=tb_engine_config_template_max_bs_32,
                           gen_config=gen_config_template,
                           max_out_len=MAX_NEW_TOKENS,
                           max_seq_len=MAX_SESSION_LEN,
                           batch_size=32,
                           concurrency=32,
                           meta_template=internlm_meta_template,
                           run_cfg=run_cfg_tp1_template,
                           end_str='<eoa>')

# config for pt internlm-chat-7b
pt_internlm_chat_7b = dict(type=LmdeployPytorchModel,
                           abbr='internlm-chat-7b-pytorch',
                           path='internlm/internlm-chat-7b',
                           engine_config=pt_engine_config_template_max_bs_16,
                           gen_config=gen_config_template,
                           max_out_len=MAX_NEW_TOKENS,
                           max_seq_len=MAX_SESSION_LEN,
                           batch_size=16,
                           concurrency=16,
                           meta_template=internlm_meta_template,
                           run_cfg=run_cfg_tp1_template,
                           end_str='<eoa>')

tb_internlm_chat_7b_w4a16 = dict(
    type=TurboMindModel,
    abbr='internlm-chat-7b-4bits-turbomind',
    path='internlm/internlm-chat-7b-inner-4bits',
    engine_config=tb_awq_engine_config_template_max_bs_32,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=32,
    concurrency=32,
    meta_template=internlm_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='<eoa>')

# ===== Configs for internlm/internlm-chat-20b =====
# config for internlm-chat-20b
tb_internlm_chat_20b = dict(type=TurboMindModel,
                            abbr='internlm-chat-20b-turbomind',
                            path='internlm/internlm-chat-20b',
                            engine_config=tb_engine_config_template_max_bs_8,
                            gen_config=gen_config_template,
                            max_out_len=MAX_NEW_TOKENS,
                            max_seq_len=MAX_SESSION_LEN,
                            batch_size=8,
                            concurrency=8,
                            meta_template=internlm_meta_template,
                            run_cfg=run_cfg_tp1_template,
                            end_str='<eoa>')

# config for internlm-chat-20b-w4 model
tb_internlm_chat_20b_w4a16 = dict(
    type=TurboMindModel,
    abbr='internlm-chat-20b-4bits-turbomind',
    path='internlm/internlm-chat-20b-inner-4bits',
    engine_config=tb_awq_engine_config_template_max_bs_8,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=8,
    concurrency=8,
    meta_template=internlm_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='<eoa>')

# config for internlm-chat-20b
pt_internlm_chat_20b = dict(
    type=LmdeployPytorchModel,
    abbr='internlm-chat-20b-pytorch',
    path='internlm/internlm-chat-20b',
    engine_config=pt_engine_config_template_max_bs_8_prefill,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=8,
    concurrency=8,
    meta_template=internlm_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='<eoa>')

# ===== Configs for internlm/internlm2-chat-7b =====
# config for internlm2-chat-7b
tb_internlm2_chat_7b = dict(type=TurboMindModel,
                            abbr='internlm2-chat-7b-turbomind',
                            path='internlm/internlm2-chat-7b',
                            engine_config=tb_engine_config_template_max_bs_128,
                            gen_config=gen_config_template,
                            max_out_len=MAX_NEW_TOKENS,
                            max_seq_len=MAX_SESSION_LEN,
                            batch_size=128,
                            concurrency=128,
                            meta_template=internlm2_meta_template,
                            run_cfg=run_cfg_tp1_template,
                            end_str='<|im_end|>')

# config for internlm2-chat-7b-w4
tb_internlm2_chat_7b_w4a16 = dict(
    type=TurboMindModel,
    abbr='internlm2-chat-7b-4bits-turbomind',
    path='internlm/internlm2-chat-7b-inner-4bits',
    engine_config=tb_awq_engine_config_template_max_bs_128,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=internlm2_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='<|im_end|>')

tb_internlm2_chat_7b_kvint4 = dict(
    type=TurboMindModel,
    abbr='internlm2-chat-7b-turbomind-kvint4',
    path='internlm/internlm2-chat-7b',
    engine_config=tb_kvint4_engine_config_template_max_bs_128,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=internlm2_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='<|im_end|>')

# config for pt internlm-chat-7b
pt_internlm2_chat_7b = dict(type=LmdeployPytorchModel,
                            abbr='internlm2-chat-7b-pytorch',
                            path='internlm/internlm2-chat-7b',
                            engine_config=pt_engine_config_template_max_bs_128,
                            gen_config=gen_config_template,
                            max_out_len=MAX_NEW_TOKENS,
                            max_seq_len=MAX_SESSION_LEN,
                            batch_size=128,
                            concurrency=128,
                            meta_template=internlm2_meta_template,
                            run_cfg=run_cfg_tp1_template,
                            end_str='<|im_end|>')

# ===== Configs for internlm/internlm2-chat-20b =====
# config for internlm2-chat-20b
tb_internlm2_chat_20b = dict(
    type=TurboMindModel,
    abbr='internlm2-chat-20b-turbomind',
    path='internlm/internlm2-chat-20b',
    engine_config=tb_engine_config_template_max_bs_128_tp2,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=internlm2_meta_template,
    run_cfg=run_cfg_tp2_template,
    end_str='<|im_end|>')

# config for internlm2-chat-20b-w4 model
tb_internlm2_chat_20b_w4a16 = dict(
    type=TurboMindModel,
    abbr='internlm2-chat-20b-4bits-turbomind',
    path='internlm/internlm2-chat-20b-inner-4bits',
    engine_config=tb_awq_engine_config_template_max_bs_128_tp2,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=internlm2_meta_template,
    run_cfg=run_cfg_tp2_template,
    end_str='<|im_end|>')

# config for internlm2-chat-20b-w4 model
tb_internlm2_chat_20b_kvint4 = dict(
    type=TurboMindModel,
    abbr='internlm2-chat-20b-turbomind-kvint4',
    path='internlm/internlm2-chat-20b-inner-4bits',
    engine_config=tb_kvint4_engine_config_template_max_bs_128_tp2,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=internlm2_meta_template,
    run_cfg=run_cfg_tp2_template,
    end_str='<|im_end|>')

# config for pt internlm-chat-20b
pt_internlm2_chat_20b = dict(
    type=LmdeployPytorchModel,
    abbr='internlm2-chat-20b-pytorch',
    path='internlm/internlm2-chat-20b',
    engine_config=pt_engine_config_template_max_bs_64_prefill,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=64,
    concurrency=64,
    meta_template=internlm2_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='<|im_end|>')

# ===== Configs for Qwen/Qwen-7B-Chat =====
# config for qwen-chat-7b turbomind
tb_qwen_chat_7b = dict(type=TurboMindModel,
                       abbr='qwen-7b-chat-turbomind',
                       path='Qwen/Qwen-7B-Chat',
                       engine_config=tb_engine_config_template_max_bs_128,
                       gen_config=qwen_gen_config_template,
                       max_out_len=MAX_NEW_TOKENS,
                       max_seq_len=MAX_SESSION_LEN,
                       batch_size=128,
                       concurrency=128,
                       meta_template=qwen_meta_template,
                       run_cfg=run_cfg_tp1_template,
                       end_str='<|im_end|>')

tb_qwen_chat_7b_w4a16 = dict(
    type=TurboMindModel,
    abbr='qwen-7b-chat-4bits-turbomind',
    path='Qwen/Qwen-7B-Chat-inner-4bits',
    engine_config=tb_awq_engine_config_template_max_bs_128,
    gen_config=qwen_gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=qwen_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='<|im_end|>')

# config for qwen-chat-7b pytorch
pt_qwen_chat_7b = dict(type=LmdeployPytorchModel,
                       abbr='qwen-7b-chat-pytorch',
                       path='Qwen/Qwen-7B-Chat',
                       engine_config=pt_engine_config_template_max_bs_64,
                       gen_config=qwen_gen_config_template,
                       max_out_len=MAX_NEW_TOKENS,
                       max_seq_len=MAX_SESSION_LEN,
                       batch_size=64,
                       concurrency=64,
                       meta_template=qwen_meta_template,
                       run_cfg=run_cfg_tp1_template,
                       end_str='<|im_end|>')

# ===== Configs for meta-llama/Llama-2-7b-chat-hf =====
# config for llama2-chat-7b turbomind
tb_llama2_chat_7b = dict(type=TurboMindModel,
                         abbr='llama-2-7b-chat-turbomind',
                         path='meta-llama/Llama-2-7b-chat-hf',
                         engine_config=tb_engine_config_template_max_bs_128,
                         gen_config=gen_config_template,
                         max_out_len=MAX_NEW_TOKENS,
                         max_seq_len=MAX_SESSION_LEN,
                         batch_size=128,
                         concurrency=128,
                         meta_template=llama2_meta_template,
                         run_cfg=run_cfg_tp1_template,
                         end_str='[INST]')

# config for llama2-chat-7b-w4a16 turbomind
tb_llama2_chat_7b_wa416 = dict(
    type=TurboMindModel,
    abbr='llama-2-7b-chat-4bits-turbomind',
    path='meta-llama/Llama-2-7b-chat-hf-inner-4bits',
    engine_config=tb_awq_engine_config_template_max_bs_128,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=llama2_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='[INST]')

# config for llama2-chat-7b pytorch
pt_llama2_chat_7b = dict(type=LmdeployPytorchModel,
                         abbr='llama-2-7b-chat-pytorch',
                         path='meta-llama/Llama-2-7b-chat-hf',
                         engine_config=pt_engine_config_template_max_bs_128,
                         gen_config=gen_config_template,
                         max_out_len=MAX_NEW_TOKENS,
                         max_seq_len=MAX_SESSION_LEN,
                         batch_size=128,
                         concurrency=128,
                         meta_template=llama2_meta_template,
                         run_cfg=run_cfg_tp1_template,
                         end_str='[INST]')

# ===== Configs for baichuan-inc/Baichuan2-7B-Chat =====
# config for baichuan2-chat-7b turbomind
tb_baichuan2_chat_7b = dict(type=TurboMindModel,
                            abbr='Baichuan2-7B-Chat-turbomind',
                            path='baichuan-inc/Baichuan2-7B-Chat',
                            engine_config=tb_engine_config_template_max_bs_16,
                            gen_config=gen_config_template,
                            max_out_len=MAX_NEW_TOKENS,
                            max_seq_len=MAX_SESSION_LEN,
                            batch_size=16,
                            concurrency=16,
                            meta_template=baichuan2_meta_template,
                            run_cfg=run_cfg_tp1_template)

# config for baichuan2-chat-7b pytorch
pt_baichuan2_chat_7b = dict(type=LmdeployPytorchModel,
                            abbr='baichuan2-7b-chat-hf',
                            path='baichuan-inc/Baichuan2-7B-Chat',
                            engine_config=pt_engine_config_template_max_bs_16,
                            gen_config=gen_config_template,
                            max_out_len=MAX_NEW_TOKENS,
                            max_seq_len=MAX_SESSION_LEN,
                            batch_size=16,
                            concurrency=16,
                            meta_template=baichuan2_meta_template,
                            run_cfg=run_cfg_tp1_template,
                            end_str=None)

# ===== Configs for mistralai/Mistral-7B-Instruct-v0.1 =====
# config for pt Mistral-7B-Instruct-v0.1
pt_mistral_chat_7b = dict(type=LmdeployPytorchModel,
                          abbr='mistral-7b-instruct-v0.1-pytorch',
                          path='mistralai/Mistral-7B-Instruct-v0.1',
                          engine_config=pt_engine_config_template_max_bs_16,
                          gen_config=gen_config_template,
                          max_out_len=MAX_NEW_TOKENS,
                          max_seq_len=MAX_SESSION_LEN,
                          batch_size=16,
                          concurrency=16,
                          meta_template=mistral_meta_template,
                          run_cfg=run_cfg_tp1_template,
                          end_str='</s>')

# ===== Configs for mistralai/Mixtral-8x7B-Instruct-v0.1 =====
# config for pt Mixtral-8x7B-Instruct-v0.1
pt_mixtral_chat_8x7b = dict(
    type=LmdeployPytorchModel,
    abbr='mixtral-8x7b-instruct-v0.1-pytorch',
    path='mistralai/Mixtral-8x7B-Instruct-v0.1',
    engine_config=pt_engine_config_template_max_bs_8_prefill_tp2,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=8,
    concurrency=8,
    meta_template=mistral_meta_template,
    run_cfg=run_cfg_tp2_template,
    end_str='</s>')

# ===== Configs for Qwen/Qwen1.5-7B-Chat =====
tb_qwen1_5_chat_7b = dict(type=TurboMindModel,
                          abbr='qwen1.5-7b-chat-turbomind',
                          path='Qwen/Qwen1.5-7B-Chat',
                          engine_config=tb_engine_config_template_max_bs_128,
                          gen_config=gen_config_template,
                          max_out_len=MAX_NEW_TOKENS,
                          max_seq_len=MAX_SESSION_LEN,
                          batch_size=128,
                          concurrency=128,
                          meta_template=qwen1_5_meta_template,
                          run_cfg=run_cfg_tp1_template,
                          end_str='<|im_end|>')

tb_qwen1_5_chat_7b_w4a16 = dict(
    type=TurboMindModel,
    abbr='qwen1.5-7b-chat-4bits-turbomind',
    path='Qwen/Qwen1.5-7B-Chat-inner-4bits',
    engine_config=tb_awq_engine_config_template_max_bs_128,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=qwen1_5_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='<|im_end|>')

tb_qwen1_5_chat_7b_kvint4 = dict(
    type=TurboMindModel,
    abbr='qwen1.5-7b-chat-turbomind-kvint4',
    path='Qwen/Qwen1.5-7B-Chat',
    engine_config=tb_kvint4_engine_config_template_max_bs_128,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=qwen1_5_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='<|im_end|>')

pt_qwen1_5_chat_7b = dict(type=LmdeployPytorchModel,
                          abbr='qwen1.5-7b-chat-pytorch',
                          path='Qwen/Qwen1.5-7B-Chat',
                          engine_config=pt_engine_config_template_max_bs_128,
                          gen_config=gen_config_template,
                          max_out_len=MAX_NEW_TOKENS,
                          max_seq_len=MAX_SESSION_LEN,
                          batch_size=128,
                          concurrency=128,
                          meta_template=qwen1_5_meta_template,
                          run_cfg=run_cfg_tp1_template,
                          end_str='<|im_end|>')

pt_qwen1_5_moe_2_7b_chat = dict(
    type=LmdeployPytorchModel,
    abbr='qwen1.5-moe-2.7b-chat-pytorch',
    path='Qwen/Qwen1.5-MoE-A2.7B-Chat',
    engine_config=pt_engine_config_template_max_bs_64,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=64,
    concurrency=64,
    meta_template=qwen1_5_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='<|im_end|>')

# ===== Configs for google/gemma-7b-it =====
pt_gemma_chat_7b = dict(type=LmdeployPytorchModel,
                        abbr='gemma-7b-it-pytorch',
                        path='google/gemma-7b-it',
                        engine_config=pt_engine_config_template_max_bs_16,
                        gen_config=gen_config_template,
                        max_out_len=MAX_NEW_TOKENS,
                        max_seq_len=MAX_SESSION_LEN,
                        batch_size=16,
                        concurrency=16,
                        meta_template=gemma_meta_template,
                        run_cfg=run_cfg_tp1_template,
                        end_str='<end_of_turn>')

# ===== Configs for meta-llama/Meta-Llama-3-8B-Instruct =====
# config for llama-3-8b-instruct turbomind
tb_llama_3_8b_instruct = dict(
    type=TurboMindModel,
    abbr='llama-3-8b-instruct-turbomind',
    path='meta-llama/Meta-Llama-3-8B-Instruct',
    engine_config=tb_engine_config_template_max_bs_128,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=llama3_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='[INST]')

tb_llama_3_8b_instruct_w4a16 = dict(
    type=TurboMindModel,
    abbr='llama-3-8b-instruct-4bits-turbomind',
    path='meta-llama/Meta-Llama-3-8B-Instruct-inner-4bits',
    engine_config=tb_awq_engine_config_template_max_bs_128,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=llama3_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='[INST]')

tb_llama_3_8b_instruct_kvint4 = dict(
    type=TurboMindModel,
    abbr='llama-3-8b-instruct-turbomind-kvint4',
    path='meta-llama/Meta-Llama-3-8B-Instruct',
    engine_config=tb_kvint4_engine_config_template_max_bs_128,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=llama3_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='[INST]')

# config for llama-3-8b-instruct pytorch
pt_llama_3_8b_instruct = dict(
    type=LmdeployPytorchModel,
    abbr='llama-3-8b-instruct-pytorch',
    path='meta-llama/Meta-Llama-3-8B-Instruct',
    engine_config=pt_engine_config_template_max_bs_128,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    meta_template=llama3_meta_template,
    run_cfg=run_cfg_tp1_template,
    end_str='[INST]')

# config for code llama
tb_codellama_7b_chat = dict(type=TurboMindModel,
                            abbr='codellama-7b-chat-turbomind',
                            path='codellama/CodeLlama-7b-Instruct-hf',
                            engine_config=tb_engine_config_template_max_bs_128,
                            gen_config=gen_config_template,
                            max_out_len=MAX_NEW_TOKENS,
                            max_seq_len=MAX_SESSION_LEN,
                            batch_size=128,
                            concurrency=128,
                            run_cfg=dict(num_gpus=1, num_procs=1),
                            end_str='</s>')

tb_codellama_7b_chat_w4a16 = dict(
    type=TurboMindModel,
    abbr='codellama-7b-chat-4bits-turbomind',
    path='codellama/CodeLlama-7b-Instruct-hf-inner-4bits',
    engine_config=tb_awq_engine_config_template_max_bs_128,
    gen_config=gen_config_template,
    max_out_len=MAX_NEW_TOKENS,
    max_seq_len=MAX_SESSION_LEN,
    batch_size=128,
    concurrency=128,
    run_cfg=dict(num_gpus=1, num_procs=1),
    end_str='</s>')

pt_codellama_7b_chat = dict(type=LmdeployPytorchModel,
                            abbr='codellama-7b-chat-pytorch',
                            path='codellama/CodeLlama-7b-Instruct-hf',
                            engine_config=pt_engine_config_template_max_bs_128,
                            gen_config=gen_config_template,
                            max_out_len=MAX_NEW_TOKENS,
                            max_seq_len=MAX_SESSION_LEN,
                            batch_size=128,
                            concurrency=128,
                            run_cfg=dict(num_gpus=1, num_procs=1),
                            end_str='</s>')

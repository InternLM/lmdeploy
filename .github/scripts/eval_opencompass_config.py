from mmengine.config import read_base
from opencompass.models import (HuggingFaceCausalLM, LmdeployPytorchModel,
                                TurboMindModel)

with read_base():
    # choose a list of datasets
    # from .datasets.ceval.ceval_gen_5f30c7 import \
    #     ceval_datasets  # noqa: F401, E501
    # from .datasets.crowspairs.crowspairs_gen_381af0 import \
    #     crowspairs_datasets  # noqa: F401, E501
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import \
        gsm8k_datasets  # noqa: F401, E501
    from .datasets.mmlu.mmlu_gen_a484b3 import \
        mmlu_datasets  # noqa: F401, E501
    # from .datasets.race.race_gen_69ee4f import \
    #     race_datasets  # noqa: F401, E501
    # from .datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import \
    #     WiC_datasets  # noqa: F401, E501
    # from .datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_7902a7 import \
    #     WSC_datasets  # noqa: F401, E501
    # from .datasets.triviaqa.triviaqa_gen_2121ce import \
    #     triviaqa_datasets  # noqa: F401, E501
    # and output the results in a chosen format
    from .summarizers.medium import summarizer  # noqa: F401, E501

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

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

qwen_meta_template = dict(round=[
    dict(role='HUMAN', begin='\n<|im_start|>user\n', end='<|im_end|>'),
    dict(role='BOT',
         begin='\n<|im_start|>assistant\n',
         end='<|im_end|>',
         generate=True),
], )

qwen1dot5_meta_template = dict(
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

# ===== Configs for internlm/internlm-chat-7b =====
# config for internlm-chat-7b
hf_internlm_chat_7b = dict(type=HuggingFaceCausalLM,
                           abbr='internlm-chat-7b-hf',
                           path='internlm/internlm-chat-7b',
                           tokenizer_path='internlm/internlm-chat-7b',
                           model_kwargs=dict(
                               trust_remote_code=True,
                               device_map='auto',
                           ),
                           tokenizer_kwargs=dict(
                               padding_side='left',
                               truncation_side='left',
                               use_fast=False,
                               trust_remote_code=True,
                           ),
                           max_out_len=256,
                           max_seq_len=2048,
                           batch_size=16,
                           batch_padding=False,
                           meta_template=internlm_meta_template,
                           run_cfg=dict(num_gpus=1, num_procs=1),
                           end_str='<eoa>')

# config for internlm-chat-7b
tb_internlm_chat_7b = dict(type=TurboMindModel,
                           abbr='internlm-chat-7b-turbomind',
                           path='internlm/internlm-chat-7b',
                           engine_config=dict(session_len=2048,
                                              max_batch_size=32,
                                              rope_scaling_factor=1.0),
                           gen_config=dict(top_k=1,
                                           top_p=0.8,
                                           temperature=1.0,
                                           max_new_tokens=256),
                           max_out_len=256,
                           max_seq_len=2048,
                           batch_size=32,
                           concurrency=32,
                           meta_template=internlm_meta_template,
                           run_cfg=dict(num_gpus=1, num_procs=1),
                           end_str='<eoa>')

# config for pt internlm-chat-7b
pt_internlm_chat_7b = dict(type=LmdeployPytorchModel,
                           abbr='internlm-chat-7b-pytorch',
                           path='internlm/internlm-chat-7b',
                           engine_config=dict(session_len=2048,
                                              max_batch_size=16),
                           gen_config=dict(top_k=1,
                                           top_p=0.8,
                                           temperature=1.0,
                                           max_new_tokens=256),
                           max_out_len=256,
                           max_seq_len=2048,
                           batch_size=16,
                           concurrency=16,
                           meta_template=internlm_meta_template,
                           run_cfg=dict(num_gpus=1, num_procs=1),
                           end_str='<eoa>')

tb_internlm_chat_7b_w4a16 = dict(type=TurboMindModel,
                                 abbr='internlm-chat-7b-4bits-turbomind',
                                 path='internlm/internlm-chat-7b-4bits',
                                 engine_config=dict(session_len=2048,
                                                    max_batch_size=32,
                                                    model_format='awq',
                                                    rope_scaling_factor=1.0),
                                 gen_config=dict(top_k=1,
                                                 top_p=0.8,
                                                 temperature=1.0,
                                                 max_new_tokens=256),
                                 max_out_len=256,
                                 max_seq_len=2048,
                                 batch_size=32,
                                 concurrency=32,
                                 meta_template=internlm_meta_template,
                                 run_cfg=dict(num_gpus=1, num_procs=1),
                                 end_str='<eoa>')

# ===== Configs for internlm/internlm-chat-20b =====
# config for internlm-chat-20b
tb_internlm_chat_20b = dict(type=TurboMindModel,
                            abbr='internlm-chat-20b-turbomind',
                            path='internlm/internlm-chat-20b',
                            engine_config=dict(session_len=2048,
                                               max_batch_size=8,
                                               rope_scaling_factor=1.0),
                            gen_config=dict(top_k=1,
                                            top_p=0.8,
                                            temperature=1.0,
                                            max_new_tokens=256),
                            max_out_len=256,
                            max_seq_len=2048,
                            batch_size=8,
                            concurrency=8,
                            meta_template=internlm_meta_template,
                            run_cfg=dict(num_gpus=1, num_procs=1),
                            end_str='<eoa>')

# config for internlm-chat-20b
hf_internlm_chat_20b = dict(type=HuggingFaceCausalLM,
                            abbr='internlm-chat-20b-hf',
                            path='internlm/internlm-chat-20b',
                            tokenizer_path='internlm/internlm-chat-20b',
                            tokenizer_kwargs=dict(
                                padding_side='left',
                                truncation_side='left',
                                use_fast=False,
                                trust_remote_code=True,
                            ),
                            max_out_len=256,
                            max_seq_len=2048,
                            batch_size=8,
                            batch_padding=False,
                            model_kwargs=dict(trust_remote_code=True,
                                              device_map='auto'),
                            run_cfg=dict(num_gpus=2, num_procs=1),
                            end_str='<eoa>')

# config for internlm-chat-20b-w4 model
tb_internlm_chat_20b_w4a16 = dict(type=TurboMindModel,
                                  abbr='internlm-chat-20b-4bits-turbomind',
                                  path='internlm/internlm-chat-20b-4bits',
                                  engine_config=dict(session_len=2048,
                                                     max_batch_size=8,
                                                     model_format='awq',
                                                     rope_scaling_factor=1.0),
                                  gen_config=dict(top_k=1,
                                                  top_p=0.8,
                                                  temperature=1.0,
                                                  max_new_tokens=256),
                                  max_out_len=256,
                                  max_seq_len=2048,
                                  batch_size=8,
                                  concurrency=8,
                                  meta_template=internlm_meta_template,
                                  run_cfg=dict(num_gpus=1, num_procs=1),
                                  end_str='<eoa>')

# config for internlm-chat-20b
pt_internlm_chat_20b = dict(type=LmdeployPytorchModel,
                            abbr='internlm-chat-20b-pytorch',
                            path='internlm/internlm-chat-20b',
                            engine_config=dict(session_len=2048,
                                               max_batch_size=8),
                            gen_config=dict(top_k=1,
                                            top_p=0.8,
                                            temperature=1.0,
                                            max_new_tokens=256),
                            max_out_len=256,
                            max_seq_len=2048,
                            batch_size=8,
                            concurrency=8,
                            meta_template=internlm_meta_template,
                            run_cfg=dict(num_gpus=1, num_procs=1),
                            end_str='<eoa>')

# ===== Configs for internlm/internlm2-chat-7b =====
# config for internlm2-chat-7b
tb_internlm2_chat_7b = dict(type=TurboMindModel,
                            abbr='internlm2-chat-7b-turbomind',
                            path='internlm/internlm2-chat-7b',
                            engine_config=dict(session_len=2048,
                                               max_batch_size=32,
                                               rope_scaling_factor=1.0),
                            gen_config=dict(top_k=1,
                                            top_p=0.8,
                                            temperature=1.0,
                                            max_new_tokens=256),
                            max_out_len=256,
                            max_seq_len=2048,
                            batch_size=32,
                            concurrency=32,
                            meta_template=internlm2_meta_template,
                            run_cfg=dict(num_gpus=1, num_procs=1),
                            end_str='<|im_end|>')

# config for internlm2-chat-7b
hf_internlm2_chat_7b = dict(type=HuggingFaceCausalLM,
                            abbr='internlm2-chat-7b-hf',
                            path='internlm/internlm2-chat-7b',
                            tokenizer_path='internlm/internlm2-chat-7b',
                            model_kwargs=dict(
                                trust_remote_code=True,
                                device_map='auto',
                            ),
                            tokenizer_kwargs=dict(
                                padding_side='left',
                                truncation_side='left',
                                use_fast=False,
                                trust_remote_code=True,
                            ),
                            max_out_len=256,
                            max_seq_len=2048,
                            batch_size=16,
                            batch_padding=False,
                            meta_template=internlm2_meta_template,
                            run_cfg=dict(num_gpus=1, num_procs=1),
                            end_str='<|im_end|>')

# config for internlm2-chat-7b-w4
tb_internlm2_chat_7b_w4a16 = dict(type=TurboMindModel,
                                  abbr='internlm2-chat-7b-4bits-turbomind',
                                  path='internlm/internlm2-chat-7b-4bits',
                                  engine_config=dict(session_len=2048,
                                                     max_batch_size=32,
                                                     model_format='awq',
                                                     rope_scaling_factor=1.0),
                                  gen_config=dict(top_k=1,
                                                  top_p=0.8,
                                                  temperature=1.0,
                                                  max_new_tokens=256),
                                  max_out_len=256,
                                  max_seq_len=2048,
                                  batch_size=32,
                                  concurrency=32,
                                  meta_template=internlm2_meta_template,
                                  run_cfg=dict(num_gpus=1, num_procs=1),
                                  end_str='<|im_end|>')

# config for pt internlm-chat-7b
pt_internlm2_chat_7b = dict(type=LmdeployPytorchModel,
                            abbr='internlm2-chat-7b-pytorch',
                            path='internlm/internlm2-chat-7b',
                            engine_config=dict(session_len=2048,
                                               max_batch_size=16),
                            gen_config=dict(top_k=1,
                                            top_p=0.8,
                                            temperature=1.0,
                                            max_new_tokens=256),
                            max_out_len=256,
                            max_seq_len=2048,
                            batch_size=16,
                            concurrency=16,
                            meta_template=internlm2_meta_template,
                            run_cfg=dict(num_gpus=1, num_procs=1),
                            end_str='<|im_end|>')

# ===== Configs for internlm/internlm2-chat-20b =====
# config for internlm2-chat-20b
tb_internlm2_chat_20b = dict(type=TurboMindModel,
                             abbr='internlm2-chat-20b-turbomind',
                             path='internlm/internlm2-chat-20b',
                             engine_config=dict(session_len=2048,
                                                max_batch_size=8,
                                                rope_scaling_factor=1.0),
                             gen_config=dict(top_k=1,
                                             top_p=0.8,
                                             temperature=1.0,
                                             max_new_tokens=256),
                             max_out_len=256,
                             max_seq_len=2048,
                             batch_size=8,
                             concurrency=8,
                             meta_template=internlm2_meta_template,
                             run_cfg=dict(num_gpus=1, num_procs=1),
                             end_str='<|im_end|>')

# config for internlm2-chat-20b
hf_internlm2_chat_20b = dict(type=HuggingFaceCausalLM,
                             abbr='internlm2-chat-20b-hf',
                             path='internlm/internlm2-chat-20b',
                             tokenizer_path='internlm/internlm2-chat-20b',
                             model_kwargs=dict(
                                 trust_remote_code=True,
                                 device_map='auto',
                             ),
                             tokenizer_kwargs=dict(
                                 padding_side='left',
                                 truncation_side='left',
                                 use_fast=False,
                                 trust_remote_code=True,
                             ),
                             max_out_len=256,
                             max_seq_len=2048,
                             batch_size=8,
                             batch_padding=False,
                             meta_template=internlm2_meta_template,
                             run_cfg=dict(num_gpus=2, num_procs=1),
                             end_str='<|im_end|>')

# config for internlm2-chat-20b-w4 model
tb_internlm2_chat_20b_w4a16 = dict(type=TurboMindModel,
                                   abbr='internlm2-chat-20b-4bits-turbomind',
                                   path='internlm/internlm2-chat-20b-4bits',
                                   engine_config=dict(session_len=2048,
                                                      max_batch_size=8,
                                                      model_format='awq',
                                                      rope_scaling_factor=1.0),
                                   gen_config=dict(top_k=1,
                                                   top_p=0.8,
                                                   temperature=1.0,
                                                   max_new_tokens=256),
                                   max_out_len=256,
                                   max_seq_len=2048,
                                   batch_size=8,
                                   concurrency=8,
                                   meta_template=internlm2_meta_template,
                                   run_cfg=dict(num_gpus=1, num_procs=1),
                                   end_str='<|im_end|>')

# config for pt internlm-chat-20b
pt_internlm2_chat_20b = dict(type=LmdeployPytorchModel,
                             abbr='internlm2-chat-20b-pytorch',
                             path='internlm/internlm2-chat-20b',
                             engine_config=dict(session_len=2048,
                                                max_batch_size=8),
                             gen_config=dict(top_k=1,
                                             top_p=0.8,
                                             temperature=1.0,
                                             max_new_tokens=256),
                             max_out_len=256,
                             max_seq_len=2048,
                             batch_size=8,
                             concurrency=8,
                             meta_template=internlm2_meta_template,
                             run_cfg=dict(num_gpus=1, num_procs=1),
                             end_str='<|im_end|>')

# ===== Configs for Qwen/Qwen-7B-Chat =====
# config for qwen-chat-7b turbomind
tb_qwen_chat_7b = dict(type=TurboMindModel,
                       abbr='qwen-7b-chat-turbomind',
                       path='Qwen/Qwen-7B-Chat',
                       engine_config=dict(session_len=2048,
                                          max_batch_size=16,
                                          rope_scaling_factor=1.0),
                       gen_config=dict(top_k=1,
                                       top_p=0.8,
                                       temperature=1.0,
                                       max_new_tokens=256),
                       max_out_len=256,
                       max_seq_len=2048,
                       batch_size=16,
                       concurrency=16,
                       meta_template=qwen_meta_template,
                       run_cfg=dict(num_gpus=1, num_procs=1),
                       end_str='<|im_end|>')

# config for qwen-chat-7b pytorch
pt_qwen_chat_7b = dict(type=LmdeployPytorchModel,
                       abbr='qwen-7b-chat-pytorch',
                       path='Qwen/Qwen-7B-Chat',
                       engine_config=dict(session_len=2048, max_batch_size=16),
                       gen_config=dict(top_k=1,
                                       top_p=0.8,
                                       temperature=1.0,
                                       max_new_tokens=256),
                       max_out_len=256,
                       max_seq_len=2048,
                       batch_size=16,
                       concurrency=16,
                       meta_template=qwen_meta_template,
                       run_cfg=dict(num_gpus=1, num_procs=1),
                       end_str='<|im_end|>')

# config for qwen-chat-7b huggingface
hf_qwen_chat_7b = dict(
    type=HuggingFaceCausalLM,
    abbr='qwen-7b-chat-hf',
    path='Qwen/Qwen-7B-Chat',
    tokenizer_path='Qwen/Qwen-7B-Chat',
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    tokenizer_kwargs=dict(padding_side='left',
                          truncation_side='left',
                          trust_remote_code=True,
                          use_fast=False),
    pad_token_id=151643,
    max_out_len=100,
    max_seq_len=2048,
    batch_size=16,
    batch_padding=False,
    meta_template=qwen_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
    end_str='<|im_end|>',
)

# ===== Configs for meta-llama/Llama-2-7b-chat-hf =====
# config for llama2-chat-7b turbomind
tb_llama2_chat_7b = dict(type=TurboMindModel,
                         abbr='llama-2-7b-chat-turbomind',
                         path='meta-llama/Llama-2-7b-chat-hf',
                         engine_config=dict(session_len=2048,
                                            max_batch_size=16,
                                            rope_scaling_factor=1.0),
                         gen_config=dict(top_k=1,
                                         top_p=0.8,
                                         temperature=1.0,
                                         max_new_tokens=256),
                         max_out_len=256,
                         max_seq_len=2048,
                         batch_size=16,
                         concurrency=16,
                         meta_template=llama2_meta_template,
                         run_cfg=dict(num_gpus=1, num_procs=1),
                         end_str='[INST]')

# config for llama2-chat-7b pytorch
pt_llama2_chat_7b = dict(type=LmdeployPytorchModel,
                         abbr='llama-2-7b-chat-pytorch',
                         path='meta-llama/Llama-2-7b-chat-hf',
                         engine_config=dict(session_len=2048,
                                            max_batch_size=16),
                         gen_config=dict(top_k=1,
                                         top_p=0.8,
                                         temperature=1.0,
                                         max_new_tokens=256),
                         max_out_len=256,
                         max_seq_len=2048,
                         batch_size=16,
                         concurrency=16,
                         meta_template=llama2_meta_template,
                         run_cfg=dict(num_gpus=1, num_procs=1),
                         end_str='[INST]')

# config for llama2-chat-7b huggingface
hf_llama2_chat_7b = dict(type=HuggingFaceCausalLM,
                         abbr='llama-2-7b-chat-hf',
                         path='meta-llama/Llama-2-7b-chat-hf',
                         tokenizer_path='meta-llama/Llama-2-7b-chat-hf',
                         model_kwargs=dict(device_map='auto'),
                         tokenizer_kwargs=dict(
                             padding_side='left',
                             truncation_side='left',
                             use_fast=False,
                         ),
                         meta_template=llama2_meta_template,
                         max_out_len=256,
                         max_seq_len=2048,
                         batch_size=16,
                         batch_padding=False,
                         run_cfg=dict(num_gpus=1, num_procs=1),
                         end_str='[INST]')

# ===== Configs for baichuan-inc/Baichuan2-7B-Chat =====
# config for baichuan2-chat-7b turbomind
tb_baichuan2_chat_7b = dict(type=TurboMindModel,
                            abbr='Baichuan2-7B-Chat-turbomind',
                            path='baichuan-inc/Baichuan2-7B-Chat',
                            engine_config=dict(session_len=2048,
                                               max_batch_size=16,
                                               rope_scaling_factor=1.0),
                            gen_config=dict(top_k=1,
                                            top_p=0.8,
                                            temperature=1.0,
                                            max_new_tokens=256),
                            max_out_len=256,
                            max_seq_len=2048,
                            batch_size=16,
                            concurrency=16,
                            meta_template=baichuan2_meta_template,
                            run_cfg=dict(num_gpus=1, num_procs=1))

# config for baichuan2-chat-7b huggingface
hf_baichuan2_chat_7b = dict(type=HuggingFaceCausalLM,
                            abbr='baichuan2-7b-chat-hf',
                            path='baichuan-inc/Baichuan2-7B-Chat',
                            tokenizer_path='baichuan-inc/Baichuan2-7B-Chat',
                            tokenizer_kwargs=dict(padding_side='left',
                                                  truncation_side='left',
                                                  trust_remote_code=True,
                                                  use_fast=False),
                            meta_template=baichuan2_meta_template,
                            max_out_len=100,
                            max_seq_len=2048,
                            batch_size=16,
                            batch_padding=False,
                            model_kwargs=dict(device_map='auto',
                                              trust_remote_code=True),
                            run_cfg=dict(num_gpus=1, num_procs=1))

# config for baichuan2-chat-7b pytorch
pt_baichuan2_chat_7b = dict(type=LmdeployPytorchModel,
                            abbr='baichuan2-7b-chat-hf',
                            path='baichuan-inc/Baichuan2-7B-Chat',
                            engine_config=dict(session_len=2048,
                                               max_batch_size=16),
                            gen_config=dict(top_k=1,
                                            top_p=0.8,
                                            temperature=1.0,
                                            max_new_tokens=256),
                            max_out_len=256,
                            max_seq_len=2048,
                            batch_size=16,
                            concurrency=16,
                            meta_template=baichuan2_meta_template,
                            run_cfg=dict(num_gpus=1, num_procs=1),
                            end_str=None)

# ===== Configs for mistralai/Mistral-7B-Instruct-v0.1 =====
# config for pt Mistral-7B-Instruct-v0.1
pt_mistral_chat_7b = dict(type=LmdeployPytorchModel,
                          abbr='mistral-7b-instruct-v0.1-pytorch',
                          path='mistralai/Mistral-7B-Instruct-v0.1',
                          engine_config=dict(session_len=2048,
                                             max_batch_size=16),
                          gen_config=dict(top_k=1,
                                          top_p=0.8,
                                          temperature=1.0,
                                          max_new_tokens=256),
                          max_out_len=256,
                          max_seq_len=2048,
                          batch_size=16,
                          concurrency=16,
                          meta_template=mistral_meta_template,
                          run_cfg=dict(num_gpus=1, num_procs=1),
                          end_str='</s>')

# config for hf Mistral-7B-Instruct-v0.1
hf_mistral_chat_7b = dict(abbr='mistral-7b-instruct-v0.1-hf',
                          type=HuggingFaceCausalLM,
                          path='mistralai/Mistral-7B-Instruct-v0.1',
                          tokenizer_path='mistralai/Mistral-7B-Instruct-v0.1',
                          model_kwargs=dict(device_map='auto',
                                            trust_remote_code=True),
                          tokenizer_kwargs=dict(
                              padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                          ),
                          meta_template=mistral_meta_template,
                          max_out_len=256,
                          max_seq_len=2048,
                          batch_size=16,
                          batch_padding=False,
                          run_cfg=dict(num_gpus=1, num_procs=1),
                          end_str='</s>')

# ===== Configs for mistralai/Mixtral-8x7B-Instruct-v0.1 =====
# config for hf Mixtral-8x7B-Instruct-v0.1
hf_mixtral_chat_8x7b = dict(
    abbr='mixtral-8x7b-instruct-v0.1-hf',
    type=HuggingFaceCausalLM,
    path='mistralai/Mixtral-8x7B-Instruct-v0.1',
    tokenizer_path='mistralai/Mixtral-8x7B-Instruct-v0.1',
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    tokenizer_kwargs=dict(padding_side='left',
                          truncation_side='left',
                          trust_remote_code=True),
    meta_template=mistral_meta_template,
    max_out_len=256,
    max_seq_len=2048,
    batch_size=8,
    batch_padding=False,
    run_cfg=dict(num_gpus=2, num_procs=1),
    end_str='</s>')

# config for pt Mixtral-8x7B-Instruct-v0.1
pt_mixtral_chat_8x7b = dict(type=LmdeployPytorchModel,
                            abbr='mixtral-8x7b-instruct-v0.1-pytorch',
                            path='mistralai/Mixtral-8x7B-Instruct-v0.1',
                            engine_config=dict(session_len=2048,
                                               tp=2,
                                               max_batch_size=8),
                            gen_config=dict(top_k=1,
                                            top_p=0.8,
                                            temperature=1.0,
                                            max_new_tokens=256),
                            max_out_len=256,
                            max_seq_len=2048,
                            batch_size=8,
                            concurrency=8,
                            meta_template=mistral_meta_template,
                            run_cfg=dict(num_gpus=2, num_procs=1),
                            end_str='</s>')

# ===== Configs for Qwen/Qwen1.5-7B-Chat =====
hf_qwen1dot5_chat_7b = dict(type=HuggingFaceCausalLM,
                            abbr='qwen1.5-7b-chat-hf',
                            path='Qwen/Qwen1.5-7B-Chat',
                            model_kwargs=dict(device_map='auto',
                                              trust_remote_code=True),
                            tokenizer_kwargs=dict(padding_side='left',
                                                  truncation_side='left',
                                                  trust_remote_code=True,
                                                  use_fast=False),
                            meta_template=qwen1dot5_meta_template,
                            pad_token_id=151645,
                            max_out_len=256,
                            max_seq_len=2048,
                            batch_size=8,
                            batch_padding=False,
                            run_cfg=dict(num_gpus=1, num_procs=1),
                            end_str='<|im_end|>')

pt_qwen1dot5_chat_7b = dict(type=LmdeployPytorchModel,
                            abbr='qwen1.5-7b-chat-pytorch',
                            path='Qwen/Qwen1.5-7B-Chat',
                            engine_config=dict(session_len=2048,
                                               cache_max_entry_count=0.5,
                                               max_batch_size=16),
                            gen_config=dict(top_k=1,
                                            top_p=0.8,
                                            temperature=1.0,
                                            max_new_tokens=256),
                            max_out_len=256,
                            max_seq_len=2048,
                            batch_size=16,
                            concurrency=16,
                            meta_template=qwen1dot5_meta_template,
                            run_cfg=dict(num_gpus=1, num_procs=1),
                            end_str='<|im_end|>')

# ===== Configs for google/gemma-7b-it =====
hf_gemma_chat_7b = dict(type=HuggingFaceCausalLM,
                        abbr='gemma-7b-it-pytorch',
                        path='google/gemma-7b-it',
                        tokenizer_path='google/gemma-7b-it',
                        model_kwargs=dict(device_map='auto',
                                          trust_remote_code=True),
                        tokenizer_kwargs=dict(
                            padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                        ),
                        meta_template=mistral_meta_template,
                        max_out_len=256,
                        max_seq_len=2048,
                        batch_size=16,
                        batch_padding=False,
                        run_cfg=dict(num_gpus=1, num_procs=1),
                        end_str='end_of_turn')

pt_gemma_chat_7b = dict(type=LmdeployPytorchModel,
                        abbr='gemma-7b-it-pytorch',
                        path='google/gemma-7b-it',
                        engine_config=dict(session_len=2048,
                                           max_batch_size=16),
                        gen_config=dict(top_k=1,
                                        top_p=0.8,
                                        temperature=1.0,
                                        max_new_tokens=256),
                        max_out_len=256,
                        max_seq_len=2048,
                        batch_size=16,
                        concurrency=16,
                        meta_template=gemma_meta_template,
                        run_cfg=dict(num_gpus=1, num_procs=1),
                        end_str='<end_of_turn>')

# #!/bin/bash

dataset_path="benchmark/ShareGPT_V3_unfiltered_cleaned_split.json"
########################################## TurboMind engine: fp16 or bf16 ##########################################
## 7B. gemm_tune -> profile_throughput
tp=1
max_batch_size=256
cache_max_entry_count=0.95
model_path="/workspace/models-140/llama2/huggingface/llama-2-7b-chat"
python3 -m lmdeploy.turbomind.generate_gemm_config --tensor-para-size ${tp} --max-batch-size ${max_batch_size} --model-path ${model_path}
python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count}

## 13B. gemm_tune -> profile_throughput
tp=1
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models-140/llama2/huggingface/llama-2-13b-chat"
python3 -m lmdeploy.turbomind.generate_gemm_config --tensor-para-size ${tp} --max-batch-size ${max_batch_size} --model-path ${model_path}
python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count}

# 20B. gemm_tune -> profile_throughput
tp=2
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models-140/InternLM/internlm-chat-20b"
python3 -m lmdeploy.turbomind.generate_gemm_config --tensor-para-size ${tp} --max-batch-size ${max_batch_size} --model-path ${model_path}
python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count}

# 70B
tp=1
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models-140/llama2/huggingface/llama-2-70b-chat-hf"
python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count}

# ########################################## TurboMind engine: w4a16 ##########################################
## 7B
tp=1
max_batch_size=256
cache_max_entry_count=0.95
model_path="/workspace/models/quantization/llama-2-7b-chat-4bit"
python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --model-format awq --num-prompts 10000

## 13B
tp=1
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models-140/llama2/huggingface/llama-2-13b-chat"
python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --model-format awq --num-prompts 10000

## 20B
tp=2
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models-140/InternLM/internlm-chat-20b"
python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --model-format awq --num-prompts 10000

## 70B
tp=4
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models-140/llama2/huggingface/llama-2-70b-chat-hf"
python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --model-format awq --num-prompts 10000

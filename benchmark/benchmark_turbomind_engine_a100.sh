# #!/bin/bash

dataset_path="benchmark/ShareGPT_V3_unfiltered_cleaned_split.json"
########################################## TurboMind engine: fp16 or bf16 ##########################################
# 7B. gemm_tune -> profile_throughput
tp=1
max_batch_size=256
cache_max_entry_count=0.95
model_path="/workspace/models-140/llama2/huggingface/llama-2-7b-chat"
CUDA_VISIBLE_DEVICES="6" python3 -m lmdeploy.turbomind.generate_gemm_config --tensor-para-size ${tp} --max-batch-size ${max_batch_size} --model-path ${model_path}
CUDA_VISIBLE_DEVICES="6" python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --csv llama2_7b_thr.csv
rm gemm_config.in

# 13B. gemm_tune -> profile_throughput
tp=1
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models-140/llama2/huggingface/llama-2-13b-chat"
CUDA_VISIBLE_DEVICES="6" python3 -m lmdeploy.turbomind.generate_gemm_config --tensor-para-size ${tp} --max-batch-size ${max_batch_size} --model-path ${model_path}
CUDA_VISIBLE_DEVICES="6" python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --csv llama2_13b_thr.csv
rm gemm_config.in

# 20B. gemm_tune -> profile_throughput
tp=2
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models-140/InternLM/internlm-chat-20b"
CUDA_VISIBLE_DEVICES="5,6" python3 -m lmdeploy.turbomind.generate_gemm_config --tensor-para-size ${tp} --max-batch-size ${max_batch_size} --model-path ${model_path}
CUDA_VISIBLE_DEVICES="5,6" python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --csv internlm_20b_thr.csv
rm gemm_config.in

# 70B
tp=4
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models-140/llama2/huggingface/llama-2-70b-chat-hf"
CUDA_VISIBLE_DEVICES="4,5,6,7" python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --csv llama2_70b_thr.csv

# ########################################## TurboMind engine: w4a16 ##########################################
# 7B
tp=1
max_batch_size=256
cache_max_entry_count=0.95
model_path="/workspace/models/quantization/llama-2-7b-chat-4bit"
CUDA_VISIBLE_DEVICES="6" python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --model-format awq --num-prompts 10000 --csv llama2_7b_4bit_thr.csv

# 13B
tp=1
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models/quantization/llama-2-13b-chat-4bit"
CUDA_VISIBLE_DEVICES="6" python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --model-format awq --num-prompts 10000 --csv llama2_13b_4bit_thr.csv

# 20B
tp=2
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models/quantization/internlm-chat-20b-4bit"
CUDA_VISIBLE_DEVICES="5,6" python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --model-format awq --num-prompts 10000 --csv internlm_20b_4bit_thr.csv

# 70B
tp=4
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models/quantization/llama-2-70b-chat-hf-4bit"
CUDA_VISIBLE_DEVICES="4,5,6,7" python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --model-format awq --num-prompts 10000 --csv llama2_70b_4bit_thr.csv

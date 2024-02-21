#!/bin/bash

dataset_path="benchmark/ShareGPT_V3_unfiltered_cleaned_split.json"
########################################## PyTorch engine: fp16 or bf16 ##########################################
## 7B
tp=1
max_batch_size=256
cache_max_entry_count=0.95
model_path="/workspace/models-140/llama2/huggingface/llama-2-7b-chat"
CUDA_VISIBLE_DEVICES="6" python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --backend pytorch --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --csv llama2_pt_7b_thr.csv

## 13B
tp=1
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models-140/llama2/huggingface/llama-2-13b-chat"
CUDA_VISIBLE_DEVICES="6" python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --backend pytorch --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --csv llama2_pt_7b_thr.csv

# 20B
tp=2
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models-140/InternLM/internlm-chat-20b"
CUDA_VISIBLE_DEVICES="5,6" python3 benchmark/profile_throughput.py ${dataset_path} ${model_path}  --backend pytorch --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --csv llama2_pt_7b_thr.csv

# 70B
tp=4
max_batch_size=256
cache_max_entry_count=0.9
model_path="/workspace/models-140/llama2/huggingface/llama-2-70b-chat-hf"
CUDA_VISIBLE_DEVICES="4,5,6,7" python3 benchmark/profile_throughput.py ${dataset_path} ${model_path}  --backend pytorch --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --csv llama2_pt_7b_thr.csv

########################################## PyTorch engine: w8a8 ##########################################

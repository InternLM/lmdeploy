# #!/bin/bash

dataset_path="benchmark/ShareGPT_V3_unfiltered_cleaned_split.json"
########################################## TurboMind engine: fp16 or bf16 ##########################################
## 7B. gemm_tune -> profile_throughput
tp=1
cache_max_entry_count=0.95
model_path="/workspace/models-140/llama2/huggingface/llama-2-7b-chat"
python3 -m lmdeploy.turbomind.generate_gemm_config --tensor-para-size ${tp} --max-batch-size 256 --model-path ${model_path}
python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --cache-max-entry-count ${cache_max_entry_count}

## 13B. gemm_tune -> profile_throughput
tp=1
cache_max_entry_count=0.9
model_path="/workspace/models-140/llama2/huggingface/llama-2-13b-chat"
python3 -m lmdeploy.turbomind.generate_gemm_config --tensor-para-size ${tp} --max-batch-size 256 --model-path ${model_path}
python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --cache-max-entry-count ${cache_max_entry_count}
exit 0
# 20B. gemm_tune -> profile_throughput
tp=2
cache_max_entry_count=0.9
model_path="/workspace/models-140/InternLM/internlm-chat-20b"
python3 -m lmdeploy.turbomind.generate_gemm_config --tensor-para-size ${tp} --max-batch-size 256 --model-path ${model_path}
python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --cache-max-entry-count ${cache_max_entry_count}

# 70B
tp=1
cache_max_entry_count=0.9
model_path="/workspace/models-140/llama2/huggingface/llama-2-70b-chat-hf"
python3 benchmark/profile_throughput.py ${dataset_path} ${model_path} --cache-max-entry-count ${cache_max_entry_count}

########################################## TurboMind engine: w4a16 ##########################################

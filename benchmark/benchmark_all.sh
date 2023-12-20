#!/bin/bash
apt-get update
apt-get install crudini -y

# fp16 or bf16
# CUDA_VISIBLE_DEVICES=7 ./benchmark/benchmark.sh ./benchmark/config/llama_7b.ini
# CUDA_VISIBLE_DEVICES=7 ./benchmark/benchmark.sh ./benchmark/config/llama_13b.ini
# CUDA_VISIBLE_DEVICES="6,7" ./benchmark/benchmark.sh ./benchmark/config/internlm_20b.ini
# CUDA_VISIBLE_DEVICES="4,5,6,7" ./benchmark/benchmark.sh ./benchmark/config/llama_70b.ini

# w4a16
# CUDA_VISIBLE_DEVICES=7 ./benchmark/benchmark.sh ./benchmark/config/llama_7b_4bit.ini
CUDA_VISIBLE_DEVICES=7 ./benchmark/benchmark.sh ./benchmark/config/llama_13b_4bit.ini
CUDA_VISIBLE_DEVICES="6,7" ./benchmark/benchmark.sh ./benchmark/config/internlm_20b_4bit.ini
CUDA_VISIBLE_DEVICES="4,5,6,7" ./benchmark/benchmark.sh ./benchmark/config/llama_70b_4bit.ini

#!/bin/bash
tp=1
model_name=llama2
model_path=/workspace/models-140/llama2/huggingface/llama-2-7b-chat/
turbomind_model_path=workspace/llama2-7b-chat
foldername=$(basename "$turbomind_model_path")

# convert
lmdeploy convert ${model_name} ${model_path} --dst-path ${turbomind_model_path} --tp ${tp}

# update recommended config to config.ini
config_path=${turbomind_model_path}/triton_models/weights/config.ini

apt-get install crudini

crudini --set ${config_path} llama max_context_token_num 4
crudini --set ${config_path} llama cache_chunk_size -1
crudini --set ${config_path} llama cache_max_entry_count 1000
crudini --set ${config_path} llama max_batch_size 128
# end of update config

benchmark_rpm () {
    output_path=$1
    mkdir -p "${output_path}"

    batches=(64 128)
    for batch in "${batches[@]}"
    do
        for i in {1..3}
        do
        python3 benchmark/profile_throughput.py \
            benchmark/ShareGPT_V3_unfiltered_cleaned_split.json \
            ${turbomind_model_path} \
            --concurrency "$batch" \
            --num_prompts 3000 \
            --csv ${output_path}/rpm_localhost_batch_"${batch}"_"${i}"th.csv
        done
    done
}

benchmark_generation () {
    output_path=$1
    mkdir -p "${output_path}"

    python3 benchmark/profile_generation.py \
    ${turbomind_model_path} \
    --concurrency 1 16 32 64 \
    --csv ${output_path}/generation.csv
}

################################# BENCHMARK AFTER TUNING GEMM #################################
output_path=benchmark/output/"${foldername}"-tunned-gemm-tp"${tp}"

# tune gemm
head_num=$(crudini --get "${config_path}" llama head_num)
size_per_head=$(crudini --get "${config_path}" llama size_per_head)
vocab_size=$(crudini --get "${config_path}" llama vocab_size)
inter_size=$(crudini --get "${config_path}" llama inter_size)
tensor_para_size=$(crudini --get "${config_path}" llama tensor_para_size)
max_batch_size=$(crudini --get "${config_path}" llama max_batch_size)

echo $head_num, $size_per_head, $vocab_size, $inter_size, $tensor_para_size, $max_batch_size

python3 lmdeploy/turbomind/generate_gemm_config.py \
    --head_num ${head_num} \
    --size_per_head ${size_per_head} \
    --vocab_size ${vocab_size} \
    --inter_size ${inter_size} \
    --tensor_para_size ${tensor_para_size} \
    --max_batch_size ${max_batch_size}

# benchmark request throughput and static inference
benchmark_rpm ${output_path}
benchmark_generation ${output_path}

mv gemm_config.in ${output_path}

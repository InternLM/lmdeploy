#!/bin/bash
tp=4
model_name=llama2
model_path=/workspace/models-140/llama2/huggingface/llama-2-70b-chat-hf/
turbomind_model_path=workspace/llama2-70b-chat
foldername=$(basename "$turbomind_model_path")

# convert
lmdeploy convert ${model_name} ${model_path} --dst-path ${turbomind_model_path} --tp ${tp}

# update recommended config to config.ini
config_path=${turbomind_model_path}/triton_models/weights/config.ini

apt-get install crudini

crudini --set ${config_path} llama max_context_token_num 4
crudini --set ${config_path} llama cache_chunk_size -1
crudini --set ${config_path} llama cache_max_entry_count 4000
crudini --set ${config_path} llama max_batch_size 256
# end of update config

benchmark_rpm () {
    output_path=$1
    mkdir -p ${output_path}

    batches=(64 128 256)
    for batch in "${batches[@]}"
    do
        for i in {1..6}
        do
        python3 benchmark/profile_throughput.py \
            benchmark/ShareGPT_V3_unfiltered_cleaned_split.json \
            ${turbomind_model_path} \
            --concurrency "$batch" \
            --num_prompts 5000 \
            --csv ${output_path}/rpm_localhost_batch_"${batch}"_"${i}"th.csv
        done
    done
}

benchmark_generation () {
    output_path=$1
    python3 benchmark/profile_generation.py \
    ${turbomind_model_path} \
    --concurrency 1 64 128 256 \
    --csv ${output_path}/generation.csv
}


output_path=benchmark/output/"${foldername}"-tp"${tp}"
# benchmark request throughput and static inference
benchmark_rpm ${output_path}
benchmark_generation  ${output_path}

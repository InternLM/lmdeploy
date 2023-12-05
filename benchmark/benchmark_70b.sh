#!/bin/bash
if [ -z "$1" ]
then
    echo "Error. Please input the model path of llama2-70b model"
    exit 1
fi

workspace_dir=$(dirname $(realpath "$0"))

tp=4
model_path="$1"
model_foldername=$(basename "$model_path")
turbomind_model_path="${workspace_dir}"/workspace/"${model_foldername}"

# convert
lmdeploy convert llama2 ${model_path} --dst-path ${turbomind_model_path} --tp ${tp}
if [ $? != 0 ]
then
    exit 1
fi

# update recommended config to config.ini
config_path=${turbomind_model_path}/triton_models/weights/config.ini

apt-get update
apt-get install crudini -y

crudini --set ${config_path} llama max_context_token_num 4
crudini --set ${config_path} llama cache_chunk_size -1
crudini --set ${config_path} llama cache_max_entry_count 4000
crudini --set ${config_path} llama max_batch_size 256
# end of update config

cd ${workspace_dir}

# download dataset
wget -O ShareGPT_V3_unfiltered_cleaned_split.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

benchmark_rpm () {
    output_path=$1
    mkdir -p "${output_path}"

    batches=(64 128 256)
    for batch in "${batches[@]}"
    do
        for i in {1..3}
        do
        python3 profile_throughput.py \
            ShareGPT_V3_unfiltered_cleaned_split.json \
            ${turbomind_model_path} \
            --concurrency "$batch" \
            --num_prompts 5000 \
            --csv ${output_path}/rpm_localhost_batch_"${batch}"_"${i}"th.csv
        done
    done
}

benchmark_generation () {
    output_path=$1
    mkdir -p "${output_path}"

    python3 profile_generation.py \
    ${turbomind_model_path} \
    --concurrency 1 64 128 256 \
    --csv ${output_path}/generation.csv
}

output_path="${workspace_dir}"/output/"${model_foldername}"-tp"${tp}"
# benchmark request throughput and static inference
benchmark_rpm ${output_path}
benchmark_generation  ${output_path}

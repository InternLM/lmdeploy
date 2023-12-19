#!/bin/bash
if [ -z "$1" ]
then
    echo "Error. Please input the model path of llama2-7b model"
    exit 1
fi

workspace_dir=$(dirname $(realpath "$0"))

tp=1
model_path="$1"
model_foldername=$(basename "$model_path")"-4bit"
turbomind_model_path="${workspace_dir}"/workspace/turbomind/"${model_foldername}"
quantized_model_path="${workspace_dir}"/workspace/quantization/"${model_foldername}"

echo "          model path: ${model_path}"
echo "quantized model path: ${quantized_model_path}"
echo "turbomind model path: ${turbomind_model_path}"

# quantization
echo "start to quantize model..."
lmdeploy lite calibrate ${model_path} --work_dir ${quantized_model_path}
lmdeploy lite auto_awq ${model_path} --work_dir ${quantized_model_path}

# convert
echo "star to convert model..."
lmdeploy convert llama2 ${quantized_model_path} --dst-path ${turbomind_model_path} --tp ${tp} --model-format awq --group-size 128
if [ $? != 0 ]
then
exit 1
fi

# update recommended config to config.ini
config_path=${turbomind_model_path}/triton_models/weights/config.ini

apt-get update
apt-get install crudini -y

# crudini --set ${config_path} llama max_context_token_num 4
crudini --set ${config_path} llama cache_chunk_size -1
crudini --set ${config_path} llama cache_max_entry_count 1000
crudini --set ${config_path} llama max_batch_size 128
# end of update config

cd ${workspace_dir}

# download dataset
wget -O workspace/ShareGPT_V3_unfiltered_cleaned_split.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

benchmark_rpm () {
    output_path=$1
    mkdir -p "${output_path}"

    batches=(64 128)
    for batch in "${batches[@]}"
    do
        for i in {1..3}
        do
        python3 profile_throughput.py \
            workspace/ShareGPT_V3_unfiltered_cleaned_split.json \
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

    python3 profile_generation.py \
        ${turbomind_model_path} \
        --concurrency 1 16 32 64 \
        --csv ${output_path}/generation.csv
}

################################# BENCHMARK AFTER TUNING GEMM #################################
output_path="${workspace_dir}/workspace/output/${model_foldername}-tp${tp}"

# lmdeploy hasn't support tuning gemm for 4bit model
# # tune gemm
# head_num=$(crudini --get "${config_path}" llama head_num)
# size_per_head=$(crudini --get "${config_path}" llama size_per_head)
# vocab_size=$(crudini --get "${config_path}" llama vocab_size)
# inter_size=$(crudini --get "${config_path}" llama inter_size)
# tensor_para_size=$(crudini --get "${config_path}" llama tensor_para_size)
# max_batch_size=$(crudini --get "${config_path}" llama max_batch_size)

# echo $head_num, $size_per_head, $vocab_size, $inter_size, $tensor_para_size, $max_batch_size

# python3 -m lmdeploy.turbomind.generate_gemm_config \
#     --head_num ${head_num} \
#     --size_per_head ${size_per_head} \
#     --vocab_size ${vocab_size} \
#     --inter_size ${inter_size} \
#     --tensor_para_size ${tensor_para_size} \
#     --max_batch_size ${max_batch_size}

# benchmark request throughput and static inference
benchmark_rpm ${output_path}
benchmark_generation ${output_path}

# mv gemm_config.in ${output_path}

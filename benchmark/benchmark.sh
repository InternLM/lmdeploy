#!/bin/bash
# bash benchmark_fp16.sh <model_path> <engine_config>

benchmark_rpm () {
    echo "start to benchmark rpm..."
    turbomind_model_path=$1
    dataset_path=$2
    output_path=$3

    batches=(64 128)
    for batch in "${batches[@]}"
    do
        for i in {1..3}
        do
        python3 profile_throughput.py \
            ${dataset_path} \
            ${turbomind_model_path} \
            --concurrency "$batch" \
            --num_prompts 3000 \
            --csv ${output_path}/rpm_localhost_batch_"${batch}"_"${i}"th.csv
        done
    done
}

benchmark_generation () {
    echo "start to benchmark generation..."
    turbomind_model_path=$1
    output_path=$2

    python3 profile_generation.py \
        ${turbomind_model_path} \
        --concurrency 1 16 32 64 \
        --csv ${output_path}/generation.csv
}

tune_gemm () {
    echo "start to tune gemm..."
    turbomind_config_path=$1
    output_path=$2

    head_num=$(crudini --get "${turbomind_config_path}" llama head_num)
    size_per_head=$(crudini --get "${turbomind_config_path}" llama size_per_head)
    vocab_size=$(crudini --get "${turbomind_config_path}" llama vocab_size)
    inter_size=$(crudini --get "${turbomind_config_path}" llama inter_size)
    tensor_para_size=$(crudini --get "${turbomind_config_path}" llama tensor_para_size)
    max_batch_size=$(crudini --get "${turbomind_config_path}" llama max_batch_size)

    echo $head_num, $size_per_head, $vocab_size, $inter_size, $tensor_para_size, $max_batch_size

    python3 -m lmdeploy.turbomind.generate_gemm_config \
        --head_num ${head_num} \
        --size_per_head ${size_per_head} \
        --vocab_size ${vocab_size} \
        --inter_size ${inter_size} \
        --tensor_para_size ${tensor_para_size} \
        --max_batch_size ${max_batch_size}
    cp gemm_config.in ${output_path}
}

workspace_dir=$(dirname $(realpath "$0"))

if [ -z "$1" ]
then
    echo "Error. Please input the path of test setting config"
    exit 1
fi

config_path="$1"
echo "---------- input config is ----------"
cat ${config_path}
echo "-------------------------------------"

# get engine config
turbomind_model_path=$(crudini --get "${config_path}" llama turbomind_model_path 2>/dev/null)
model_path=$(crudini --get "${config_path}" llama model_path 2>/dev/null)
dataset_path=$(crudini --get "${config_path}" llama dataset_path 2>/dev/null)
tp=$(crudini --get "${config_path}" llama tp 2>/dev/null)
tune_gemm=$(crudini --get "${config_path}" llama tune_gemm 2>/dev/null)
w4a16=$(crudini --get "${config_path}" llama w4a16 2>/dev/null)
kvint8=$(crudini --get "${config_path}" llama kvint8 2>/dev/null)
max_context_token_num=$(crudini --get "${config_path}" llama max_context_token_num 2>/dev/null)
cache_chunk_size=$(crudini --get "${config_path}" llama cache_chunk_size 2>/dev/null)
cache_max_entry_count=$(crudini --get "${config_path}" llama cache_max_entry_count 2>/dev/null)
max_batch_size=$(crudini --get "${config_path}" llama max_batch_size 2>/dev/null)
profile_rpm=$(crudini --get "${config_path}" llama profile_rpm 2>/dev/null)
profile_generation=$(crudini --get "${config_path}" llama profile_generation 2>/dev/null)


if [ -n "${turbomind_model_path}" ]
then
    echo "turbomind model path is provided: ${turbomind_model_path}"
    model_foldername=$(basename "$turbomind_model_path")
else
    echo "turbomind model path is not provided."
    echo "model path is provided: ${model_path}"
    model_foldername=$(basename "$model_path")
    turbomind_model_path="${workspace_dir}"/workspace/turbomind/"${model_foldername}"
    quantized_model_path="${workspace_dir}"/workspace/quantization/"${model_foldername}"
    output_path="${workspace_dir}/workspace/output/${model_foldername}-tp${tp}"

    if [ "$w4a16" == 1 ]
    then
        echo "start to quantize model..."
        lmdeploy lite calibrate ${model_path} --work_dir ${quantized_model_path}
        lmdeploy lite auto_awq ${model_path} --work_dir ${quantized_model_path}
        # after quantization, the source model path will be the quantized model path
        model_path=${quantized_model_path}
        model_format=awq
        group_size=128
    else
        model_format=hf
        group_size=0
    fi

    # convert
    echo "start converting..."
    lmdeploy convert llama2 ${model_path} --dst-path ${turbomind_model_path} --tp ${tp} --model-format ${model_format} --group-size ${group_size}
    if [ $? != 0 ]
    then
        echo "convert model ${model_path} failed"
        exit 1
    fi
fi

output_path="${workspace_dir}/workspace/output/${model_foldername}-tp${tp}"
mkdir -p ${output_path}
cp ${config_path} ${output_path}

# update engine config to config.ini
turbomind_config_path=${turbomind_model_path}/triton_models/weights/config.ini
crudini --set ${turbomind_config_path} llama max_context_token_num ${max_context_token_num}
crudini --set ${turbomind_config_path} llama cache_chunk_size ${cache_chunk_size}
crudini --set ${turbomind_config_path} llama cache_max_entry_count ${cache_max_entry_count}
crudini --set ${turbomind_config_path} llama max_batch_size ${max_batch_size}
cat ${turbomind_config_path}
# end of update config

cd ${workspace_dir}

# tune gemm
if [ "${tune_gemm}" == 1 ]
then
    tune_gemm ${turbomind_config_path} ${output_path}
fi

# benchmark request throughput
if [ "${profile_rpm}" == 1 ]
then
    benchmark_rpm ${turbomind_model_path} ${dataset_path} ${output_path}
fi

# benchmark static inference
if [ "${profile_generation}" ==  1 ]
then
    benchmark_generation ${turbomind_model_path} ${output_path}
fi

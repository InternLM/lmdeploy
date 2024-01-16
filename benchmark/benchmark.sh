#!/bin/bash

benchmark_rpm () {
    echo "start to benchmark rpm..."
    model_path=$1
    model_format=$2
    dataset_path=$3
    max_batch_size=$4
    cache_count=$5
    output_path=$6

    batches=(64 128 256)
    for batch in "${batches[@]}"
    do
        if [ "$batch" -gt "$max_batch_size" ]
        then
            continue
        fi

        python3 profile_throughput.py \
            ${dataset_path} \
            ${model_path} \
            --concurrency "$batch" \
            --cache-count "$cache_count" \
            --csv ${output_path}/rpm_localhost_batch_"${batch}".csv
    done
}

benchmark_generation () {
    echo "start to benchmark generation..."
    model_path=$1
    model_format=$2
    max_batch_size=$3
    cache_count=$4
    output_path=$5

    python3 profile_generation.py \
        ${model_path} \
        --concurrency 1 16 32 64 \
        --warmup-round 1 --test-round 3 \
        --cache-count "$cache_count" \
        --csv ${output_path}/generation.csv
}

tune_gemm () {
    echo "start to tune gemm..."
    model_path=$1
    tp=$2
    max_batch_size=$3
    output_path=$4

    python3 -m lmdeploy.turbomind.generate_gemm_config \
        --tensor-para-size $tp \
        --max-batch-size $max_batch_size \
        --model-path ${model_path}
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

# read config file
model_path=$(crudini --get "${config_path}" llama model_path 2>/dev/null)
dataset_path=$(crudini --get "${config_path}" llama dataset_path 2>/dev/null)
tp=$(crudini --get "${config_path}" llama tp 2>/dev/null)
tune_gemm=$(crudini --get "${config_path}" llama tune_gemm 2>/dev/null)
kvint8=$(crudini --get "${config_path}" llama kvint8 2>/dev/null)
cache_max_entry_count=$(crudini --get "${config_path}" llama cache_max_entry_count 2>/dev/null)
max_batch_size=$(crudini --get "${config_path}" llama max_batch_size 2>/dev/null)
profile_rpm=$(crudini --get "${config_path}" llama profile_rpm 2>/dev/null)
profile_generation=$(crudini --get "${config_path}" llama profile_generation 2>/dev/null)



model_foldername=$(basename "$model_path")

if [ "$w4a16" == 1 ]
then
    model_format=awq
    group_size=128
else
    model_format=hf
    group_size=0
fi

output_path="${workspace_dir}/workspace/output/${model_foldername}-tp${tp}"
mkdir -p ${output_path}
cp ${config_path} ${output_path}

cd ${workspace_dir}

# tune gemm
if [ "${tune_gemm}" -eq 1 ]
then
    tune_gemm ${model_path} ${tp} ${max_batch_size} ${output_path}
fi

# benchmark request throughput
if [ "${profile_rpm}" -eq 1 ]
then
    benchmark_rpm ${model_path} ${model_format} ${dataset_path} ${max_batch_size} ${cache_max_entry_count} ${output_path}
fi

# benchmark static inference
if [ "${profile_generation}" -eq 1 ]
then
    benchmark_generation ${model_path} ${model_format} ${max_batch_size} ${cache_max_entry_count} ${output_path}
fi

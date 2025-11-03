# Model Evaluation Guide

This document describes how to evaluate a model's capabilities on academic datasets using OpenCompass and LMDeploy. The complete evaluation process consists of two main stages: inference stage and evaluation stage.

During the inference stage, the target model is first deployed as an inference service using LMDeploy. OpenCompass then sends dataset content as requests to this service and collects the generated responses.

In the evaluation stage, the OpenCompass evaluation model `opencompass/CompassVerifier-32B` is deployed as a service via LMDeploy. OpenCompass subsequently submits the inference results to this service to obtain final evaluation scores.

If sufficient computational resources are available, please refer to the [End-to-End Evaluation](#end-to-end-evaluation) section for complete workflow execution. Otherwise, we recommend following the [Step-by-Step Evaluation](#step-by-step-evaluation) section to execute both stages sequentially.

## Environment Setup

```shell
pip install lmdeploy
pip install "opencompass[full]"

# Download the lmdeploy source code, which will be used in subsequent steps to access eval/gen_config.py
git clone --depth=1 https://github.com/InternLM/lmdeploy.git
```

It is recommended to install LMDeploy and OpenCompass in separate Python virtual environments to avoid potential dependency conflicts.

## End-to-End Evaluation

1. **Deploy Target Model**

```shell
lmdeploy serve api_server <model_path> --server-port 10000 <--other-options>
```

2. **Deploy Evaluation Model (Judger)**

```shell
lmdeploy serve api_server opencompass/CompassVerifier-32B --server-port 20000 --tp 2
```

3. **Generate Evaluation Configuration and Execute**

```shell
# Generate evaluation configuration
cd {the/root/path/of/lmdeploy/repo}
python eval/gen_config.py <task_name> \
    --api-server http://{api-server-ip}:10000 \
    --judger-server http://{judger-server-ip}:20000 \
    -o /path/to/e2e_config.py

# Run evaluation task
## Specify the dataset path. OC will download the datasets automatically if they are
## not found in the path
export HF_DATASETS_CACHE=/nvme4/huggingface_hub/datasets
export COMPASS_DATA_CACHE=/nvme1/shared/opencompass/.cache
opencompass /path/to/e2e_config.py -w {oc_output_dir}
```

For detailed usage instructions about `gen_config.py`, such as specifying evaluation datasets, please run `python evaluation/gen_config.py --help`.

After evaluation completion, results are saved in `{oc_output_dir}/{yyyymmdd_hhmmss}`, where `{yyyymmdd_hhmmss}` represents the task timestamp.

## Step-by-Step Evaluation

### Inference Stage

This stage generates model responses for the dataset.

1. **Deploy Target Model**

```shell
lmdeploy serve api_server <model_path> --server-port 10000 <--other-options>
```

2. **Generate Inference Configuration and Execute**

```shell
# Generate inference configuration
cd {the/root/path/of/lmdeploy/repo}
python eval/gen_config.py <task_name> --mode infer \
    --api-server http://{api_server_ip}:10000 \
    -o /path/to/infer_config.py

# Run inference task
## Specify the dataset path. OC will download the datasets automatically if they are
## not found in the path
export COMPASS_DATA_CACHE=/nvme1/shared/opencompass/.cache
export HF_DATASETS_CACHE=/nvme4/huggingface_hub/datasets
opencompass /path/to/infer_config.py -m infer -w {oc_output_dir}
```

For detailed usage instructions about `gen_config.py`, such as specifying evaluation datasets, please run `python evaluation/gen_config.py --help`.

### Evaluation Stage

This stage uses the evaluation model (Judger) to assess the quality of inference results.

1. **Deploy Evaluation Model (Judger)**

```shell
lmdeploy serve api_server opencompass/CompassVerifier-32B --server-port 20000 --tp 2
```

2. **Generate Evaluation Configuration and Execute**

```shell
# Generate evaluation configuration
cd {the/root/path/of/lmdeploy/repo}
python eval/gen_config.py {task_name} --mode eval \
    --judger-server http://{judger_serverip}:20000 \
    -o /path/to/judger_config.py

# Run evaluation task
## Specify the dataset path. OC will download the datasets automatically if they are
## not found in the path
export COMPASS_DATA_CACHE=/nvme1/shared/opencompass/.cache
export HF_DATASETS_CACHE=/nvme4/huggingface_hub/datasets
opencompass /path/to/judger_config.py -m eval -w {oc_output_dir} -r {yyyymmdd_hhmmss}
```

Important Notes:

- `task_name` must be identical to the one used in the inference stage
- The `oc_output_dir` specified with `-w` must match the directory used in the inference stage
- The `-r` parameter indicates "previous outputs & results" and should specify the timestamp directory generated during the inference stage (the subdirectory under `{oc_output_dir}`)

For detailed usage instructions about `gen_config.py`, such as specifying evaluation datasets, please run `python evaluation/gen_config.py --help`.

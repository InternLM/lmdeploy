# 模型评测指南

本文档介绍如何使用 OpenCompass 和 LMDeploy 对模型在学术数据集上的能力进行评测。完整的评测流程包含两个主要阶段：推理阶段和评判阶段。

在推理阶段，首先通过 LMDeploy 将待评测模型部署为推理服务，随后使用 OpenCompass 将数据集内容作为请求发送至该服务，并获取模型生成的结果。

在评判阶段，需将 OpenCompass 提供的评测模型 `opencompass/CompassVerifier-32B` 通过 LMDeploy 部署为服务，再使用 OpenCompass 将推理阶段生成的结果提交至该服务，从而获得最终的评测结果。

若评测资源充足，建议参考[端到端评测](#端到端评测)章节执行完整流程；若资源有限，则建议按照[逐步评测](#逐步评测)章节依次执行两个阶段。

## 环境准备

```shell
pip install lmdeploy
pip install "opencompass[full]"

# 下载 lmdeploy 源码，在后续步骤中会使用到 eval/gen_config.py
git clone --depth=1 https://github.com/InternLM/lmdeploy.git
```

建议将 LMDeploy 和 OpenCompass 安装在不同的 Python 虚拟环境中，以避免可能的依赖冲突。

## 端到端评测

1. **部署待评测模型**

```shell
lmdeploy serve api_server <model_path> --server-port 10000 <--other-options>
```

2. **部署评测模型（Judger）**

```shell
lmdeploy serve api_server opencompass/CompassVerifier-32B --server-port 20000 --tp 2
```

3. **生成评测配置并执行评测**

```shell
# 生成评测配置文件
cd {the/root/path/of/lmdeploy/repo}
python eval/gen_config.py <task_name> \
    --api-server http://{api-server-ip}:10000 \
    --judger-server http://{judger-server-ip}:20000 \
    -o /path/to/e2e_config.py

# 执行评测任务
## 指定数据集路径。如果在路径下没有找到评测数据集，OC会自动下载
export HF_DATASETS_CACHE=/nvme4/huggingface_hub/datasets
export COMPASS_DATA_CACHE=/nvme1/shared/opencompass/.cache
opencompass /path/to/e2e_config.py -w {oc_output_dir}
```

关于 `gen_config.py` 的详细使用方法，比如指定评测集，请通过 `python evaluation/gen_config.py --help` 查阅。

评测任务完成后，结果将保存在 `{oc_output_dir}/{yyyymmdd_hhmmss}` 目录中，其中 `{yyyymmdd_hhmmss}` 为任务执行的时间戳。

## 逐步评测

### 推理阶段

本阶段用于生成模型对数据集的回答结果。

1. **部署待评测模型**

```shell
lmdeploy serve api_server <model_path> --server-port 10000 <--other-options>
```

2. **生成推理配置并执行推理**

```shell
# 生成推理配置文件
cd {the/root/path/of/lmdeploy/repo}
python eval/gen_config.py <task_name> --mode infer \
    --api-server http://{api_server_ip}:10000 \
    -o /path/to/infer_config.py

# 执行推理任务
## 指定数据集路径。如果在路径下没有找到评测数据集，OC会自动下载
export HF_DATASETS_CACHE=/nvme4/huggingface_hub/datasets
export COMPASS_DATA_CACHE=/nvme1/shared/opencompass/.cache
opencompass /path/to/infer_config.py -m infer -w {oc_output_dir}
```

关于 `gen_config.py` 的详细使用方法，比如指定评测集，请通过 `python evaluation/gen_config.py --help` 查阅。

### 评判阶段

本阶段由评测模型（Judger）对推理阶段生成的结果进行判断。

1. **部署评测模型（Judger）**

```shell
lmdeploy serve api_server opencompass/CompassVerifier-32B --server-port 20000 --tp 2
```

2. **生成评判配置并执行评判**

```shell
# 生成评判配置文件
cd {the/root/path/of/lmdeploy/repo}
python eval/gen_config.py {task_name} --mode eval \
    --judger-server http://{judger_serverip}:20000 \
    -o /path/to/judger_config.py

# 执行评判任务
## 指定数据集路径。如果在路径下没有找到评测数据集，OC会自动下载
export HF_DATASETS_CACHE=/nvme4/huggingface_hub/datasets
export COMPASS_DATA_CACHE=/nvme1/shared/opencompass/.cache
opencompass /path/to/judger_config.py -m eval -w {oc_output_dir} -r {yyyymmdd_hhmmss}
```

注意事项：

- `task_name` 必须与推理阶段的任务名称保持一致
- `-w` 参数指定的输出目录 `oc_output_dir` 需与推理阶段一致
- `-r` 参数用于指定“之前的输出与结果”，应填入推理阶段生成的时间戳目录名，即 `{oc_output_dir}` 下的子目录名称

关于 `gen_config.py` 的详细使用方法，比如指定评测集，请通过 `python evaluation/gen_config.py --help` 查阅。

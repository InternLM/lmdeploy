# 性能测试

测试之前，请安装 lmdeploy 预编译包，并下载测试脚本和数据。

```shell
pip install lmdeploy
# 下载 lmdeploy 源码，获取其中的性能测试脚本
git clone --depth=1 https://github.com/InternLM/lmdeploy
cd lmdeploy
# 切换到与已安装 lmdeploy 版本对应的 tag：
git fetch --tags
# 查看已安装 lmdeploy 的版本：
pip show lmdeploy | grep Version
# 切换到对应的 tag（将 <version> 替换为实际的版本号）：
git checkout <version>
# 下载测试数据
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## 测试 pipeline 接口

```shell
python3 benchmark/profile_pipeline_api.py ShareGPT_V3_unfiltered_cleaned_split.json meta-llama/Meta-Llama-3-8B-Instruct
```

可通过 `python3 benchmark/profile_pipeline_api.py -h` 查看脚本中的参数详情

## 测试推理引擎接口

```shell
python3 benchmark/profile_throughput.py ShareGPT_V3_unfiltered_cleaned_split.json meta-llama/Meta-Llama-3-8B-Instruct
```

可通过 `python3 benchmark/profile_throughput.py -h` 查看脚本中的参数详情

## 测试 api_server 性能

启动模型服务（可以参考[这里](../llm/api_server.md)）。接着，使用下面的命令:

```shell
python3 benchmark/profile_restful_api.py --backend lmdeploy  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```

关于 `profile_restful_api.py`的帮助信息，可以通过`python3 benchmark/profile_restful_api.py -h`查阅

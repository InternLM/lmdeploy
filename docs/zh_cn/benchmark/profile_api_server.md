# api_server 性能测试

测试之前，请安装 lmdeploy 预编译包，并下载测试脚本和数据。

```shell
pip install lmdeploy
git clone --depth=1 https://github.com/InternLM/lmdeploy
cd lmdeploy/benchmark
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

然后，启动模型服务（可以参考[这里](../llm/api_server.md)）。接着，使用下面的命令:

```shell
python3 profile_restful_api.py --backend lmdeploy  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```

关于 `profile_restful_api.py`的帮助信息，可以通过`python3 profile_restful_api.py -h`查阅

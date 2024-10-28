# Profile API Server

Before benchmarking the api_server, please install the lmdeploy precompiled package and download the script and the test dataset:

```shell
pip install lmdeploy
git clone --depth=1 https://github.com/InternLM/lmdeploy
cd lmdeploy/benchmark
```

Launch the server first (you may refer [here](../llm/api_server.md) for guide) and run the following command:

```shell
python3 benchmark/profile_restful_api.py --backend lmdeploy --num-prompts 5000 --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```

For detailed argument specification of `profile_restful_api.py`, please run the help command `python3 benchmark/profile_restful_api.py -h`.

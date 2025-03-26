# Benchmark

Please install the lmdeploy precompiled package and download the script and the test dataset:

```shell
pip install lmdeploy
# clone the repo to get the benchmark script
git clone --depth=1 https://github.com/InternLM/lmdeploy
cd lmdeploy
# switch to the tag corresponding to the installed version:
git fetch --tags
# Check the installed lmdeploy version:
pip show lmdeploy | grep Version
# Then, check out the corresponding tag (replace <version> with the version string):
git checkout <version>
# download the test dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Benchmark offline pipeline API

```shell
python3 benchmark/profile_pipeline_api.py ShareGPT_V3_unfiltered_cleaned_split.json meta-llama/Meta-Llama-3-8B-Instruct
```

For a comprehensive list of available arguments, please execute `python3 benchmark/profile_pipeline_api.py -h`

## Benchmark offline engine API

```shell
python3 benchmark/profile_throughput.py ShareGPT_V3_unfiltered_cleaned_split.json meta-llama/Meta-Llama-3-8B-Instruct
```

Detailed argument specification can be retrieved by running `python3 benchmark/profile_throughput.py -h`

## Benchmark online serving

Launch the server first (you may refer [here](../llm/api_server.md) for guide) and run the following command:

```shell
python3 benchmark/profile_restful_api.py --backend lmdeploy --num-prompts 5000 --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```

For detailed argument specification of `profile_restful_api.py`, please run the help command `python3 benchmark/profile_restful_api.py -h`.

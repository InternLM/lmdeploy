# Benchmark

We provide several profiling tools to benchmark our models.

## profiling with dataset

Download the dataset below or create your own dataset.

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Profiling your model with `profile_throughput.py`

```bash
python profile_throughput.py \
 ShareGPT_V3_unfiltered_cleaned_split.json \
 /path/to/your/model \
 ${ModelType} \
 --concurrency 64
```

## profile without dataset

`profile_generation.py` perform benchmark with dummy data.

```bash
python profile_generation.py \
 /path/to/your/model \
 ${ModelType} \
 --concurrency 8 --input_seqlen 0 --output_seqlen 2048
```

## profile serving

Tools above profile models with Python API. `profile_serving.py` is used to do benchmark on serving.

```bash
python profile_serving.py \
    ${TritonServerAddress} \
    ${ModelName} \
    /path/to/tokenizer \
    /path/to/dataset \
    --concurrency 64
```

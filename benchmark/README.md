# Benchmark

We provide several profiling tools to benchmark our models.

## profile with dataset

Download the dataset below or create your own dataset.

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Profiling your model with `profile_throughput.py`

```bash
python profile_throughput.py \
 ShareGPT_V3_unfiltered_cleaned_split.json \
 /path/to/your/model \
 --concurrency 64
```

## profile restful api

`profile_restful_api.py` is used to do benchmark on api server.

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python3 profile_restful_api.py --backend lmdeploy --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json
```

## benchmark guided decoding

`benchmark_guided.py` measures the overhead of guided decoding
(`response_format`: json_schema / json_object / regex_schema) by running the
same workload with and without guided decoding, then printing a side-by-side
comparison.

### Why per-token latency is the primary metric

Guided decoding changes the output length distribution: the grammar may cause
early termination (no valid continuation) or force longer output.  Therefore
throughput (tok/s, req/s) is **length-biased** and not a fair comparison.  The
**primary metric is TPOT / ITL** (per-token latency), which directly reflects
the grammar bitmask overhead regardless of output length.  The comparison table
also computes a "per-token overhead %" for a quick summary.

### Quick start

```bash
# JSON schema (default schema), compare with baseline
python3 benchmark_guided.py \
    ShareGPT_V3_unfiltered_cleaned_split.json \
    Qwen/Qwen2.5-7B-Instruct \
    --response-format json_schema \
    --concurrency 64

# JSON object
python3 benchmark_guided.py \
    ShareGPT_V3_unfiltered_cleaned_split.json \
    Qwen/Qwen2.5-7B-Instruct \
    --response-format json_object

# Regex schema
python3 benchmark_guided.py \
    ShareGPT_V3_unfiltered_cleaned_split.json \
    Qwen/Qwen2.5-7B-Instruct \
    --response-format regex_schema \
    --regex-schema '[A-Z][a-z]+ lives in [A-Z][a-z]+\.'

# Custom JSON schema file
python3 benchmark_guided.py \
    ShareGPT_V3_unfiltered_cleaned_split.json \
    Qwen/Qwen2.5-7B-Instruct \
    --response-format json_schema \
    --json-schema-path my_schema.json
```

### Two run modes

| Mode                       | Flag           | When to use                                                                                  |
| -------------------------- | -------------- | -------------------------------------------------------------------------------------------- |
| **Natural stop** (default) | *(none)*       | Production-realistic. Both runs stop naturally. TPOT is the fair metric.                     |
| **Forced length**          | `--ignore-eos` | Isolates pure per-step grammar bitmask overhead. Both runs try to generate `max_new_tokens`. |

### Key arguments

| Argument             | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| `--response-format`  | **Required.** One of `json_schema`, `json_object`, `regex_schema`. |
| `--json-schema-path` | Path to a `.json` schema file. Uses a built-in default if omitted. |
| `--regex-schema`     | Regex pattern string. Uses a built-in default if omitted.          |
| `--ignore-eos`       | Force `max_new_tokens` output to isolate pure per-step overhead.   |
| `--no-baseline`      | Skip the baseline run; only benchmark guided decoding.             |
| `--csv`              | Append results (baseline + guided rows) to a CSV file.             |

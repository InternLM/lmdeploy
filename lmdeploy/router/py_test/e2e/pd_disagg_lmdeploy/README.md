# LMDeploy P/D Disaggregation Tests

These tests validate that LMDeploy Router routes OpenAI-compatible traffic
between externally managed LMDeploy Prefill and Decode API servers.

## Architecture

The tests do not manage RDMA, DLSlime, or Mooncake resources. Those must be
provisioned before the run:

- LMDeploy Prefill API servers started with `--role Prefill`.
- LMDeploy Decode API servers started with `--role Decode`.
- Both roles using the same `--migration-backend`.
- The model name reported by all servers matching `MODEL_NAME`.

The router is started with `--lmdeploy-pd-disaggregation`, `--prefill`, and
`--decode`.

## Files

- `run_accuracy_test.sh`: starts the router and runs completion/streaming validation.
- `test_pd_accuracy.py`: checks health, non-streaming completions, and streaming completions.
- `test_lm_eval_accuracy.py`: optional LM-Eval evaluation through the router.
- `tp_config_sweep_test.sh`: runs the accuracy script for externally provisioned TP configurations.

## Run

```bash
export LMDEPLOY_ROUTER_BIN="$PWD/target/release/lmdeploy-router"
export PREFILL_URLS=http://prefill-host:9001
export DECODE_URLS=http://decode-host:9002
export MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
export MODEL_NAME=Qwen2.5-0.5B-Instruct

py_test/e2e/pd_disagg_lmdeploy/run_accuracy_test.sh
```

For a tensor-parallel sweep, provision TP=1 and TP=2 servers and provide their
URLs explicitly:

```bash
TP_SIZES="1 2" \
PREFILL_TP1_URLS=http://prefill-tp1:9001 \
DECODE_TP1_URLS=http://decode-tp1:9002 \
PREFILL_TP2_URLS=http://prefill-tp2:9003 \
DECODE_TP2_URLS=http://decode-tp2:9004 \
py_test/e2e/pd_disagg_lmdeploy/tp_config_sweep_test.sh
```

Optional LM-Eval:

```bash
./run_accuracy_test.sh

python3 test_lm_eval_accuracy.py \
  --router-url http://127.0.0.1:8300 \
  --model "$MODEL_NAME"

# Blocked FP8 Quantization

LMDeploy supports a weight-only blocked FP8 quantization method. This approach quantizes the weights of a model to 8-bit floating-point numbers in a blocked format, which can reduce the model's memory footprint while maintaining good performance on supported hardware.

Before proceeding, please ensure that lmdeploy is installed by following the [installation guide](../get_started/installation.md). A typical installation command is:

```shell
pip install lmdeploy[all]
```

## Quantization

A single command is all that is needed to perform blocked FP8 quantization. The script will load the model, quantize the linear layers to blocked FP8, and save the resulting model and configuration to the specified working directory.

The command for this is `lmdeploy lite blocked_fp8`.

Here is an example of how to quantize `internlm/internlm2_5-7b-chat`:

```shell
export HF_MODEL=OpenGVLab/InternVL3_5-8B
export WORK_DIR=OpenGVLab/InternVL3_5-8B-FP8

lmdeploy lite blocked_fp8 $HF_MODEL \
  --work-dir $WORK_DIR \
  --quant-dtype fp8 \
  --block-size 128
```

Key arguments for the command:

- `--work-dir`: The directory where the quantized model weights and configuration will be saved.
- `--quant-dtype`: The target FP8 format. Can be `float8_e4m3fn` (same as passing 'fp8', recommended) or `float8_e5m2`.
- `--block-size`: The block size for quantization. The default of `128` is generally a good choice.

## Inference

You can perform batched offline inference with the quantized model using both the `turbomind` and `pytorch` backend.

Here is a simple code example:

```python
from lmdeploy import pipeline

pipe = pipeline("OpenGVLab/InternVL3_5-8B-FP8")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)
```

## Service

LMDeploy's `api_server` can be used to serve the blocked FP8 model.

```shell
lmdeploy serve api_server OpenGVLab/InternVL3_5-8B-FP8
```

The default port for the `api_server` is `23333`. Once the server is running, you can interact with it from another terminal using the `api_client`:

```shell
lmdeploy serve api_client http://0.0.0.0:23333
```

You can also view the available API endpoints through the Swagger UI at `http://0.0.0.0:23333`. For more details on the API, please refer to the [API Server documentation](../llm/api_server.md).

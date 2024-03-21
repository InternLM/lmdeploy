# LMDeploy Triton Python Backend

This is an efficient [Triton Python Backend](https://github.com/triton-inference-server/python_backend/tree/main) for LMDeploy.

## Deployment Procedure

### 1. Specify the folder layout

The folder layout of Triton Inference Server model repository should be as follows. More details can be found in the [user guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html).

```
model_repository
└── lmdeploy_model
    ├── 1
    │   ├── model.py
    │   └── weights
    └── config.pbtxt
```

- The `model.py` and `config.pbtxt` are provided here.
- The `weights` folder contains the weights of models. You should download the model weights at first. And then move it to the `weights` folder. You also can use soft link to avoid moving, for example:

```
ln -s /model_download_path/llama2_13b_chat /model_repository_path/lmdeploy_model/1/weights
```

### 2. Run the Triton server

```
tritonserver \
    --model-repository=/path/to/your/model_repository \
    --backend-directory /opt/tritonserver/backends \
    --allow-http=0 \
    --allow-grpc=1 \
    --grpc-port=33337 \
    --allow-metrics=1 \
    --log-info=1 \
    --log-dir /path/to/put/your/logs
```

In the above command, you should specify the path of your `model_repository` prepared in step 1.

## Client Test

Once the deployment done, we can use the provided `client.py` to test the deployed Triton server.

```
python3 client.py
```

You should make sure the server address and port are configured correctly in the client file.

## Profile

`profile_triton_python_backend.py` under benchmark folder is provided to profile this Triton Python backend.

```
python3 benchmark/profile_triton_python_backend.py 0.0.0.0:33337 /model_path/llama2_13b_chat/ /dataset_path/ShareGPT_V3_unfiltered_cleaned_split.json --num_prompts 1000 --concurrency=128
```

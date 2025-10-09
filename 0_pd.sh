model_path="/mnt/137_nvme3/interns1-mini-remote"


CUDA_VISIBLE_DEVICES=2 lmdeploy serve api_server \
    $model_path \
    --server-port 23334 \
    --role Hybrid \
    --proxy-url http://0.0.0.0:8001 \
    --tp 1 \
    --backend pytorch \
    --disable-vision-encoder \
    --log-level INFO

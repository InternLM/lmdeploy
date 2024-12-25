from huggingface_hub import HfApi

api = HfApi()
api.snapshot_download(
    token='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',
    local_dir='/nvme/shared_data/llama3/Llama-3.2-3B-Instruct',
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    repo_type='model',
    ignore_patterns=['*.pth'],
)

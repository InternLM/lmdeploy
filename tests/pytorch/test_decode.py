import torch
from transformers import AutoTokenizer

from lmdeploy.pytorch.decode import Engine, decode_single
from lmdeploy.pytorch.model import init_model


def _test_decode_dist(model_path, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    engine = Engine(model_path, tokenizer=tokenizer)
    probs = engine.decode(prompt, sort=True, max_bs=4, pad=True)

    return probs


def _test_decode_single(model_path, prompt, gpu_id=0):
    model, tokenizer = init_model(model_path)
    model = model.eval()

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    inputs = tokenizer(prompt, return_tensors='pt', padding=True)

    input_ids = inputs.input_ids.cuda(gpu_id)
    # attention_mask = None
    attention_mask = inputs.attention_mask.cuda(gpu_id)
    probs = decode_single(model, input_ids, attention_mask)

    return probs


def test_compare():
    gpu_id = 0

    torch.set_default_device(gpu_id)
    torch.set_printoptions(linewidth=200, edgeitems=5)
    import numpy as np
    np.set_printoptions(linewidth=200, edgeitems=5)

    model_path = 'llama2/huggingface/llama-2-7b'

    prompt = [
        'I believe the meaning of life is to find your gift. The purpose of life is to give it away.',  # noqa: E501
        'Simply put, the theory of relativity states that '
    ] * 8

    p_single = _test_decode_single(model_path, prompt, gpu_id)
    p_dist = _test_decode_dist(model_path, prompt)

    print(p_single[0])
    print(p_dist[0])

    print(p_single[1])
    print(p_dist[1])

    # assert torch.allclose(p_single, p_dist, rtol=1e-3, atol=1e-3)

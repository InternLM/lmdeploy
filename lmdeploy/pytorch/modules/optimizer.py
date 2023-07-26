from lmdeploy.pytorch.modules import attention_pass, norm_pass

from typing import List, Tuple
import random
import torch
from tqdm import tqdm
import time
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
def optimize_inference(model):
    attention_pass(model)
    norm_pass(model)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import perf_counter
import numpy as np
import transformers
from transformers.models.llama import LlamaTokenizer
# hide generation warnings
transformers.logging.set_verbosity_error()

def measure_latency(model, tokenizer, payload, generation_args, device):
    input_ids = tokenizer(payload, return_tensors="pt").input_ids.to(device)
    latencies = []
    # warm up
    for _ in range(20):
        _ =  model.generate(input_ids, **generation_args)
    # Timed run
    for _ in range(50):
        start_time = perf_counter()
        _ = model.generate(input_ids, **generation_args)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    return f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f};", time_p95_ms

model_id = '/nvme/share_data/llama-7b/'
# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast = False)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
optimize_inference(model)
model = model.cuda()
payload = "Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend but it"
input_ids = tokenizer(payload,return_tensors="pt").input_ids.to(model.device)
print(f"input payload: \n \n{payload}")
logits = model.generate(input_ids, do_sample=False, num_beams=1, min_length=128, max_new_tokens=128)
print(f"prediction: \n \n {tokenizer.decode(logits[0].tolist()[len(input_ids[0]):])}")

payload="Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend but it"*2
print(f'Payload sequence length is: {len(tokenizer(payload)["input_ids"])}')

# generation arguments
generation_args = dict(
  do_sample=False,
  num_beams=1,
  min_length=128,
  max_new_tokens=128
)
# vanilla_results = measure_latency(model, tokenizer, payload, generation_args, model.device)

# print(f"Vanilla model: {vanilla_results[0]}")


with torch.inference_mode():
    lmd_results = measure_latency(model, tokenizer, payload, generation_args, model.device)
print(f"LMDeploy model: {lmd_results[0]}")

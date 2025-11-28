import re
from typing import List, Iterator
from functools import lru_cache
from transformers import AutoTokenizer



def parse_sse_stream(content: str) -> list:
    """
    Parse SSE (Server-Sent Events) stream content into a list of events.
    Each event is either a JSON string or "[DONE]".
    """
    lines = content.strip().split("\n")
    events = []
    for line in lines:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]  # remove "data: "
            if data.strip() == "[DONE]":
                events.append("[DONE]")
            else:
                events.append(data)
    return events



@lru_cache(maxsize=4)
def _load_tokenizer_cached(model_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from '{model_path}': {e}")


def encode_text(model_path: str, text: str, add_bos: bool = True) -> List[int]:
    """
    Encode text to token IDs using the model's tokenizer.
    Automatically handles BOS for Llama-3, Qwen, etc.
    """
    tokenizer = _load_tokenizer_cached(model_path)
    encoded = tokenizer.encode(text, add_special_tokens=False)

    bos_token_id = getattr(tokenizer, 'bos_token_id', None)
    if add_bos and bos_token_id is not None:
        if not encoded or encoded[0] != bos_token_id:
            encoded = [bos_token_id] + encoded

    return encoded
from functools import lru_cache
from typing import List

from transformers import AutoTokenizer


def parse_sse_stream(content: str) -> list:
    """Parse SSE (Server-Sent Events) stream content into a list of events.

    Each event is either a JSON string or "[DONE]".
    """
    lines = content.strip().split('\n')
    events = []
    for line in lines:
        line = line.strip()
        if line.startswith('data: '):
            data = line[6:]  # remove "data: "
            if data.strip() == '[DONE]':
                events.append('[DONE]')
            else:
                events.append(data)
    return events


@lru_cache(maxsize=4)
def _load_tokenizer_cached(model_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if tokenizer.eos_token is None:
            tokenizer.eos_token = '<|end_of_text|>'

        return tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from '{model_path}': {e}")


def encode_text(model_path: str, text: str, add_bos: bool = True) -> List[int]:
    tokenizer = _load_tokenizer_cached(model_path)

    encoded = tokenizer.encode(text, add_special_tokens=False)

    if add_bos:
        bos_token_id = getattr(tokenizer, 'bos_token_id', None)
        if bos_token_id is not None:
            if not encoded or encoded[0] != bos_token_id:
                encoded = [bos_token_id] + encoded

    return encoded

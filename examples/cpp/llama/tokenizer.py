import os.path as osp
from typing import List

import fire


class Tokenizer:

    def __init__(self, model_file: str):
        if model_file.endswith('.model'):
            from sentencepiece import SentencePieceProcessor
            self.model = SentencePieceProcessor(model_file=model_file)
            self.vocab_size = self.model.vocab_size()
            self.start_id = self.model.bos_id()
            self.end_id = self.model.eos_id()
            self.pad_id = self.model.pad_id()
        else:
            from transformers import AutoTokenizer
            self.model = AutoTokenizer.from_pretrained(model_file,
                                                       trust_remote_code=True)
            self.vocab_size = self.model.vocab_size
            self.start_id = self.model.bos_token_id
            self.end_id = self.model.eos_token_id
            self.pad_id = self.model.pad_token_id

    def encode(self, s: str):
        if hasattr(self.model, 'Encode'):
            return self.model.Encode(s, add_bos=True)
        else:
            return self.model.encode(s, add_special_tokens=True)

    def decode(self, t: List[int]):
        if hasattr(self.model, 'Decode'):
            return self.model.Decode(t)
        else:
            return self.model.decode(t)


def main(model_file: str = '/data/llama/model/tokenizer.model',
         encode_file: str = None,
         decode_file: str = None):
    tokenizer = Tokenizer(model_file)
    if encode_file:
        with open(encode_file, 'r') as f:
            xs = tokenizer.encode(f.read())
            xs = ','.join(map(str, xs))
            print(xs)

        output_dir = osp.dirname(osp.abspath(__file__))
        with open(osp.join(output_dir, 'start_ids.csv'), 'w') as f:
            f.write(xs)

    elif decode_file:
        with open(decode_file, 'r') as f:
            token_ids = f.read()
            token_ids = token_ids.splitlines()
            for _token_ids in token_ids:
                _token_ids = _token_ids.split(',')
                _token_ids = [int(token_id) for token_id in _token_ids]
                ys = tokenizer.decode(_token_ids)
                print(ys)
    else:
        first = True
        while True:
            try:
                s = input()
            except EOFError:
                break
            if not first:
                print('---------------------------------------------')
            first = False
            try:
                xs = map(int, s.strip().split(' '))
                s = tokenizer.decode(list(xs))
                print(s)
            except ValueError:
                xs = tokenizer.encode(s)
                print(' '.join(map(str, xs)))


if __name__ == '__main__':
    fire.Fire(main)

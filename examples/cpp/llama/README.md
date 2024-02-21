How to generate start_ids.csv

```bash
# update `model_file` path and `encode_line` content according to the actual situation
python3 tokenizer.py --model_file /workdir/llama2_13b_chat/tokenizer.model --encode_line 'LMDeploy is a toolkit for compressing, deploying, and serving LLMs.'
# refer to tokenizer.py for more usage scenarios
```

# Usage

```bash
# https://huggingface.co/FasterDecoding/medusa-vicuna-13b-v1.3

# fp16 tp1
# default medusa pt path: /workdir/medusa-vicuna-13b-v1.3/medusa_lm_head.pt
# default medusa output path: /workdir/medusa_output/fp16/tp1
# default tp: 1
# default medusa weight type: fp16
python3 medusa_converter.py

# fp16 tp1
python3 medusa_converter.py --medusa_pt_path=/workdir/medusa-vicuna-13b-v1.3/medusa_lm_head.pt --medusa_output_path=/workdir/medusa_output/fp16/tp1 --tp=1 --medusa_weight_type=fp16

# fp16 tp2
python3 medusa_converter.py --medusa_pt_path=/workdir/medusa-vicuna-13b-v1.3/medusa_lm_head.pt --medusa_output_path=/workdir/medusa_output/fp16/tp2 --tp=2 --medusa_weight_type=fp16

# bf16 tp1
python3 medusa_converter.py --medusa_pt_path=/workdir/medusa-vicuna-13b-v1.3/medusa_lm_head.pt --medusa_output_path=/workdir/medusa_output/bf16/tp1 --tp=1 --medusa_weight_type=bf16

# bf16 tp2
python3 medusa_converter.py --medusa_pt_path=/workdir/medusa-vicuna-13b-v1.3/medusa_lm_head.pt --medusa_output_path=/workdir/medusa_output/bf16/tp2 --tp=2 --medusa_weight_type=bf16
```

## Support LLava-Interleave-Qwen-7B-hf

### AWQ

lmdeploy lite auto_awq --work_dir models/llava-interleave-qwen-7b-hf/awq models/llava-interleave-qwen-7b-hf
lmdeploy serve api_server models/llava-interleave-qwen-7b-hf/awq --model-format awq 


### Offline

python3 offline_vl.py models/llava-interleave-qwen-7b-hf

python3 offline_vl.py models/llava-interleave-qwen-7b-hf/awq --model-format awq

```text
Response(text="The image is a photograph capturing a moment between a person and a dog on a sandy beach. The person is seated on the sand, wearing a plaid shirt and pants, with their legs crossed. They are holding a small object in their hand, which appears to be a toy or a small treat, and are extending their hand towards the dog. The dog, which is standing on the sand, has its front paws raised towards the person's hand, suggesting an interaction or a gesture of play or gratitude. The dog is wearing a colorful harness with a pattern that includes blue, red, and yellow colors. The background features a calm sea with gentle waves lapping at the shore, and the sky is clear with a soft gradient from light to darker blue, indicating either sunrise or sunset. The lighting in the photograph is warm, contributing to the serene atmosphere of the scene. There are no visible texts or brands in the image.", generate_token_len=187, input_token_len=753, session_id=0, finish_reason='stop', token_ids=[785, 2168, 374, 264, 10300, 39780, 264, 4445, 1948, 264, 1697, 323, 264, 5562, 389, 264, 67439, 11321, 13, 576, 1697, 374, 46313, 389, 279, 9278, 11, 12233, 264, 625, 3779, 15478, 323, 24549, 11, 448, 862, 14201, 27031, 13, 2379, 525, 9963, 264, 2613, 1633, 304, 862, 1424, 11, 892, 7952, 311, 387, 264, 21357, 476, 264, 2613, 4228, 11, 323, 525, 32359, 862, 1424, 6974, 279, 5562, 13, 576, 5562, 11, 892, 374, 11259, 389, 279, 9278, 11, 702, 1181, 4065, 281, 8635, 9226, 6974, 279, 1697, 594, 1424, 11, 22561, 458, 16230, 476, 264, 30157, 315, 1486, 476, 45035, 13, 576, 5562, 374, 12233, 264, 33866, 32408, 448, 264, 5383, 429, 5646, 6303, 11, 2518, 11, 323, 13753, 7987, 13, 576, 4004, 4419, 264, 19300, 9396, 448, 21700, 16876, 326, 3629, 518, 279, 30184, 11, 323, 279, 12884, 374, 2797, 448, 264, 8413, 20169, 504, 3100, 311, 39030, 6303, 11, 18860, 2987, 63819, 476, 42984, 13, 576, 17716, 304, 279, 10300, 374, 8205, 11, 28720, 311, 279, 94763, 16566, 315, 279, 6109, 13, 2619, 525, 902, 9434, 21984, 476, 15721, 304, 279, 2168, 13], logprobs=None, index=0)
prompt:How many people in the image?
Response(text='There is one person in the image.', generate_token_len=8, input_token_len=756, session_id=1, finish_reason='stop', token_ids=[3862, 374, 825, 1697, 304, 279, 2168, 13], logprobs=None, index=0)
```
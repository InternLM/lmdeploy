import torch
import random
import lmdeploy
from lmdeploy import PytorchEngineConfig

if __name__ == "__main__":
    torch.manual_seed(10)
    random.seed(10)
    pipe = lmdeploy.pipeline("/data/models/Qwen-14B-Chat",
                            backend_config = PytorchEngineConfig(tp=1,
                                                                 block_size=16,
                                                                 device_type='maca',
                                                                 cache_max_entry_count=0.4))
    # warm up
    response = pipe("How are you?", do_preprocess=True)
    # import pdb; pdb.set_trace()

    print(response.text)

    question = ["How are you?"]
    question = ["Please introduce Shanghai."]
    question = ["What functions do you have?"]
    question = ["How are you?", "How are you?"]
    question = ["How are you?", "Please introduce Shanghai."]
    response = pipe(question, do_preprocess=True)
    for idx, r in enumerate(response):
        print(f"Q: {question[idx]}")
        print(f"A: {r.text}")
        print()

    # test multi session
    sess = pipe.chat("I am living in shanghai!")
    print("session 1:", sess)
    sess = pipe.chat("please introduce it.", session=sess)
    print("session 2: ", sess)

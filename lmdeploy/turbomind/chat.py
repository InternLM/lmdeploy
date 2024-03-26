# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

from lmdeploy.messages import EngineGenerationConfig, TurbomindEngineConfig
from lmdeploy.model import MODELS, ChatTemplateConfig
from lmdeploy.serve.async_engine import deduce_a_name
from lmdeploy.tokenizer import DetokenizeState
from lmdeploy.utils import _stop_words

os.environ['TM_LOG_LEVEL'] = 'ERROR'


def input_prompt(model_name):
    """Input a prompt in the consolo interface."""
    if model_name == 'codellama':
        print('\nenter !! to end the input >>>\n', end='')
        sentinel = '!!'
    else:
        print('\ndouble enter to end input >>> ', end='')
        sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def main(model_path: str,
         model_name: str = None,
         session_id: int = 1,
         top_k: float = 40,
         top_p: float = 0.8,
         temperature: float = 0.8,
         repetition_penalty: float = 1.0,
         cap: str = 'chat',
         tp: int = 1,
         max_batch_size: int = 128,
         model_format: str = None,
         quant_policy: int = 0,
         cache_max_entry_count: float = 0.8,
         cache_block_seq_len: int = 64,
         rope_scaling_factor: float = 0.0,
         session_len: int = None,
         stream_output: bool = True,
         request_output_len: int = 1024,
         chat_template: str = None,
         **kwargs):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the path of the deployed model
        model_name (str): the name of deployed model
        session_id (int): the identical id of a session
        top_k (int): sampling top k.
        top_p (int): sampling top p.
        temperature (float): sampling temperature.
        repetition_penalty (float): parameter to penalize repetition
        cap (str): the capability of a model. For example, codellama has the ability among ['completion', 'infilling', 'chat', 'python']
        tp (int): GPU number used in tensor parallelism
        max_batch_size (int): max batch size
        model_format (str): the layout of the deployed model. It can be one of the following values [hf, llama, awq]
        quant_policy (int): default to 0. When k/v is quantized into 8 bit, set it to 4
        cache_max_entry_count (float): the percentage of gpu memory occupied by the k/v cache.
        cache_block_seq_len (int): the length of the token sequence in a k/v block, default to 64
        rope_scaling_factor (float): scaling factor used for dynamic ntk, default to 0. TurboMind follows the implementation of transformer LlamaAttention
        session_len (int): the length input output tokens
        stream_output (bool): indicator for streaming output or not
        request_output_len (int): output token nums
        chat_template (str): user defined chat template
        kwargs (dict): unused args
    """ # noqa: E 501
    print('unused kwargs', kwargs, sep='')

    # chat template
    if chat_template is not None:
        chat_template_config = ChatTemplateConfig.from_json(chat_template)
        model = chat_template_config.chat_template
    else:
        model_name = deduce_a_name(model_path, model_name, None, None)
        model = MODELS.get(model_name)(capability=cap)

    # engine
    if session_len is None:
        session_len = model.session_len

    engine_cfg = TurbomindEngineConfig(
        max_batch_size=max_batch_size,
        model_name=model_name,
        model_format=model_format,
        session_len=session_len,
        cache_max_entry_count=cache_max_entry_count,
        cache_block_seq_len=cache_block_seq_len,
        quant_policy=quant_policy,
        rope_scaling_factor=rope_scaling_factor,
        tp=tp)
    print('engine_cfg:\n', engine_cfg, sep='', flush=True)

    from lmdeploy import turbomind as tm
    tm_model = tm.TurboMind.from_pretrained(model_path,
                                            engine_config=engine_cfg)
    generator = tm_model.create_instance()

    # generateion config
    tokenizer = tm_model.tokenizer
    stop_words = _stop_words(model.stop_words, tokenizer)
    if stop_words is not None:
        stop_words = stop_words[0][0].tolist()

    gen_config = EngineGenerationConfig(max_new_tokens=request_output_len,
                                        top_k=top_k,
                                        top_p=top_p,
                                        temperature=temperature,
                                        repetition_penalty=repetition_penalty,
                                        stop_words=stop_words)

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)
    while True:
        prompt = input_prompt(model_name)
        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            generator.end(session_id)
            nth_round = 1
            step = 0
            seed = random.getrandbits(64)
        else:
            prompt = model.get_prompt(prompt, nth_round == 1)
            print(f'{prompt} ', end='', flush=True)
            input_ids = tokenizer.encode(prompt, nth_round == 1)
            gen_config.random_seed = seed

            if model.capability == 'chat':
                sequence_start = (nth_round == 1)
                sequence_end = False
                step = step
            else:
                sequence_start = True
                sequence_end = True
                step = 0

            if step + len(
                    input_ids) + request_output_len >= tm_model.session_len:
                print('WARNING: exceed session max length.'
                      ' Please end the session.')
                continue

            state = DetokenizeState()
            for outputs in generator.stream_infer(
                    session_id=session_id,
                    input_ids=[input_ids],
                    gen_config=gen_config,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    step=step,
                    stream_output=stream_output):
                _, res, tokens = outputs
                # decode res
                response, state = tokenizer.detokenize_incrementally(
                    res, state=state)
                print(response, end='', flush=True)

            # update step
            step += len(input_ids) + tokens
            print()

            nth_round += 1


if __name__ == '__main__':
    import fire

    fire.Fire(main)

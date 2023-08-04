# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import random
import threading
from functools import partial
from typing import Sequence

import fire
import gradio as gr

from lmdeploy import turbomind as tm
from lmdeploy.model import MODELS
from lmdeploy.serve.gradio.css import CSS
from lmdeploy.serve.turbomind.chatbot import Chatbot
from lmdeploy.turbomind.chat import valid_str
from lmdeploy.turbomind.tokenizer import Tokenizer

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
    font=[gr.themes.GoogleFont('Inconsolata'), 'Arial', 'sans-serif'])


def chat_stream(state_chatbot: Sequence, llama_chatbot: Chatbot,
                request: gr.Request):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        state_chatbot (Sequence): the chatting history
        llama_chatbot (Chatbot): the instance of a chatbot
        request (gr.Request): the request from a user
        model_name (str): the name of deployed model
    """
    instruction = state_chatbot[-1][0]
    session_id = threading.current_thread().ident
    if request is not None:
        session_id = int(request.kwargs['client']['host'].replace('.', ''))

    bot_response = llama_chatbot.stream_infer(
        session_id, instruction, f'{session_id}-{len(state_chatbot)}')

    for status, tokens, _ in bot_response:
        state_chatbot[-1] = (state_chatbot[-1][0], tokens)
        yield (state_chatbot, state_chatbot, '')

    return (state_chatbot, state_chatbot, '')


def reset_all_func(instruction_txtbox: gr.Textbox, state_chatbot: gr.State,
                   llama_chatbot: gr.State, triton_server_addr: str,
                   model_name: str):
    """reset the session."""
    state_chatbot = []
    log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
    llama_chatbot = Chatbot(triton_server_addr,
                            model_name,
                            log_level=log_level,
                            display=True)

    return (
        llama_chatbot,
        state_chatbot,
        state_chatbot,
        gr.Textbox.update(value=''),
    )


def cancel_func(
    instruction_txtbox: gr.Textbox,
    state_chatbot: gr.State,
    llama_chatbot: gr.State,
):
    """cancel the session."""
    session_id = llama_chatbot._session.session_id
    llama_chatbot.cancel(session_id)

    return (
        llama_chatbot,
        state_chatbot,
    )


def add_instruction(instruction, state_chatbot):
    state_chatbot = state_chatbot + [(instruction, None)]
    return ('', state_chatbot)


def run_server(triton_server_addr: str,
               server_name: str = 'localhost',
               server_port: int = 6006):
    """chat with AI assistant through web ui.

    Args:
        triton_server_addr (str): the communication address of inference server
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
    """
    with gr.Blocks(css=CSS, theme=THEME) as demo:
        log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
        llama_chatbot = gr.State(
            Chatbot(triton_server_addr, log_level=log_level, display=True))
        state_chatbot = gr.State([])
        model_name = llama_chatbot.value.model_name
        reset_all = partial(reset_all_func,
                            model_name=model_name,
                            triton_server_addr=triton_server_addr)

        with gr.Column(elem_id='container'):
            gr.Markdown('## LMDeploy Playground')

            chatbot = gr.Chatbot(elem_id='chatbot', label=model_name)
            instruction_txtbox = gr.Textbox(
                placeholder='Please input the instruction',
                label='Instruction')
            with gr.Row():
                cancel_btn = gr.Button(value='Cancel')
                reset_btn = gr.Button(value='Reset')

        send_event = instruction_txtbox.submit(
            add_instruction, [instruction_txtbox, state_chatbot],
            [instruction_txtbox, state_chatbot]).then(
                chat_stream, [state_chatbot, llama_chatbot],
                [state_chatbot, chatbot])

        cancel_btn.click(cancel_func,
                         [instruction_txtbox, state_chatbot, llama_chatbot],
                         [llama_chatbot, chatbot],
                         cancels=[send_event])

        reset_btn.click(
            reset_all, [instruction_txtbox, state_chatbot, llama_chatbot],
            [llama_chatbot, state_chatbot, chatbot, instruction_txtbox],
            cancels=[send_event])

    print(f'server is gonna mount on: http://{server_name}:{server_port}')
    demo.queue(concurrency_count=4, max_size=100, api_open=True).launch(
        max_threads=10,
        share=True,
        server_port=server_port,
        server_name=server_name,
    )


# a IO interface mananing global variables
class InterFace:
    tokenizer_model_path = None
    tokenizer = None
    tm_model = None
    request2instance = None
    model_name = None
    model = None


def chat_stream_local(
    instruction: str,
    state_chatbot: Sequence,
    step: gr.State,
    nth_round: gr.State,
    request: gr.Request,
):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        state_chatbot (Sequence): the chatting history
        step (gr.State): chat history length
        nth_round (gr.State): round num
        request (gr.Request): the request from a user
    """
    session_id = threading.current_thread().ident
    if request is not None:
        session_id = int(request.kwargs['client']['host'].replace('.', ''))
    if str(session_id) not in InterFace.request2instance:
        InterFace.request2instance[str(
            session_id)] = InterFace.tm_model.create_instance()
    llama_chatbot = InterFace.request2instance[str(session_id)]
    seed = random.getrandbits(64)
    bot_summarized_response = ''
    state_chatbot = state_chatbot + [(instruction, None)]
    instruction = InterFace.model.get_prompt(instruction, nth_round == 1)
    if step >= InterFace.tm_model.session_len:
        raise gr.Error('WARNING: exceed session max length.'
                       ' Please end the session.')
    input_ids = InterFace.tokenizer.encode(instruction)
    bot_response = llama_chatbot.stream_infer(
        session_id, [input_ids],
        stream_output=True,
        request_output_len=512,
        sequence_start=(nth_round == 1),
        sequence_end=False,
        step=step,
        stop=False,
        top_k=40,
        top_p=0.8,
        temperature=0.8,
        repetition_penalty=1.0,
        ignore_eos=False,
        random_seed=seed if nth_round == 1 else None)

    yield (state_chatbot, state_chatbot, step, nth_round,
           f'{bot_summarized_response}'.strip())

    response_size = 0
    for outputs in bot_response:
        res, tokens = outputs[0]
        # decode res
        response = InterFace.tokenizer.decode(res)[response_size:]
        response = valid_str(response)
        response_size += len(response)
        if state_chatbot[-1][-1] is None:
            state_chatbot[-1] = (state_chatbot[-1][0], response)
        else:
            state_chatbot[-1] = (state_chatbot[-1][0],
                                 state_chatbot[-1][1] + response
                                 )  # piece by piece
        yield (state_chatbot, state_chatbot, step, nth_round,
               f'{bot_summarized_response}'.strip())

    step += len(input_ids) + tokens
    nth_round += 1
    yield (state_chatbot, state_chatbot, step, nth_round,
           f'{bot_summarized_response}'.strip())


def reset_local_func(instruction_txtbox: gr.Textbox, state_chatbot: gr.State,
                     step: gr.State, nth_round: gr.State, request: gr.Request):
    """reset the session.

    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        step (gr.State): chat history length
        nth_round (gr.State): round num
        request (gr.Request): the request from a user
    """
    state_chatbot = []
    step = 0
    nth_round = 1

    session_id = threading.current_thread().ident
    if request is not None:
        session_id = int(request.kwargs['client']['host'].replace('.', ''))
    InterFace.request2instance[str(
        session_id)] = InterFace.tm_model.create_instance()

    return (
        state_chatbot,
        state_chatbot,
        step,
        nth_round,
        gr.Textbox.update(value=''),
    )


def run_local(model_path: str,
              server_name: str = 'localhost',
              server_port: int = 6006):
    """chat with AI assistant through web ui.

    Args:
        model_path (str): the path of the deployed model
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
    """
    InterFace.tokenizer_model_path = osp.join(model_path, 'triton_models',
                                              'tokenizer')
    InterFace.tokenizer = Tokenizer(InterFace.tokenizer_model_path)
    InterFace.tm_model = tm.TurboMind(model_path,
                                      eos_id=InterFace.tokenizer.eos_token_id)
    InterFace.request2instance = dict()
    InterFace.model_name = InterFace.tm_model.model_name
    InterFace.model = MODELS.get(InterFace.model_name)()

    with gr.Blocks(css=CSS, theme=THEME) as demo:
        state_chatbot = gr.State([])
        nth_round = gr.State(1)
        step = gr.State(0)

        with gr.Column(elem_id='container'):
            gr.Markdown('## LMDeploy Playground')

            chatbot = gr.Chatbot(elem_id='chatbot', label=InterFace.model_name)
            instruction_txtbox = gr.Textbox(
                placeholder='Please input the instruction',
                label='Instruction')
            with gr.Row():
                gr.Button(value='Cancel')  # noqa: E501
                reset_btn = gr.Button(value='Reset')

        send_event = instruction_txtbox.submit(
            chat_stream_local,
            [instruction_txtbox, state_chatbot, step, nth_round],
            [state_chatbot, chatbot, step, nth_round])
        instruction_txtbox.submit(
            lambda: gr.Textbox.update(value=''),
            [],
            [instruction_txtbox],
        )

        reset_btn.click(
            reset_local_func,
            [instruction_txtbox, state_chatbot, step, nth_round],
            [state_chatbot, chatbot, step, nth_round, instruction_txtbox],
            cancels=[send_event])

    print(f'server is gonna mount on: http://{server_name}:{server_port}')
    demo.queue(concurrency_count=4, max_size=100, api_open=True).launch(
        max_threads=10,
        share=True,
        server_port=server_port,
        server_name=server_name,
    )


def run(model_path_or_server: str,
        server_name: str = 'localhost',
        server_port: int = 6006):
    """chat with AI assistant through web ui.

    Args:
        model_path_or_server (str): the path of the deployed model or the
            tritonserver URL. The former is for directly running service with
            gradio. The latter is for running with tritonserver
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
    """
    if ':' in model_path_or_server:
        run_server(model_path_or_server, server_name, server_port)
    else:
        run_local(model_path_or_server, server_name, server_port)


if __name__ == '__main__':
    fire.Fire(run)

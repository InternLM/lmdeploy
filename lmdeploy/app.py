# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import threading
from functools import partial
from typing import Sequence

import fire
import gradio as gr

from lmdeploy.serve.turbomind.chatbot import Chatbot
from lmdeploy import turbomind as tm
from lmdeploy.model import MODELS
from lmdeploy.turbomind.chat import valid_str
from lmdeploy.turbomind.tokenizer import Tokenizer

CSS = """
#container {
    width: 95%;
    margin-left: auto;
    margin-right: auto;
}

#chatbot {
    height: 500px;
    overflow: auto;
}

.chat_wrap_space {
    margin-left: 0.5em
}
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
    font=[gr.themes.GoogleFont('Inconsolata'), 'Arial', 'sans-serif'])


def chat_stream(instruction: str,
                state_chatbot: Sequence,
                llama_chatbot: Chatbot,
                model_name: str = None):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        state_chatbot (Sequence): the chatting history
        llama_chatbot (Chatbot): the instance of a chatbot
        model_name (str): the name of deployed model
    """
    bot_summarized_response = ''
    model_type = 'turbomind'
    state_chatbot = state_chatbot + [(instruction, None)]
    session_id = threading.current_thread().ident
    bot_response = llama_chatbot.stream_infer(
        session_id, instruction, f'{session_id}-{len(state_chatbot)}')

    yield (state_chatbot, state_chatbot, f'{bot_summarized_response}'.strip())

    for status, tokens, _ in bot_response:
        if state_chatbot[-1][-1] is None or model_type != 'fairscale':
            state_chatbot[-1] = (state_chatbot[-1][0], tokens)
        else:
            state_chatbot[-1] = (state_chatbot[-1][0],
                                 state_chatbot[-1][1] + tokens
                                 )  # piece by piece
        yield (state_chatbot, state_chatbot,
               f'{bot_summarized_response}'.strip())

    yield (state_chatbot, state_chatbot, f'{bot_summarized_response}'.strip())


def reset_all_func(instruction_txtbox: gr.Textbox, state_chatbot: gr.State,
                   llama_chatbot: gr.State, triton_server_addr: str,
                   model_name: str):
    """reset the session."""
    state_chatbot = []
    log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
    llama_chatbot = Chatbot(
        triton_server_addr, model_name, log_level=log_level, display=True)

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


def run(triton_server_addr: str,
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
        _chatbot = Chatbot(
            triton_server_addr, log_level=log_level, display=True)
        model_name = _chatbot.model_name
        chat_interface = partial(chat_stream, model_name=model_name)
        reset_all = partial(
            reset_all_func,
            model_name=model_name,
            triton_server_addr=triton_server_addr)
        llama_chatbot = gr.State(_chatbot)
        state_chatbot = gr.State([])

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
            chat_interface,
            [instruction_txtbox, state_chatbot, llama_chatbot],
            [state_chatbot, chatbot],
            batch=False,
            max_batch_size=1,
        )
        instruction_txtbox.submit(
            lambda: gr.Textbox.update(value=''),
            [],
            [instruction_txtbox],
        )

        cancel_btn.click(
            cancel_func, [instruction_txtbox, state_chatbot, llama_chatbot],
            [llama_chatbot, chatbot],
            cancels=[send_event])

        reset_btn.click(
            reset_all, [instruction_txtbox, state_chatbot, llama_chatbot],
            [llama_chatbot, state_chatbot, chatbot, instruction_txtbox],
            cancels=[send_event])

    demo.queue(
        concurrency_count=4, max_size=100, api_open=True).launch(
            max_threads=10,
            share=True,
            server_port=server_port,
            server_name=server_name,
        )


def chat_stream_local(instruction: str, state_chatbot: Sequence,
                      step: gr.State, nth_round: gr.State, model, tm_model,
                      tokenizer, llama_chatbot):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        state_chatbot (Sequence): the chatting history
        llama_chatbot (Chatbot): the instance of a chatbot
        model_name (str): the name of deployed model
    """
    bot_summarized_response = ''
    state_chatbot = state_chatbot + [(instruction, None)]
    instruction = model.get_prompt(instruction, nth_round == 1)
    if step >= tm_model.session_len:
        print('WARNING: exceed session max length.'
              ' Please end the session.')
    print('instruction', instruction)
    input_ids = tokenizer.encode(instruction)
    session_id = threading.current_thread().ident
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
        random_seed=None)

    yield (state_chatbot, state_chatbot, step, nth_round,
           f'{bot_summarized_response}'.strip())

    response_size = 0
    for outputs in bot_response:
        res, tokens = outputs[0]
        # decode res
        response = tokenizer.decode(res)[response_size:]
        response = valid_str(response)
        print(f'{response}', end='', flush=True)
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
                     step: gr.State, nth_round: gr.State):
    """reset the session."""
    state_chatbot = []
    log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
    # llama_chatbot = tm_model.create_instance()
    step = 0
    nth_round = 1

    return (
        state_chatbot,
        state_chatbot,
        step,
        nth_round,
        gr.Textbox.update(value=''),
    )


def run_local_model(model_path: str,
                    server_name: str = 'localhost',
                    server_port: int = 6006):
    """chat with AI assistant through web ui.

    Args:
        triton_server_addr (str): the communication address of inference server
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
    """
    tokenizer_model_path = osp.join(model_path, 'triton_models', 'tokenizer')
    tokenizer = Tokenizer(tokenizer_model_path)
    tm_model = tm.TurboMind(model_path, eos_id=tokenizer.eos_token_id)
    llama_chatbot = tm_model.create_instance()
    model_name = tm_model.model_name
    model = MODELS.get(model_name)()

    with gr.Blocks(css=CSS, theme=THEME) as demo:
        log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
        model_name = tm_model.model_name
        state_chatbot = gr.State([])
        nth_round = gr.State(1)
        step = gr.State(0)
        chat_stream_interface = partial(chat_stream_local,
            model=model,
            tm_model=tm_model,
            tokenizer=tokenizer,
            llama_chatbot=llama_chatbot)

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
            chat_stream_interface,
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

    demo.queue(
        concurrency_count=4, max_size=100, api_open=True).launch(
            max_threads=10,
            share=True,
            server_port=server_port,
            server_name=server_name,
        )


if __name__ == '__main__':
    fire.Fire(run_local_model)

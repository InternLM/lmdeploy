# Copyright (c) OpenMMLab. All rights reserved.
import os
import threading
from functools import partial
from typing import Sequence

import fire
import gradio as gr

from lmdeploy.serve.turbomind.chatbot import Chatbot

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


def run(triton_server_addr: str,
        model_name: str,
        server_name: str = 'localhost',
        server_port: int = 6006):
    """chat with AI assistant through web ui.

    Args:
        triton_server_addr (str): the communication address of inference server
        model_name (str): the name of the deployed model
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
    """
    with gr.Blocks(css=CSS, theme=THEME) as demo:
        chat_interface = partial(chat_stream, model_name=model_name)
        reset_all = partial(reset_all_func,
                            model_name=model_name,
                            triton_server_addr=triton_server_addr)
        log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
        llama_chatbot = gr.State(
            Chatbot(triton_server_addr,
                    model_name,
                    log_level=log_level,
                    display=True))
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

        cancel_btn.click(cancel_func,
                         [instruction_txtbox, state_chatbot, llama_chatbot],
                         [llama_chatbot, chatbot],
                         cancels=[send_event])

        reset_btn.click(
            reset_all, [instruction_txtbox, state_chatbot, llama_chatbot],
            [llama_chatbot, state_chatbot, chatbot, instruction_txtbox],
            cancels=[send_event])

    demo.queue(concurrency_count=4, max_size=100, api_open=True).launch(
        max_threads=10,
        share=True,
        server_port=server_port,
        server_name=server_name,
    )


if __name__ == '__main__':
    fire.Fire(run)

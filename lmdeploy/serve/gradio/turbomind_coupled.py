# Copyright (c) OpenMMLab. All rights reserved.
from threading import Lock
from typing import Sequence

import gradio as gr

from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.gradio.constants import CSS, THEME, disable_btn, enable_btn


class InterFace:
    async_engine: AsyncEngine = None
    global_session_id: int = 0
    lock = Lock()


async def chat_stream_local(
    instruction: str,
    state_chatbot: Sequence,
    cancel_btn: gr.Button,
    reset_btn: gr.Button,
    session_id: int,
):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        state_chatbot (Sequence): the chatting history
        cancel_btn (gr.Button): the cancel button
        reset_btn (gr.Button): the reset button
        session_id (int): the session id
    """
    state_chatbot = state_chatbot + [(instruction, None)]

    yield (state_chatbot, state_chatbot, disable_btn, enable_btn)

    async for outputs in InterFace.async_engine.generate(
            instruction,
            session_id,
            stream_response=True,
            sequence_start=(len(state_chatbot) == 1),
            sequence_end=False):
        response = outputs.response
        if outputs.finish_reason == 'length':
            gr.Warning('WARNING: exceed session max length.'
                       ' Please restart the session by reset button.')
        if outputs.generate_token_len < 0:
            gr.Warning('WARNING: running on the old session.'
                       ' Please restart the session by reset button.')
        if state_chatbot[-1][-1] is None:
            state_chatbot[-1] = (state_chatbot[-1][0], response)
        else:
            state_chatbot[-1] = (state_chatbot[-1][0],
                                 state_chatbot[-1][1] + response
                                 )  # piece by piece
        yield (state_chatbot, state_chatbot, enable_btn, disable_btn)

    yield (state_chatbot, state_chatbot, disable_btn, enable_btn)


async def reset_local_func(instruction_txtbox: gr.Textbox,
                           state_chatbot: Sequence, session_id: int):
    """reset the session.

    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        session_id (int): the session id
    """
    state_chatbot = []
    # end the session
    async for out in InterFace.async_engine.generate('',
                                                     session_id,
                                                     request_output_len=1,
                                                     stream_response=True,
                                                     sequence_start=False,
                                                     sequence_end=True):
        pass
    return (state_chatbot, state_chatbot, gr.Textbox.update(value=''))


async def cancel_local_func(state_chatbot: Sequence, cancel_btn: gr.Button,
                            reset_btn: gr.Button, session_id: int):
    """stop the session.

    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        cancel_btn (gr.Button): the cancel button
        reset_btn (gr.Button): the reset button
        session_id (int): the session id
    """
    yield (state_chatbot, disable_btn, enable_btn)
    async for out in InterFace.async_engine.generate('',
                                                     session_id,
                                                     request_output_len=0,
                                                     stream_response=True,
                                                     sequence_start=False,
                                                     sequence_end=False,
                                                     stop=True):
        pass
    messages = []
    for qa in state_chatbot:
        messages.append(dict(role='user', content=qa[0]))
        if qa[1] is not None:
            messages.append(dict(role='assistant', content=qa[1]))
    async for out in InterFace.async_engine.generate(messages,
                                                     session_id,
                                                     request_output_len=0,
                                                     stream_response=True,
                                                     sequence_start=True,
                                                     sequence_end=False):
        pass
    yield (state_chatbot, disable_btn, enable_btn)


def run_local(model_path: str,
              server_name: str = 'localhost',
              server_port: int = 6006,
              batch_size: int = 4,
              tp: int = 1,
              **kwargs):
    """chat with AI assistant through web ui.

    Args:
        model_path (str): the path of the deployed model
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
        batch_size (int): batch size for running Turbomind directly
        tp (int): tensor parallel for Turbomind
    """
    InterFace.async_engine = AsyncEngine(model_path=model_path,
                                         instance_num=batch_size,
                                         tp=tp,
                                         **kwargs)

    with gr.Blocks(css=CSS, theme=THEME) as demo:
        state_chatbot = gr.State([])
        state_session_id = gr.State(0)

        with gr.Column(elem_id='container'):
            gr.Markdown('## LMDeploy Playground')

            chatbot = gr.Chatbot(
                elem_id='chatbot',
                label=InterFace.async_engine.tm_model.model_name)
            instruction_txtbox = gr.Textbox(
                placeholder='Please input the instruction',
                label='Instruction')
            with gr.Row():
                cancel_btn = gr.Button(value='Cancel', interactive=False)
                reset_btn = gr.Button(value='Reset')

        send_event = instruction_txtbox.submit(chat_stream_local, [
            instruction_txtbox, state_chatbot, cancel_btn, reset_btn,
            state_session_id
        ], [state_chatbot, chatbot, cancel_btn, reset_btn])
        instruction_txtbox.submit(
            lambda: gr.Textbox.update(value=''),
            [],
            [instruction_txtbox],
        )
        cancel_btn.click(
            cancel_local_func,
            [state_chatbot, cancel_btn, reset_btn, state_session_id],
            [state_chatbot, cancel_btn, reset_btn],
            cancels=[send_event])

        reset_btn.click(reset_local_func,
                        [instruction_txtbox, state_chatbot, state_session_id],
                        [state_chatbot, chatbot, instruction_txtbox],
                        cancels=[send_event])

        def init():
            with InterFace.lock:
                InterFace.global_session_id += 1
            new_session_id = InterFace.global_session_id
            return new_session_id

        demo.load(init, inputs=None, outputs=[state_session_id])

    print(f'server is gonna mount on: http://{server_name}:{server_port}')
    demo.queue(concurrency_count=batch_size, max_size=100,
               api_open=True).launch(
                   max_threads=10,
                   share=True,
                   server_port=server_port,
                   server_name=server_name,
               )


if __name__ == '__main__':
    import fire
    fire.Fire(run_local)

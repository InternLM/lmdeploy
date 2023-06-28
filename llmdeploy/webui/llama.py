import threading


def chat_stream(instruction,
                state_chatbot,
                llama_chatbot,
                model_name: str = None):
    bot_summarized_response = ''
    model_type = 'fastertransformer'
    state_chatbot = state_chatbot + [(instruction, None)]
    session_id = threading.current_thread().ident
    bot_response = llama_chatbot.stream_infer(
        session_id, instruction, f'{session_id}-{len(state_chatbot)}')

    yield (state_chatbot, state_chatbot,
           f'{bot_summarized_response}'.strip())

    for status, tokens, _ in bot_response:
        if state_chatbot[-1][-1] is None or model_type != 'fairscale':
            state_chatbot[-1] = (state_chatbot[-1][0], tokens)
        else:
            state_chatbot[-1] = (state_chatbot[-1][0],
                                 state_chatbot[-1][1] + tokens
                                 )  # piece by piece
        yield (state_chatbot, state_chatbot,
               f'{bot_summarized_response}'.strip())

    yield (state_chatbot, state_chatbot,
           f'{bot_summarized_response}'.strip())

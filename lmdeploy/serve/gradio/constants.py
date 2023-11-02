# Copyright (c) OpenMMLab. All rights reserved.

import gradio as gr

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

enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

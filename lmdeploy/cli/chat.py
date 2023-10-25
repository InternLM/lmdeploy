# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional


class SubCliChat(object):
    """Chat through terminal with pytorch or turbomind model."""

    def torch(self,
              model_path: str,
              tokenizer_path: Optional[str] = None,
              accel: Optional[str] = None,
              max_new_tokens: int = 128,
              temperature: float = 0.8,
              top_p: float = 0.95,
              seed: int = 0,
              use_fast_tokenizer: bool = True,
              max_alloc: int = 2048,
              max_session_len: int = None,
              log_file: Optional[str] = None,
              debug: bool = False,
              adapter: Optional[str] = None):
        """Chat with pytorch model through terminal.

        Args:
            model_path (str): Path to pytorch model.
            tokenizer_path (str): Path to tokenizer.
            accel (str): Model accelerator.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Temperature for sampling.
            top_p (float): Top p for sampling.
            seed (int): Random seed.
            use_fast_tokenizer (bool): Whether to use fast tokenizer.
                This argument is directly pass to transformer's
                ``AutoTokenizer.from_pretrained``.
                Generally, user should choose to use fast tokenizers.
                But if using fast raise some error, try to force using a slow one.
            max_alloc (int): Maximum memory to allocate (for deepspeed).
            max_session_len (int): Maximum number of tokens allowed for all chat sessions.
                This include both history and current session.
            log_file (str): Path to log file.
            debug (bool): Whether to enable debug mode.
            adapter (str): Force to use an adapter.
                Generally user should not use this argument because adapter is selected based
                on the type of model. Only when it is impossible, e.g. distinguishing llama 1/2
                based on `LlamaforCausalLM` class, this argument is required.
                Currently, only "llama1" is acceptable for llama1 models.
        """  # noqa: E501
        from lmdeploy.pytorch.chat import main as run_torch_model

        run_torch_model(model_path,
                        tokenizer_path=tokenizer_path,
                        accel=accel,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                        use_fast_tokenizer=use_fast_tokenizer,
                        max_alloc=max_alloc,
                        max_session_len=max_session_len,
                        log_file=log_file,
                        debug=debug,
                        adapter=adapter)

    def turbomind(self,
                  model_path,
                  session_id: int = 1,
                  cap: str = 'chat',
                  tp=1,
                  stream_output=True,
                  **kwargs):
        """Chat with turbomind model through terminal.

        Args:
            model_path (str): the path of the deployed model
            session_id (int): the identical id of a session
            cap (str): the capability of a model. For example, codellama has
                the ability among ['completion', 'infilling', 'chat', 'python']
            tp (int): GPU number used in tensor parallelism
            stream_output (bool): indicator for streaming output or not
            **kwarg (dict): other arguments for initializing model's chat
                template
        """
        from lmdeploy.turbomind.chat import main as run_turbomind_model

        run_turbomind_model(model_path,
                            session_id=session_id,
                            cap=cap,
                            tp=tp,
                            stream_output=stream_output,
                            **kwargs)

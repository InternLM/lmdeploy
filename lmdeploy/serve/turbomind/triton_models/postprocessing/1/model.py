# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from pathlib import Path
from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils


class Tokenizer:
    """Tokenize prompts or de-tokenize tokens into texts.

    Args:
        model_file (str): the path of the tokenizer model
    """

    def __init__(self, model_file: str):
        model_folder = osp.split(model_file)[0]
        tokenizer_config_file = osp.join(model_folder, 'tokenizer_config.json')
        use_hf_model = osp.exists(tokenizer_config_file)
        self.use_hf_model = use_hf_model
        if not self.use_hf_model:
            from sentencepiece import SentencePieceProcessor
            self.model = SentencePieceProcessor(model_file=model_file)
            self.vocab_size = self.model.vocab_size()
            self.start_id = self.model.bos_id()
            self.end_id = self.model.eos_id()
        else:
            from transformers import AutoTokenizer
            backend_tokenizer_file = osp.join(model_folder, 'tokenizer.json')
            if not osp.exists(backend_tokenizer_file):
                print('WARNING: Can not find tokenizer.json. '
                      'It may take long time to initialize the tokenizer.')
            self.model = AutoTokenizer.from_pretrained(model_folder,
                                                       trust_remote_code=True)
            self.vocab_size = self.model.vocab_size
            self.start_id = self.model.bos_token_id
            self.end_id = self.model.eos_token_id
            # save tokenizer.json to reuse
            if not osp.exists(backend_tokenizer_file) and \
                    hasattr(self.model, 'backend_tokenizer'):
                self.model.backend_tokenizer.save(backend_tokenizer_file)

    def encode(self, s: str):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        if not self.use_hf_model:
            add_bos = False
            add_eos = False
            if s.find('<BOS>') != -1:
                s = s.replace('<BOS>', '')
                add_bos = True
            if s == '<EOS>':
                s = ''
                add_eos = True
            return self.model.Encode(s, add_bos=add_bos, add_eos=add_eos)
        else:
            add_special_tokens = False
            if s.find('<BOS>') != -1:
                s = s.replace('<BOS>', '<s>')
            if s == '<EOS>':
                s = '</s>'
            if len(s) == 0:
                add_special_tokens = True
            return self.model.encode(s, add_special_tokens=add_special_tokens)

    def decode(self, t: List[int]):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
        Returns:
            str: text of decoding tokens
        """
        if not self.use_hf_model:
            return self.model.Decode(t)
        else:
            skip_special_tokens = False
            return self.model.decode(t,
                                     skip_special_tokens=skip_special_tokens)


class TritonPythonModel:
    """Your Python model must use the same class name.

    Every Python model that is created must have "TritonPythonModel" as the
    class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
          ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        self.model_config = model_config = json.loads(args['model_config'])

        # Parse model output configs
        output_config = pb_utils.get_output_config_by_name(
            model_config, 'OUTPUT')

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])

        cur_folder = Path(__file__).parent

        self.tokenizer = Tokenizer(
            osp.join(
                cur_folder, self.model_config['parameters']['tokenizer_path']
                ['string_value']))

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get input tensors
            tokens_batch = pb_utils.get_input_tensor_by_name(
                request, 'TOKENS_BATCH').as_numpy()
            sequence_length = pb_utils.get_input_tensor_by_name(
                request, 'sequence_length').as_numpy()

            # Postprocessing output data.
            outputs = self._postprocessing(tokens_batch.tolist(),
                                           sequence_length)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            output_tensor = pb_utils.Tensor(
                'OUTPUT',
                np.array(outputs).astype(self.output_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.

        Implementing `finalize` function is optional. This function allows the
        model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

    def _postprocessing(self, tokens_batch, sequence_length):
        """decode token ids into texts."""
        outputs = []
        for beam_tokens, beam_len in zip(tokens_batch, sequence_length):
            for tokens, _len in zip(beam_tokens, beam_len):
                output = self.tokenizer.decode(tokens[:_len])
                output = output.encode('utf8')
                outputs.append(output)
        return outputs

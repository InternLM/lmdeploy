# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.nn.utils.rnn import pad_sequence

# This tokenizer is `lmdeploy/turbomind/tokenizer.py`. When an LLM is served
# by triton inference server, it has to be converted first by running
# `python lmdeploy/serve/turbomind/deploy.py`. Then
# `lmdeploy/turbomind/tokenizer.py` will be copied to `tokenizer/tokenizer.py`
from .tokenizer.tokenizer import Tokenizer


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

        # Parse model output configs and convert Triton types to numpy types
        input_names = ['INPUT_ID', 'REQUEST_INPUT_LEN']
        for input_name in input_names:
            setattr(
                self,
                input_name.lower() + '_dtype',
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(
                        model_config, input_name)['data_type']))

        cur_folder = Path(__file__).parent
        self.tokenizer = Tokenizer(
            osp.join(
                cur_folder, self.model_config['parameters']['tokenizer_path']
                ['string_value']))
        self.start_id = self.tokenizer.bos_token_id
        self.end_id = self.tokenizer.eos_token_id

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
            query = pb_utils.get_input_tensor_by_name(request,
                                                      'QUERY').as_numpy()

            # Preprocessing input data.
            input_id, request_input_len = self._create_request(query)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            input_id_tensor = pb_utils.Tensor(
                'INPUT_ID',
                np.array(input_id).astype(self.input_id_dtype))
            request_input_len_tensor = pb_utils.Tensor(
                'REQUEST_INPUT_LEN',
                np.array(request_input_len).astype(
                    self.request_input_len_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[input_id_tensor, request_input_len_tensor])
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

    def _create_request(self, query):
        """Tokenize prompts and return the token ids and their length.

        Args:
            query (List[str]): a list of prompt
        Returns:
            tuple: token ids and their length
        """
        start_ids = []
        for s in query:
            _s = s[0].decode()
            if _s == '<BOS>':
                start_id = [self.start_id
                            ] if self.start_id is not None else [-1]
            elif _s == '<EOS>':
                start_id = [self.end_id] if self.end_id is not None else [-1]
            else:
                start_id = self.tokenizer.encode(_s)
            start_ids.append(torch.IntTensor(start_id))

        start_lengths = torch.IntTensor([[len(ids)] for ids in start_ids])
        start_ids = pad_sequence(start_ids,
                                 batch_first=True,
                                 padding_value=self.end_id)
        return start_ids, start_lengths

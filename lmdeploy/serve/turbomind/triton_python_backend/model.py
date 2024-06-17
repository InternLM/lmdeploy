# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import json
import os
import threading

import numpy as np
import triton_python_backend_utils as pb_utils

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline


class TritonPythonModel:

    def initialize(self, args):
        self.logger = pb_utils.Logger

        # parse model configs
        self.model_config = json.loads(args['model_config'])

        # make sure use decoupled mode
        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config)
        assert (using_decoupled
                ), 'LMDeploy model should be configured to decoupled mode'

        # parse parameters
        parameters = self.model_config['parameters']
        model_name = parameters['model_name']['string_value']
        tp = int(parameters['tp']['string_value'])

        # start lmdeploy engine
        model_path = os.path.join(args['model_repository'],
                                  args['model_version'], 'weights')
        engine_config = TurbomindEngineConfig(tp=tp)
        self.engine = pipeline(model_path=model_path,
                               model_name=model_name,
                               backend_config=engine_config)

        self.request_id = 0

        # create event loop to process requests asynchronously
        self.event_loop = asyncio.get_event_loop()
        self.engine_thread = threading.Thread(target=self._engine_loop)
        self.shutdown_event = asyncio.Event()
        self.engine_thread.start()

        self.logger.log_info('LMDeploy backend started')

    def _engine_loop(self):
        self.logger.log_info('Engine loop started')
        self.event_loop.run_until_complete(self._await_shutdown())
        self.event_loop.close()
        self.logger.log_info('Engine loop closed')

    async def _await_shutdown(self):
        # await the shutdown signal
        await self.shutdown_event.wait()
        self.logger.log_info('Get shutdown signal')

        # cancel unfinished tasks
        for task in asyncio.all_tasks(loop=self.event_loop):
            if task is not asyncio.current_task(loop=self.event_loop):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    self.logger.log_info('Unfinished task is cancelled')

    def _get_optional_configs(self, request):
        optional_configs = {}
        config_names = [
            'temperature', 'top_p', 'top_k', 'stop_words', 'bad_words',
            'repetition_penalty', 'skip_special_tokens'
        ]
        for config_name in config_names:
            input_tensor = pb_utils.get_input_tensor_by_name(
                request, config_name)
            if input_tensor is not None:
                if config_name == 'stop_words' or config_name == 'bad_words':
                    optional_configs[config_name] = [
                        obj.decode() if isinstance(obj, bytes) else obj
                        for obj in input_tensor.as_numpy().tolist()
                    ]
                else:
                    optional_configs[config_name] = input_tensor.as_numpy(
                    ).item()
        return optional_configs

    async def _process_request(self, request_id, request):
        response_sender = request.get_response_sender()

        try:
            # parse request
            prompt = pb_utils.get_input_tensor_by_name(
                request, 'prompt').as_numpy().item()
            if isinstance(prompt, bytes):
                prompt = prompt.decode()
            max_tokens = pb_utils.get_input_tensor_by_name(
                request, 'max_tokens').as_numpy().item()
            ignore_eos = pb_utils.get_input_tensor_by_name(
                request, 'ignore_eos').as_numpy().item()
            stream = pb_utils.get_input_tensor_by_name(
                request, 'stream').as_numpy().item()

            optional_configs = self._get_optional_configs(request)

            gen_config = GenerationConfig(max_new_tokens=max_tokens,
                                          ignore_eos=ignore_eos,
                                          **optional_configs)

            outputs = []
            async for output in self.engine.generate(messages=prompt,
                                                     session_id=request_id,
                                                     stream_response=stream,
                                                     gen_config=gen_config,
                                                     do_preprocess=False):

                if stream:
                    # for stream mode, send the partial response one by one
                    triton_output_tensor = pb_utils.Tensor(
                        'response',
                        np.asarray(output.response, dtype=np.object_))
                    resp = pb_utils.InferenceResponse(
                        output_tensors=[triton_output_tensor])
                    if output.finish_reason is not None:
                        response_sender.send(
                            resp,
                            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                        )
                    else:
                        response_sender.send(resp)
                else:
                    outputs.append(output.response)

            if not stream:
                # for non-stream mode, send concatenated response at one time
                triton_output_tensor = pb_utils.Tensor(
                    'response', np.asarray(''.join(outputs), dtype=np.object_))
                resp = pb_utils.InferenceResponse(
                    output_tensors=[triton_output_tensor])
                response_sender.send(
                    resp, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        except Exception as e:
            # error handling
            self.logger.log_info(f'Error when processing request: {e}')
            error = pb_utils.TritonError(f'Error when processing request: {e}')
            triton_output_tensor = pb_utils.Tensor(
                'response', np.asarray(['N/A'], dtype=np.object_))
            resp = pb_utils.InferenceResponse(
                output_tensors=[triton_output_tensor], error=error)
            response_sender.send(
                resp, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            raise e

    def execute(self, requests):
        for request in requests:
            asyncio.run_coroutine_threadsafe(
                self._process_request(self.request_id, request),
                self.event_loop)
            self.request_id += 1
        return None

    def finalize(self):
        self.logger.log_info('Finalizing LMDeploy backend')
        self.event_loop.call_soon_threadsafe(self.shutdown_event.set)
        if self.engine_thread is not None:
            self.engine_thread.join()
            self.engine_thread = None

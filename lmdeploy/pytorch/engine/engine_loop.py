# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.profiler import record_function

from lmdeploy.messages import RequestMetrics
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.pytorch.messages import MessageStatus, UpdateTokenMode
from lmdeploy.utils import get_logger

from .engine import InferOutput, ResponseType, response_reqs

if TYPE_CHECKING:
    from lmdeploy.pytorch.disagg.conn.engine_conn import EngineP2PConnection
    from lmdeploy.pytorch.engine.model_agent import BatchedOutputs
    from lmdeploy.pytorch.model_inputs import ModelInputs
    from lmdeploy.pytorch.paging import Scheduler
    from lmdeploy.pytorch.strategies.base.sequence import SequenceStrategy

    from .engine import Engine, SeqList
    from .executor import ExecutorBase
    from .inputs_maker import InputsMakerAsync
    from .request import RequestManager

logger = get_logger('lmdeploy')
_EMPTY_TOKEN = np.empty((0, ), dtype=np.int64)


class CounterEvent(asyncio.Event):

    def __init__(self):
        super().__init__()
        self._counter = 0

    def set(self):
        if self._counter > 0:
            self._counter -= 1
        if self._counter == 0:
            super().set()

    def clear(self):
        if self._counter == 0 and super().is_set():
            super().clear()
        self._counter += 1


class RunableEventAsync:
    """Awaitable async runable event."""

    def __init__(self, scheduler: 'Scheduler'):
        self.scheduler = scheduler
        self.event = asyncio.Event()

    async def wait(self):
        """Wait event."""
        await self.event.wait()

    def set(self):
        """Set event."""
        if self.scheduler.has_unfinished():
            self.event.set()
        else:
            self.event.clear()


def build_runable_event(scheduler: 'Scheduler'):
    """Build runable event."""
    return RunableEventAsync(scheduler)


@dataclass
class EngineLoopConfig:
    """Engine loop config.

    This config is added for Dependency Injection
    """
    role: EngineRole
    num_speculative_tokens: Optional[int] = None
    enable_metrics: bool = False
    enable_transfer_obj_ref: bool = False

    @staticmethod
    def from_engine(engine: 'Engine'):
        """Create engine loop config from engine."""
        if engine.specdecode_config is None:
            num_speculative_tokens = None
        else:
            num_speculative_tokens = engine.specdecode_config.num_speculative_tokens

        return EngineLoopConfig(
            role=engine.engine_config.role,
            num_speculative_tokens=num_speculative_tokens,
            enable_metrics=engine.engine_config.enable_metrics,
            enable_transfer_obj_ref=engine.engine_config.enable_transfer_obj_ref,
        )


class EngineLoop:
    """Engine loop manager should be created in an async context."""

    def __init__(self,
                 req_manager: 'RequestManager',
                 scheduler: 'Scheduler',
                 executor: 'ExecutorBase',
                 seq_strategy: 'SequenceStrategy',
                 inputs_maker: 'InputsMakerAsync',
                 config: EngineLoopConfig,
                 engine_conn: Optional['EngineP2PConnection'] = None):
        self.req_manager = req_manager
        self.scheduler = scheduler
        self.executor = executor
        self.seq_strategy = seq_strategy
        self.inputs_maker = inputs_maker
        self.config = config
        self.engine_conn = engine_conn

        # tasks and control events
        self.tasks: Set[asyncio.Task] = set()
        self.stop_event = asyncio.Event()
        self.resp_queue = asyncio.Queue()
        self.forward_event = CounterEvent()
        self.migration_event = asyncio.Event()
        self.has_runable_event = RunableEventAsync(self.scheduler)

        # check init
        if self.config.role != EngineRole.Hybrid:
            assert self.engine_conn is not None, 'Engine connection must be provided for non-hybrid engine role.'

    async def preprocess_loop(self):
        """Preprocess request."""
        while not self.stop_event.is_set():
            await self.forward_event.wait()
            await self.req_manager.step()
            self.has_runable_event.set()

    @staticmethod
    def _log_resps(outputs: List[InferOutput]):
        """Log resps."""
        if logger.level <= logging.DEBUG:
            session_ids = [out.session_id for out in outputs]
            logger.debug(f'Response sessions: {session_ids}')
        elif logger.level <= logging.INFO:
            logger.info(f'Response: num_outputs={len(outputs)}.')

    def _send_resp(self, out: InferOutput):
        """Send response."""
        resp_type = (ResponseType.FINISH if out.finish else ResponseType.SUCCESS)
        logprobs = None if out.resp.data is None else out.resp.data.get('logprobs', None)
        response_reqs(self.req_manager,
                      out.resp,
                      resp_type,
                      data=dict(token_ids=out.token_ids,
                                logits=out.logits,
                                cache_block_ids=out.cache_block_ids,
                                req_metrics=out.req_metrics,
                                routed_experts=out.routed_experts,
                                logprobs=logprobs))

    @staticmethod
    def _update_logprobs(step_outputs: List[InferOutput]):
        for out in step_outputs:
            cur_logprobs = out.logprobs
            if cur_logprobs is None:
                continue

            if out.resp.data is None:
                out.resp.data = dict()
            out.resp.data.setdefault('logprobs', [])

            # logprobs to dict
            vals = cur_logprobs[0]
            indices = cur_logprobs[1]
            cur_logprobs = dict(zip(indices, vals))
            logprobs = out.resp.data['logprobs']
            logprobs.append(cur_logprobs)

    def _send_resps(self, step_outputs: List[InferOutput]):
        """Send response callback."""
        self._log_resps(step_outputs)
        self._update_logprobs(step_outputs)

        is_done = set()
        for out in reversed(step_outputs):
            if out.session_id in is_done:
                continue
            is_done.add(out.session_id)
            self._send_resp(out)

    async def send_response_loop(self):
        """Send response to client."""
        que = self.resp_queue
        while not self.stop_event.is_set():
            num_outs = que.qsize()
            if num_outs > 0:
                resps = []
                for _ in range(num_outs):
                    resps += que.get_nowait().values()
            else:
                resps = (await que.get()).values()
            await self.forward_event.wait()
            self._send_resps(resps)

    @record_function('make_infer_outputs')
    def _make_infer_outputs(
        self,
        batched_outputs: 'BatchedOutputs',
        running: 'SeqList',
        model_inputs: 'ModelInputs',
    ):
        """Make infer output."""
        new_token_timestamp = batched_outputs.new_token_timestamp
        logits = batched_outputs.logits
        logprobs = batched_outputs.logprobs

        if logprobs is not None:
            logprobs.vals = logprobs.vals.tolist()
            logprobs.indices = logprobs.indices.tolist()

        seq_length = [seq.num_token_ids for seq in running]
        is_run = [seq.status == MessageStatus.RUNNING for seq in running]
        self.seq_strategy.update_running(running=running, batched_outputs=batched_outputs, model_inputs=model_inputs)

        # generate output
        outputs: Dict[int, InferOutput] = dict()
        for idx, msg in enumerate(running):
            if not is_run[idx]:
                continue
            token_ids = msg.generated_ids
            finish = msg.status == MessageStatus.STOPPED or msg.status == MessageStatus.TO_BE_MIGRATED
            if not finish and len(token_ids) == 0:
                continue
            resp_data = msg.resp.data
            if resp_data is not None and len(resp_data.get('token_ids', [])) == len(token_ids):
                # no new tokens
                continue
            session_id = msg.session_id
            if msg.resp_cache:
                cache_block_ids = self.scheduler.block_manager.get_block_table(msg).tolist()
            else:
                cache_block_ids = None

            # logprobs
            num_logprobs = msg.sampling_param.num_logprobs
            cur_logprobs = None
            if logprobs is not None and num_logprobs > 0:
                cur_logprobs = (logprobs.vals[idx][:num_logprobs + 1], logprobs.indices[idx][:num_logprobs + 1])
            # get spec stats info
            spec_info = None
            num_draft_tokens = self.config.num_speculative_tokens
            if num_draft_tokens is not None and model_inputs.is_decoding and self.config.enable_metrics:
                num_accepted_tokens = (batched_outputs.next_token_ids[idx] > -1).sum() - 1
                spec_info = dict(num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens)
            req_metrics = RequestMetrics(new_token_timestamp, msg.engine_events, spec_info=spec_info)
            out = InferOutput(session_id=session_id,
                              resp=msg.resp,
                              finish=finish,
                              token_ids=token_ids,
                              cache_block_ids=cache_block_ids,
                              req_metrics=req_metrics,
                              logprobs=cur_logprobs,
                              routed_experts=msg.routed_experts)
            outputs[session_id] = out

            if msg.return_logits:
                outputs[session_id].logits = logits.split(seq_length)[idx]
        return outputs

    async def _main_loop_try_send_next_inputs(self):
        """Try send next inputs."""
        scheduler = self.scheduler
        if not scheduler.has_unfinished():
            self.forward_event.set()
            await self.has_runable_event.wait()
            self.forward_event.clear()

        scheduler.collect_migration_done()
        return await self.inputs_maker.send_next_inputs()

    async def _main_loop_get_outputs(
        self,
        running: 'SeqList',
        forward_inputs: Dict[str, Any],
    ):
        """Get outputs and prefetch."""
        self.forward_event.set()
        num_loops = forward_inputs['loop_count']
        model_inputs = forward_inputs['inputs']
        for idx in range(num_loops):
            # pre-forward before get last token
            if idx == num_loops - 1:
                self.scheduler.collect_migration_done()
                forward_inputs, next_running = await self.inputs_maker.prefetch_next_inputs()

            # send output
            out = await self.executor.get_output_async()
            if out is not None:
                step_outputs = self._make_infer_outputs(out, running=running, model_inputs=model_inputs)
                self.resp_queue.put_nowait(step_outputs)

            # lock forward event
            # make sure that prefetch forward would not wait for detokenize
            # WARNING: this might have side effect on the performance
            if idx == num_loops // 2:
                self.forward_event.clear()
        return forward_inputs, next_running

    async def main_loop(self):
        """Main loop of the engine.

        Each engine instance would communicate with the engine by queue.
        """
        forward_event = self.forward_event
        has_runable_event = self.has_runable_event
        scheduler = self.scheduler
        forward_inputs = None
        next_running = None

        async def __no_running_warning():
            # TODO (JimyMa): add watermark check event instead of async sleep.
            # self.perfill_watermark_event.wait()
            logger.warning(f'no next prefill running request, Maybe cache is full, '
                           f'free gpu cache blocks: {scheduler.block_manager.get_num_free_gpu_blocks()}, '
                           f'total gpu cache blocks: {scheduler.block_manager.num_gpu_blocks}')
            forward_event.set()
            await asyncio.sleep(0.1)
            forward_event.clear()

        while not self.stop_event.is_set():
            if next_running is None:
                forward_inputs, next_running = await self._main_loop_try_send_next_inputs()
                if next_running is None:
                    await __no_running_warning()
                    continue

            with scheduler.seqs_activation(next_running):
                forward_inputs, next_running = await self._main_loop_get_outputs(
                    running=next_running,
                    forward_inputs=forward_inputs,
                )
            has_runable_event.set()

    def update_running_migration(self, running: 'SeqList', next_token_ids: np.ndarray, stopped: torch.Tensor,
                                 model_metas: List[Dict[str, Any]]):
        """Update scheduler."""
        if model_metas is None:
            model_metas = [None] * len(running)
        for token, msg, stop, model_meta in zip(next_token_ids, running, stopped, model_metas):
            if msg.status != MessageStatus.MIGRATION_RUNNING:
                continue
            update_token = token

            # fill token
            msg.update_token_ids(update_token, model_meta=model_meta, mode=UpdateTokenMode.PREFILL)
            if stop:
                update_token = _EMPTY_TOKEN
                msg.update_token_ids(update_token, model_meta=model_meta, mode=UpdateTokenMode.PREFILL)
                msg.state.finish()

    async def _migration_loop_migrate(self, migration_ready: 'SeqList'):
        """Migration loop migrate."""
        for msg in migration_ready:
            # skip dummy prefill migration
            if msg.migration_request.is_dummy_prefill:
                continue

            migration_execution_requests: List[Tuple[int, List[Tuple[int, int]]]] = []
            migration_request = msg.migration_request
            prefill_block_ids = migration_request.remote_block_ids
            decode_block_ids = list(self.scheduler.block_manager.get_block_table(msg=msg))

            assert len(prefill_block_ids) == len(decode_block_ids), (
                f'#prefill block ids ({len(prefill_block_ids)}) must equal to '
                f'#decode block ids ({len(decode_block_ids)})'
                f'all id length: {len(msg.num_token_ids)}')
            migration_execution_requests.append((
                migration_request.remote_engine_id,
                list(zip(prefill_block_ids, decode_block_ids)),
            ))
            migration_inputs = MigrationExecutionBatch(protocol=migration_request.protocol,
                                                       requests=migration_execution_requests)
            logger.info(f'migrating session: {msg.session_id} begin')
            await self.executor.migrate(migration_inputs)
            logger.info(f'migrating session: {msg.session_id} done')
            await self.engine_conn.zmq_send(remote_engine_id=migration_request.remote_engine_id,
                                            remote_session_id=migration_request.remote_session_id)

    async def _migration_loop_get_outputs(self, migration_ready: 'SeqList'):
        """Migration loop get outputs."""
        outputs: Dict[int, InferOutput] = dict()
        for _, msg in enumerate(migration_ready):
            session_id = msg.session_id
            msg.resp.type = ResponseType.SUCCESS
            token_ids = [msg.migration_request.remote_token_id]
            # MUST be a wall-clock time
            new_token_timestamp = time.time()
            req_metrics = RequestMetrics(new_token_timestamp, msg.engine_events)
            out = InferOutput(
                session_id=session_id,
                resp=msg.resp,
                finish=False,
                token_ids=np.array(token_ids),
                req_metrics=req_metrics,
            )
            outputs[session_id] = out
            self.update_running_migration([msg], np.array([token_ids]), [False], [None])
        self.resp_queue.put_nowait(outputs)

    async def _migration_loop_process_ready(self, migration_ready: 'SeqList'):
        """Process migration ready."""
        await self._migration_loop_migrate(migration_ready)

        # generate output
        with self.scheduler.seqs_migration_activation(migration_ready):
            await self._migration_loop_get_outputs(migration_ready)
        self.has_runable_event.set()

    async def migration_loop(self):
        """Async loop migration."""
        while not self.stop_event.is_set():
            migration_ready = self.scheduler._schedule_migration()
            if not migration_ready and not self.scheduler.has_migration_waiting():
                await self.migration_event.wait()
            elif migration_ready:
                self.migration_event.clear()
                await self._migration_loop_process_ready(migration_ready)
            else:
                # release coroutine for decoding
                await asyncio.sleep(.5)

    def _add_loop_tasks_done_callback(self):
        """Add loop tasks done callback."""

        def __task_callback(task: asyncio.Task) -> None:
            """Raise exception on finish."""
            task_name = task.get_name()
            try:
                task.result()
            except asyncio.CancelledError:
                logger.info(f'Task <{task_name}> cancelled.')
            except BaseException:
                logger.exception(f'Task <{task_name}> failed')
            finally:
                self.stop()
                self.cancel()

        for task in self.tasks:
            task.add_done_callback(__task_callback)

    def create_tasks(self, event_loop: asyncio.AbstractEventLoop):
        """Create async tasks."""
        logger.info('Starting async task MainLoopPreprocessMessage.')
        self.tasks.add(event_loop.create_task(self.preprocess_loop(), name='MainLoopPreprocessMessage'))
        logger.info('Starting async task MainLoopResponse.')
        self.tasks.add(event_loop.create_task(self.send_response_loop(), name='MainLoopSendResponse'))
        logger.info('Starting async task MainLoop.')
        self.tasks.add(event_loop.create_task(self.main_loop(), name='MainLoopMain'))
        if self.config.role != EngineRole.Hybrid:
            logger.info('Starting async task MigrationLoop.')
            self.tasks.add(event_loop.create_task(self.migration_loop(), name='MainLoopMigration'))

        self._add_loop_tasks_done_callback()

    async def wait_tasks(self):
        """Wait for all tasks to finish."""
        if not self.tasks:
            return

        try:
            done, pending = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_EXCEPTION)

            # cancel all pending tasks
            for task in pending:
                task.cancel()

            for task in done:
                try:
                    task.result()
                except asyncio.CancelledError:
                    logger.debug('Task cancelled.')
        except asyncio.CancelledError:
            logger.info('Engine loop wait tasks cancelled.')
            raise
        except BaseException:
            logger.exception('Engine loop wait tasks failed.')
        finally:
            self.stop()
            self.cancel()

    def stop(self):
        """Stop all loops."""
        self.stop_event.set()

    def cancel(self):
        """Cancel all loops."""
        for task in self.tasks:
            if not task.done():
                task.cancel()
        self.tasks.clear()


def build_engine_loop(engine: 'Engine'):
    """Build engine loop."""
    from .inputs_maker import build_inputs_maker

    config = EngineLoopConfig.from_engine(engine)
    inputs_maker = build_inputs_maker(engine)
    return EngineLoop(
        req_manager=engine.req_manager,
        scheduler=engine.scheduler,
        executor=engine.executor,
        seq_strategy=engine.seq_strategy,
        inputs_maker=inputs_maker,
        config=config,
        engine_conn=engine.engine_conn,
    )

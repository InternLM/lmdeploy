# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import queue
import json
import threading
import time
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.qos_engine.usage_stats import UsageStats
from lmdeploy.serve.qos_engine.inner_group_schd import UserRequestQueue

from lmdeploy.serve.openai.protocol import (ChatCompletionRequestQos, CompletionRequestQos, GenerateRequestQos)

import logging
logger = logging.getLogger(__name__)

class QosConfig:
    def __init__(self, qos_tag=""):
        try:
            qos_config = json.loads(qos_tag)
            self.is_qos_enabled = qos_config["enable_user_qos"]
            self.user_id_maps = qos_config["user_group_map"]
            self.user_group_prio = qos_config["user_groups"]
        except:
            self.is_qos_enabled = False
            self.user_id_maps = dict()
            self.user_group_prio = []
        logger.debug(f"is_qos_enabled: {self.is_qos_enabled}")
        logger.debug(f"user_id_maps:  {self.user_id_maps}")
        logger.debug(f"user_group_prio: {self.user_group_prio}")

class QosEngine:
    def __init__(self, instance_num=16, qos_tag="", engine=None, **kwargs) -> None:
        self.engine = engine
        self.availSlots = instance_num
        self._stop_event = threading.Event()
        self._dequeue_thread = threading.Thread(target=self._serve, daemon=True)
        self.qos_config = QosConfig(qos_tag)

        self.qos_user_group = QosGroupQueue(self.qos_config)

        self.usage_stats = UsageStats(60, 6, 0, self.qos_config.user_group_prio)
        self.user_served_reqs = dict()
        self._dump_stats_thread = threading.Thread(target=self._dump_stats, daemon=True)

        self.lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
    def start(self):
        if self.is_qos_enabled():
            self._dequeue_thread.start()
            self._dump_stats_thread.start()
    
    def is_qos_enabled(self):
        return self.qos_config.is_qos_enabled

    def stop_session(self, session_id: int):
        """Stop a session by a session_id."""
        self.engine.stop_session(session_id)

    async def generate(self, request):

        if isinstance(request,CompletionRequestQos):
            generators = []
            for i in range(len(request.prompt)):
                result_generator = self.engine.generate(
                    request.prompt[i],
                    request.session_id + i,
                    True,  # always use stream to enable batching
                    sequence_start=True,
                    sequence_end=True,
                    request_output_len=request.max_tokens
                    if request.max_tokens else 512,
                    stop=False,
                    top_p=request.top_p,
                    temperature=request.temperature,
                    repetition_penalty=request.repetition_penalty,
                    ignore_eos=request.ignore_eos,
                    do_preprocess=False)
                generators.append(result_generator)
            return generators
            
        elif isinstance(request,GenerateRequestQos):
            async_engine = self.engine
            sequence_start = async_engine.steps.get(str(request.session_id), 0) == 0
            sequence_end = not request.interactive_mode

            generation = async_engine.generate(
                request.prompt,
                request.session_id,
                stream_response=True,  # always use stream to enable batching
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                request_output_len=request.request_output_len,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty,
                ignore_eos=request.ignore_eos)
            return generation

        elif isinstance(request,ChatCompletionRequestQos):
            # default chat/completions
            result_generator =self.engine.generate(
                request.messages,
                request.session_id,
                True,  # always use stream to enable batching
                sequence_start=True,
                sequence_end=True,
                request_output_len=request.max_tokens if request.max_tokens else 512,
                stop=request.stop,
                top_p=request.top_p,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty,
                ignore_eos=request.ignore_eos)
            return result_generator

        return time.sleep(0.01)

    async def generate_with_qos(self, request):
        if not self.is_qos_enabled():
            return await self.generate(request)

        # push (request,event) to queue
        event = asyncio.Event()
        request_event = (request,event)
        with self.lock:
            self.qos_user_group.enqueue(request_event)

        await event.wait()

        result_generator = await self.generate(request)
        
        # release self.availSlots resources
        with self.lock:
            if hasattr(request,'prompt'):
                self.availSlots += len(request.prompt)
            else:
                self.availSlots += 1

        # Update number of served requests for each user
        with self.stats_lock:
            if request.user_id not in self.user_served_reqs:
                self.user_served_reqs[request.user_id] = 1
            else:
                self.user_served_reqs[request.user_id] += 1

        return result_generator


    def _serve(self):
        while not self._stop_event.is_set():
            if self.availSlots > 0:
                with self.lock:
                    request_event = self.dequeue(self.usage_stats)
                    if request_event != None:
                        # Update usage_stats
                        user_group = self.qos_user_group.get_user_group(request_event[0].user_id)
                        self.usage_stats.update_usage(request_event[0].user_id, user_group, 100, int(time.time()))
                        if hasattr(request_event[0],'prompt'):
                            self.availSlots -= len(request_event[0].prompt)
                        else:
                            self.availSlots -= 1
                        request_event[1].set()
                        logger.debug(f"Available slot decrease, now: {self.availSlots}")
            time.sleep(0)

    def _dump_stats(self):
        ts = 0
        while not self._stop_event.is_set():
            outdata = ""
            with self.stats_lock:
                if not self.user_served_reqs:
                    outdata = "none"
                else:
                    sorted_uids = sorted(self.user_served_reqs.keys())
                    for uid in sorted_uids:
                        outdata += f"{uid} {self.user_served_reqs[uid]} reqs, "
                    self.user_served_reqs = dict()
            logger.info(f"qos service running for {ts} seconds, served in last 20 seconds: {outdata}")
            ts += 20
            time.sleep(20)

    def dequeue(self, usage_stats):
        return self.qos_user_group.dequeue(usage_stats)

    def stop(self):
        self._stop_event.set()
        self._dequeue_thread.join()

class QosGroupQueue:
    def __init__(self,qos_config):
        if qos_config == None:
            self.user_list = {}
            self.queues = {}
        else:    
            self.user_list = qos_config.user_id_maps
            self.queues = {}
            for user_group in qos_config.user_group_prio:
                self.queues[user_group] = UserRequestQueue(user_group, self.user_list[user_group])
        self.user_group_list = list(self.user_list.keys())
        self.default_user_group = self.user_group_list[2] if len(self.user_group_list)>=3 else "None"
        logger.debug(self.user_list)
        logger.debug(self.queues)
        logger.debug(self.default_user_group)

    def get_user_group(self, user_id):
        for category, users in self.user_list.items():
            for user in users:
                if user_id == user['id']:
                    return category
        return self.default_user_group
    def enqueue(self, request_event):
        user_id = self.get_user_group(request_event[0].user_id)
        self.queues[user_id].enqueue(request_event)

    def dequeue(self, usage_stats):
        for user_group_id, user_group_queue in self.queues.items():
            if user_group_queue.empty():
                continue
            else:
                return user_group_queue.dequeue(usage_stats)
        return None


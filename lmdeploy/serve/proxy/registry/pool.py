# Copyright (c) OpenMMLab. All rights reserved.

import copy
import json
import threading
import time
from http import HTTPStatus

import requests

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.core.errors import ErrorCodes, err_msg
from lmdeploy.serve.proxy.core.replica import ReplicaLoad
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class ReplicaPool:
    """In-memory registry of api_server replicas."""

    def __init__(self, pd_connection_pool) -> None:
        self._replicas: dict[str, ReplicaLoad] = {}
        self._lock = threading.Lock()
        self.pd_connection_pool = pd_connection_pool

    def get_by_role(self, role: EngineRole) -> dict[str, ReplicaLoad]:
        with self._lock:
            items = list(self._replicas.items())
        return {url: load for url, load in items if load.role == role}

    @property
    def hybrid_replicas(self) -> dict[str, ReplicaLoad]:
        return self.get_by_role(EngineRole.Hybrid)

    @property
    def prefill_replicas(self) -> dict[str, ReplicaLoad]:
        return self.get_by_role(EngineRole.Prefill)

    @property
    def decode_replicas(self) -> dict[str, ReplicaLoad]:
        return self.get_by_role(EngineRole.Decode)

    def snapshot(self) -> dict[str, ReplicaLoad]:
        with self._lock:
            return copy.deepcopy(self._replicas)

    @property
    def model_list(self) -> list[str]:
        names: list[str] = []
        with self._lock:
            for load in self._replicas.values():
                names.extend(load.models)
        return list(dict.fromkeys(names))

    def add(self, url: str, load: ReplicaLoad | None = None) -> bytes | None:
        if load is None:
            with self._lock:
                existing = self._replicas.get(url)
                load = existing.model_copy(deep=True) if existing else ReplicaLoad()
        if load.models:
            self.remove(url)
            with self._lock:
                self._replicas[url] = load.model_copy(deep=True)
            return None
        try:
            from lmdeploy.serve.openai.api_client import APIClient

            client = APIClient(api_server_url=url)
            discovered = load.model_copy(deep=True) if load is not None else ReplicaLoad()
            discovered.models = client.available_models
            with self._lock:
                self._replicas[url] = discovered
        except requests.exceptions.RequestException as e:
            logger.error(f'exception happened when adding replica {url}, {e}')
            return self._api_timeout_bytes(url)
        return None

    def remove(self, url: str) -> None:
        with self._lock:
            existed = self._replicas.pop(url, None)
        if existed is not None:
            self.pd_connection_pool.dereg_instance(url)

    def terminate(self, url: str) -> bool:
        success = True
        with self._lock:
            status_exists = url in self._replicas
            if status_exists:
                self._replicas.pop(url)
        if status_exists:
            headers = {'accept': 'application/json'}
            try:
                response = requests.get(f'{url}/terminate', headers=headers)
                if response.status_code != HTTPStatus.OK:
                    success = False
                    logger.error(f'Failed to terminate replica {url}, '
                                 f'error_code={response.status_code}, '
                                 f'error_msg={response.text}')
            except Exception as e:
                logger.error(f'exception happened when terminating replica {url}, {e}')
                success = False
        else:
            logger.error(f'terminating replica {url} failed since it does not exist. '
                         'May try /nodes/status to check the replica list')
            success = False
        return success

    def terminate_all(self) -> bool:
        with self._lock:
            urls = list(self._replicas.keys())
        all_success = True
        for url in urls:
            if not self.terminate(url):
                all_success = False
        return all_success

    def inflight_start(self, url: str) -> float | None:
        with self._lock:
            load = self._replicas.get(url)
            if load is None:
                return None
            load.unfinished += 1
        return time.time()

    def inflight_finish(self, url: str, start: float | None) -> None:
        if start is None:
            return
        with self._lock:
            load = self._replicas.get(url)
            if load is None:
                return
            load.unfinished -= 1
            load.record_latency(time.time() - start)

    def remove_stale_replicas(self) -> None:
        with self._lock:
            urls = list(self._replicas.keys())
        to_delete: list[str] = []
        for url in urls:
            health_url = f'{url}/health'
            headers = {'accept': 'application/json'}
            try:
                response = requests.get(health_url, headers=headers)
                if response.status_code != HTTPStatus.OK:
                    to_delete.append(url)
            except requests.exceptions.RequestException:
                to_delete.append(url)
        for url in to_delete:
            self.remove(url)
            logger.info(f'Removed replica url: {url} due to heart beat expiration')

    @staticmethod
    def _api_timeout_bytes(url: str) -> bytes:
        logger.warning(f'api timeout: {url}')
        ret = {
            'error_code': ErrorCodes.API_TIMEOUT.value,
            'text': err_msg[ErrorCodes.API_TIMEOUT],
        }
        return json.dumps(ret).encode() + b'\n'

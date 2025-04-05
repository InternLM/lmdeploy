import requests

from lmdeploy.disagg.messages import RemoteEngineConfig, RDMAConnectRequest, RDMAInitRequest


def pd_consolidation(
    prefill_engine_id: int,
    prefill_endpoint: str,
    decode_engine_id: int,
    decode_endpoint: str,
    mode="rdma",
):
    """pd consolidation."""
    if mode == "rdma":
        def get_config(engine_id, endpoint):
            engine_config_raw = requests.get(
                f"{endpoint}/distserve/get_disagg_info", timeout=5
            ).json()
            config = RDMAInitRequest(
                remote_engine_id=engine_id,
                remote_engine_config=RemoteEngineConfig.model_validate_json(
                    engine_config_raw
                ),
            )
            return config

        prefill_config = get_config(prefill_engine_id, prefill_endpoint)
        decode_config = get_config(decode_engine_id, decode_endpoint)
        prefill_endpoint_info = requests.post(
            f"{prefill_endpoint}/distserve/init_rdma_endpoint",
            json=decode_config.model_dump(),
            timeout=5,
        ).json()
        decode_endpoint_info = requests.post(
            f"{decode_endpoint}/distserve/init_rdma_endpoint",
            json=prefill_config.model_dump(),
            timeout=5,
        ).json()

        requests.post(
            f"{decode_endpoint}/distserve/rdma_connect",
            json=RDMAConnectRequest(
                remote_engine_id=prefill_engine_id,
                remote_endpoint_info=prefill_endpoint_info,
            ).model_dump(),
            timeout=5
        )
        requests.post(
            f"{prefill_endpoint}/distserve/rdma_connect",
            json=RDMAConnectRequest(
                remote_engine_id=decode_engine_id,
                remote_endpoint_info=decode_endpoint_info,
            ).model_dump(),
            timeout=5,
        )
    else:
        raise NotImplementedError

import collections
import concurrent.futures
import time

import pytest
import requests


@pytest.mark.integration
@pytest.mark.pd_engine
def test_pd_power_of_two_decode_attribution(router_manager, mock_workers):
    # Start two prefill and three decode mock workers via fixture
    _, prefill_urls_raw, prefill_ids = mock_workers(n=2)
    _, decode_urls_raw, decode_ids_list = mock_workers(n=3)
    prefill_urls = list(prefill_urls_raw)
    decode_urls = list(decode_urls_raw)
    decode_ids = set(decode_ids_list)

    rh = router_manager.start_router(
        policy="power_of_two",
        lmdeploy_pd_disaggregation=True,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    counts = collections.Counter()
    with requests.Session() as s:
        for i in range(30):
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"p{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            assert wid in decode_ids
            counts[wid] += 1

    assert sum(1 for v in counts.values() if v > 0) >= 2


@pytest.mark.integration
@pytest.mark.pd_engine
def test_pd_power_of_two_skews_to_faster_decode(router_manager, mock_workers):
    # Start two prefill workers (fast)
    _, prefill_urls_raw, _ = mock_workers(n=2)

    # Start two decode workers: one very slow, one fast
    # Use 2000ms latency to create a clear load difference
    _, [decode_slow_url], [slow_id] = mock_workers(
        n=1, args=["--latency-ms", "2000"]
    )  # 2 second latency - very slow
    _, [decode_fast_url], [fast_id] = mock_workers(n=1)
    decode_urls_raw = [decode_slow_url, decode_fast_url]

    prefill_urls = list(prefill_urls_raw)
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="power_of_two",
        lmdeploy_pd_disaggregation=True,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    # Prime the router with some initial requests
    def _prime_call(i):
        try:
            requests.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"warm-{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=10,
            )
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_prime_call, range(50)))
    time.sleep(1)

    # Create sustained load on slow worker by sending requests in background
    # These will take 2 seconds each, keeping the slow worker loaded
    stop_background_load = False

    def _background_load():
        i = 0
        while not stop_background_load:
            try:
                requests.post(
                    f"{decode_slow_url}/v1/completions",
                    json={
                        "model": "test-model",
                        "prompt": f"bg-{i}",
                        "max_tokens": 1,
                        "stream": False,
                    },
                    timeout=10,
                )
                i += 1
            except Exception:
                pass

    # Start background load in separate threads
    import threading

    bg_threads = [
        threading.Thread(target=_background_load, daemon=True) for _ in range(4)
    ]
    for t in bg_threads:
        t.start()

    # Wait for slow worker to accumulate load
    time.sleep(3)

    # Now send test requests - power-of-two should strongly prefer the fast worker
    def call(i):
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": f"p{i}",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=10,
        )
        assert r.status_code == 200
        return r.headers.get("X-Worker-Id") or r.json().get("worker_id")

    counts = collections.Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        for wid in ex.map(call, range(200)):
            counts[wid] += 1

    # Stop background load
    stop_background_load = True

    # With sustained load on slow worker, fast worker should get significantly more requests
    # Allow for some variance, but expect at least 60/40 split favoring fast worker
    assert (
        counts[fast_id] > counts[slow_id]
    ), f"Expected fast worker to handle more requests, got {counts}"
    assert (
        counts[fast_id] >= counts[slow_id] * 1.2
    ), f"Expected fast worker to handle at least 20% more requests than slow worker, got {counts}"

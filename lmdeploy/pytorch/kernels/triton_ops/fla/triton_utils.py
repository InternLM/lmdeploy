from typing import Any, Dict

import torch
import triton

HAS_TRITON = True
_NUM_AICORE = -1
_NUM_VECTORCORE = -1


def init_device_properties_triton():
    global _NUM_AICORE, _NUM_VECTORCORE
    if _NUM_AICORE == -1 and HAS_TRITON:
        device_properties: Dict[str, Any] = (
            triton.runtime.driver.active.utils.get_device_properties(
                torch.npu.current_device()))
        _NUM_AICORE = device_properties.get("num_aicore", -1)
        _NUM_VECTORCORE = device_properties.get("num_vectorcore", -1)
        assert _NUM_AICORE > 0 and _NUM_VECTORCORE > 0, "Failed to detect device properties."


def get_aicore_num():
    global _NUM_AICORE
    assert _NUM_AICORE > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _NUM_AICORE


def get_vectorcore_num():
    global _NUM_VECTORCORE
    assert _NUM_VECTORCORE > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _NUM_VECTORCORE

import torch
from lmdeploy.lite.utils import (
    cal_qparams_per_channel_absmax,
    cal_qparams_per_channel_minmax,
    cal_qparams_per_group_absmax,
    cal_qparams_per_group_minmax,
    cal_qparams_per_tensor_absmax,
    cal_qparams_per_tensor_minmax
)

def test_cal_qparams():
    """Test function for quantization parameter calculation."""
    
    # Create a dummy tensor
    w = torch.randn(64, 64)

    # Test per-channel absmax method
    qparams = cal_qparams_per_channel_absmax(w, 8)
    assert qparams.scales.shape == (64, 1)
    assert qparams.zero_points is None

    # Test per-channel minmax method
    qparams = cal_qparams_per_channel_minmax(w, 8)
    assert qparams.scales.shape == (64, 1)
    assert qparams.zero_points.shape == (64, 1)

    # Test per-group absmax method
    qparams = cal_qparams_per_group_absmax(w, 8, 16)
    assert qparams.scales.shape == (64, 1)
    assert qparams.zero_points is None

    # Test per-group minmax method
    qparams = cal_qparams_per_group_minmax(w, 8, 16)
    assert qparams.scales.shape == (64, 1)
    assert qparams.zero_points.shape == (64, 1)

    # Test per-tensor absmax method
    qparams = cal_qparams_per_tensor_absmax(w, 8)
    assert qparams.scales.shape == ()
    assert qparams.zero_points is None

    # Test per-tensor minmax method
    qparams = cal_qparams_per_tensor_minmax(w, 8)
    assert qparams.scales.shape == ()
    assert qparams.zero_points.shape == ()
import torch
from transformers import AutoConfig

from lmdeploy.utils import FlattenedTensorBucket, _get_and_verify_max_len


def test_flattened_tensor_bucket_preallocated_buffer():
    """Test FlattenedTensorBucket with preallocated buffer."""
    if not torch.cuda.is_available():
        print('CUDA not available, skipping test')
        return

    # Create test tensors on CUDA
    tensor1 = torch.randn(10, 10, dtype=torch.float32, device='cuda')  # 100 elements
    tensor2 = torch.randn(5, 20, dtype=torch.float32, device='cuda')  # 100 elements
    named_tensors = [('tensor1', tensor1), ('tensor2', tensor2)]

    # Test 1: Without preallocated buffer (original behavior)
    bucket1 = FlattenedTensorBucket(named_tensors=named_tensors)
    reconstructed = bucket1.reconstruct_tensors()
    assert len(reconstructed) == 2
    # reconstruct_tensors returns List[Tuple[str, torch.Tensor]]
    reconstructed_dict = dict(reconstructed)
    assert torch.allclose(reconstructed_dict['tensor1'], tensor1)
    assert torch.allclose(reconstructed_dict['tensor2'], tensor2)

    # Test 2: With valid preallocated buffer
    preallocated = torch.empty(200, dtype=torch.float32, device='cuda')
    bucket2 = FlattenedTensorBucket(named_tensors=named_tensors, flattened_tensor=preallocated)
    assert bucket2.flattened_tensor is preallocated  # Should use the same tensor

    # Test 3: With preallocated buffer larger than needed
    preallocated_large = torch.empty(500, dtype=torch.float32, device='cuda')
    bucket3 = FlattenedTensorBucket(named_tensors=named_tensors, flattened_tensor=preallocated_large)
    assert bucket3.flattened_tensor is preallocated_large

    # Test 4: Error case - buffer too small
    preallocated_small = torch.empty(50, dtype=torch.float32, device='cuda')  # Only 50 elements, need 200
    try:
        FlattenedTensorBucket(named_tensors=named_tensors, flattened_tensor=preallocated_small)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'smaller than required numel' in str(e)

    # Test 5: Error case - wrong dtype
    preallocated_wrong_dtype = torch.empty(200, dtype=torch.float64, device='cuda')
    try:
        FlattenedTensorBucket(named_tensors=named_tensors, flattened_tensor=preallocated_wrong_dtype)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'dtype' in str(e)

    # Test 6: Error case - wrong device (CPU buffer for CUDA tensors)
    preallocated_cpu = torch.empty(200, dtype=torch.float32, device='cpu')
    try:
        FlattenedTensorBucket(named_tensors=named_tensors, flattened_tensor=preallocated_cpu)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'device' in str(e)

    # Test 7: Error case - non-contiguous tensor
    preallocated_non_contig = torch.empty(400, dtype=torch.float32, device='cuda')[::2]  # Strided view
    try:
        FlattenedTensorBucket(named_tensors=named_tensors, flattened_tensor=preallocated_non_contig)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'contiguous' in str(e)

    # Test 8: Error case - not 1-D tensor
    preallocated_2d = torch.empty(10, 20, dtype=torch.float32, device='cuda')
    try:
        FlattenedTensorBucket(named_tensors=named_tensors, flattened_tensor=preallocated_2d)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert '1-D tensor' in str(e)


def test_get_and_verify_max_len():
    # with PretrainedConfig
    config = AutoConfig.from_pretrained('OpenGVLab/InternVL-Chat-V1-5-AWQ', trust_remote_code=True)
    assert (_get_and_verify_max_len(config, None) == 32768)
    assert (_get_and_verify_max_len(config, 1024) == 1024)
    assert (_get_and_verify_max_len(config, 102400) == 102400)

    # with PretrainedConfig
    config = AutoConfig.from_pretrained('internlm/internlm2-chat-7b', trust_remote_code=True)
    assert (_get_and_verify_max_len(config, None) == 32768)
    assert (_get_and_verify_max_len(config, 1024) == 1024)
    assert (_get_and_verify_max_len(config, 102400) == 102400)

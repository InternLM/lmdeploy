"""
Simple test script to verify LMDeploy installation
"""
import sys

def test_imports():
    """Test basic imports"""
    print("=" * 60)
    print("Testing LMDeploy Installation")
    print("=" * 60)

    # Test PyTorch
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"     CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"[FAIL] PyTorch import failed: {e}")
        return False

    # Test LMDeploy
    try:
        from lmdeploy import pipeline
        print(f"[OK] LMDeploy pipeline imported successfully")
    except Exception as e:
        print(f"[FAIL] LMDeploy import failed: {e}")
        return False

    # Test transformers
    try:
        import transformers
        print(f"[OK] Transformers version: {transformers.__version__}")
    except Exception as e:
        print(f"[FAIL] Transformers import failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("All imports successful!")
    print("=" * 60)
    return True


def test_pipeline_creation():
    """Test creating a pipeline (without loading a model)"""
    print("\nTesting pipeline configuration...")

    try:
        from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
        from lmdeploy.messages import QuantPolicy
        print("[OK] CacheConfig and SchedulerConfig imported")

        # Create simple configs with correct parameters
        cache_config = CacheConfig(
            max_batches=10,
            block_size=16,
            num_cpu_blocks=100,
            num_gpu_blocks=100,
        )
        print(f"[OK] CacheConfig created: block_size={cache_config.block_size}")

        scheduler_config = SchedulerConfig(max_batches=10, max_session_len=2048)
        print(f"[OK] SchedulerConfig created: max_batches={scheduler_config.max_batches}")

        return True
    except Exception as e:
        print(f"[FAIL] Pipeline configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_imports()
    if success:
        success = test_pipeline_creation()

    if success:
        print("\n" + "=" * 60)
        print("LMDeploy installation verified successfully!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n[ERROR] Some tests failed")
        sys.exit(1)

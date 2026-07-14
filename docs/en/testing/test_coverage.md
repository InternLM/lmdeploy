# Testing and Coverage Guide

This guide explains how to run tests and measure code coverage for LMDeploy.

## Running Tests

### Basic Test Execution

Run all tests:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_lmdeploy/test_model.py
```

Run tests with verbose output:

```bash
pytest tests/ -v
```

### Using the Test Runner Script

The project includes a convenient test runner script with coverage support:

**Linux/macOS:**
```bash
./scripts/run_tests.sh
```

**Windows:**
```bash
scripts\run_tests.bat
```

### Customizing Test Runs

Environment variables allow customization:

```bash
# Disable coverage
COVERAGE=0 ./scripts/run_tests.sh

# Change coverage threshold
THRESHOLD=60 ./scripts/run_tests.sh

# Run specific test path
TEST_PATH=tests/pytorch ./scripts/run_tests.sh

# Disable parallel execution
PARALLEL=0 ./scripts/run_tests.sh

# Change number of workers
NUM_WORKERS=8 ./scripts/run_tests.sh
```

You can combine multiple options:

```bash
COVERAGE=1 THRESHOLD=50 TEST_PATH=tests/test_lmdeploy ./scripts/run_tests.sh
```

## Coverage Configuration

### Configuration Files

Coverage is configured in two places:

1. **pyproject.toml** - Modern configuration (preferred)
2. **.coveragerc** - Backward-compatible configuration

### Coverage Thresholds

The current minimum coverage threshold is **40%**. This is intentionally set low to accommodate:

- C++/CUDA code in `src/turbomind/` (not measured)
- Test files themselves (excluded)
- Third-party code (excluded)

To increase the threshold as coverage improves, edit `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 60  # Increase this value
```

Or set via environment variable:

```bash
THRESHOLD=60 ./scripts/run_tests.sh
```

### Excluded Paths

The following paths are excluded from coverage:

- `*/tests/*` - Test files
- `*/test_*` - Test modules
- `*/__pycache__/*` - Python cache
- `*/third_party/*` - Third-party code
- `*/src/turbomind/*` - C++/CUDA code

### Excluded Code Patterns

These patterns are automatically excluded from coverage:

- `pragma: no cover` comments
- `__repr__` methods
- Debug-only code (`if self.debug:`)
- `NotImplementedError` raises
- Abstract methods
- Protocol classes

## Coverage Reports

### Terminal Report

The default terminal report shows missing lines:

```bash
pytest --cov=lmdeploy --cov-report=term-missing
```

Example output:
```
Name                                      Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------
lmdeploy/api.py                              45      3    93%   12-14
lmdeploy/messages.py                        120     15    88%   45-50, 78-82
lmdeploy/pytorch/config.py                   85      8    91%   23-25, 67
-----------------------------------------------------------------------
TOTAL                                      2500    250    90%
```

### HTML Report

Generate an interactive HTML report:

```bash
pytest --cov=lmdeploy --cov-report=html
```

Open the report:

```bash
# Linux/macOS
open coverage_html_report/index.html

# Windows
start coverage_html_report\index.html
```

The HTML report provides:
- File-by-file coverage breakdown
- Line-by-line coverage visualization
- Click-to-view source code
- Missing lines highlighted in red

### XML Report

Generate XML report for CI/CD integration:

```bash
pytest --cov=lmdeploy --cov-report=xml
```

The `coverage.xml` file follows the Cobertura format, compatible with:
- GitHub Actions
- GitLab CI
- Jenkins
- Codecov
- Coveralls

## Test Markers

LMDeploy uses pytest markers to categorize tests:

| Marker | Description | Usage |
|--------|-------------|-------|
| `slow` | Long-running tests | `pytest -m "not slow"` |
| `gpu` | Requires GPU | `pytest -m gpu` |
| `multi_gpu` | Requires multiple GPUs | `pytest -m multi_gpu` |
| `vlm` | Vision-language model tests | `pytest -m vlm` |
| `quantization` | Quantization tests | `pytest -m quantization` |

### Running Specific Test Categories

Run only GPU tests:
```bash
pytest -m gpu
```

Skip slow tests:
```bash
pytest -m "not slow"
```

Run VLM and quantization tests:
```bash
pytest -m "vlm or quantization"
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements/test.txt
        pip install -e .

    - name: Run tests with coverage
      run: |
        pytest tests/ \
          --cov=lmdeploy \
          --cov-report=xml \
          --cov-report=term-missing \
          -m "not slow and not multi_gpu"

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

### GitLab CI Example

```yaml
test:
  stage: test
  script:
    - pip install -r requirements/test.txt
    - pip install -e .
    - pytest tests/ --cov=lmdeploy --cov-report=xml -m "not slow"
  coverage: '/TOTAL.*?(\d+%)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
```

## Improving Coverage

### Tips for Writing Testable Code

1. **Keep functions small and focused**
   ```python
   # Good: Easy to test
   def calculate_attention(q, k, v):
       scores = q @ k.T
       weights = softmax(scores)
       return weights @ v

   # Bad: Hard to test
   def process_model(data):
       # 100 lines of mixed logic
   ```

2. **Use dependency injection**
   ```python
   # Good: Testable
   class ModelAgent:
       def __init__(self, scheduler=None):
           self.scheduler = scheduler or DefaultScheduler()

   # Bad: Hard to mock
   class ModelAgent:
       def __init__(self):
           self.scheduler = DefaultScheduler()  # Hard-coded
   ```

3. **Separate pure logic from I/O**
   ```python
   # Good: Pure function
   def format_response(text: str) -> dict:
       return {"text": text, "length": len(text)}

   # Good: I/O wrapper
   def generate_and_format(model, prompt: str) -> dict:
       text = model.generate(prompt)  # I/O
       return format_response(text)   # Pure
   ```

### Identifying Coverage Gaps

Use the HTML report to find untested code:

1. Open `coverage_html_report/index.html`
2. Sort by "Missing" column
3. Focus on files with low coverage
4. Add tests for critical paths

## Practical Test Examples

This section provides concrete examples of writing tests for LMDeploy modules.

### Example 1: Testing a Simple Configuration Class

Here's how to write tests for a Pydantic configuration class (from `tests/pytorch/disagg/test_config.py`):

```python
import pytest
from lmdeploy.pytorch.disagg.config import DistServeRDMAConfig, RDMALinkType

class TestDistServeRDMAConfig:
    """Tests for DistServeRDMAConfig."""

    def test_default_with_gdr_is_true(self):
        """Test that with_gdr defaults to True."""
        config = DistServeRDMAConfig()
        assert config.with_gdr is True

    def test_default_link_type_is_roce(self):
        """Test that link_type defaults to RoCE."""
        config = DistServeRDMAConfig()
        assert config.link_type == RDMALinkType.RoCE

    def test_can_set_custom_values(self):
        """Test setting custom values."""
        config = DistServeRDMAConfig(
            with_gdr=False,
            link_type=RDMALinkType.IB
        )
        assert config.with_gdr is False
        assert config.link_type == RDMALinkType.IB

    def test_validation_rejects_invalid_types(self):
        """Test that invalid types are rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DistServeRDMAConfig(with_gdr="invalid")
```

**Key takeaways:**
- Test default values
- Test custom value assignment
- Test validation behavior (Pydantic raises `ValidationError`)
- Use descriptive test names and docstrings

### Example 2: Testing Enum Classes

Testing enum classes is straightforward but important (from `tests/pytorch/disagg/test_config.py`):

```python
from lmdeploy.pytorch.disagg.config import ServingStrategy

class TestServingStrategy:
    """Tests for ServingStrategy enum."""

    def test_serving_strategy_has_hybrid(self):
        """Test that Hybrid strategy exists."""
        assert hasattr(ServingStrategy, 'Hybrid')
        assert ServingStrategy.Hybrid.name == 'Hybrid'

    def test_serving_strategy_values_are_unique(self):
        """Test that enum values are unique."""
        values = [strategy.value for strategy in ServingStrategy]
        assert len(values) == len(set(values))

    def test_serving_strategy_count(self):
        """Test that there are exactly 2 strategies."""
        assert len(list(ServingStrategy)) == 2
```

**Key takeaways:**
- Verify enum members exist
- Check uniqueness of values
- Verify expected count

### Example 3: Testing Complex Configuration with Multiple Fields

For configurations with many required fields (from `tests/pytorch/disagg/test_config.py`):

```python
import pytest
from lmdeploy.pytorch.disagg.config import DistServeEngineConfig

class TestDistServeEngineConfig:
    """Tests for DistServeEngineConfig."""

    def test_basic_config_creation(self):
        """Test basic engine config creation."""
        config = DistServeEngineConfig(
            tp_size=1,
            ep_size=1,
            dp_size=1,
            pp_size=None,
            dp_rank=0,
            block_size=16,
            num_cpu_blocks=100,
            num_gpu_blocks=1000,
        )
        assert config.tp_size == 1
        assert config.ep_size == 1
        assert config.dp_size == 1
        assert config.pp_size is None
        assert config.dp_rank == 0
        assert config.block_size == 16
        assert config.num_cpu_blocks == 100
        assert config.num_gpu_blocks == 1000

    def test_config_requires_all_fields(self):
        """Test that all required fields must be provided."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DistServeEngineConfig(
                tp_size=1,
                # Missing other required fields
            )

    def test_config_accepts_different_parallel_sizes(self):
        """Test config with various parallel sizes."""
        configs = [
            (1, 1, 1, None),   # Single GPU
            (2, 1, 1, None),   # TP=2
            (1, 1, 2, None),   # DP=2
            (2, 1, 2, 2),      # TP=2, DP=2, PP=2
        ]

        for tp, ep, dp, pp in configs:
            config = DistServeEngineConfig(
                tp_size=tp,
                ep_size=ep,
                dp_size=dp,
                pp_size=pp,
                dp_rank=0,
                block_size=16,
                num_cpu_blocks=100,
                num_gpu_blocks=1000,
            )
            assert config.tp_size == tp
            assert config.ep_size == ep
```

**Key takeaways:**
- Test all fields individually
- Test validation (missing required fields)
- Use parameterized testing for multiple scenarios
- Include comments explaining test cases

### Example 4: Testing Inheritance

When a class inherits from another (from `tests/pytorch/disagg/test_config.py`):

```python
from lmdeploy.pytorch.disagg.config import (
    DistServeEngineConfig,
    MooncakeEngineConfig,
)

class TestMooncakeEngineConfig:
    """Tests for MooncakeEngineConfig."""

    def test_mooncake_config_inherits_from_distserve(self):
        """Test that MooncakeEngineConfig inherits from DistServeEngineConfig."""
        assert issubclass(MooncakeEngineConfig, DistServeEngineConfig)

    def test_mooncake_config_has_same_fields_as_parent(self):
        """Test that Mooncake config has same fields as parent."""
        parent_config = DistServeEngineConfig(
            tp_size=1, ep_size=1, dp_size=1, pp_size=None,
            dp_rank=0, block_size=16,
            num_cpu_blocks=100, num_gpu_blocks=1000,
        )
        mooncake_config = MooncakeEngineConfig(
            tp_size=1, ep_size=1, dp_size=1, pp_size=None,
            dp_rank=0, block_size=16,
            num_cpu_blocks=100, num_gpu_blocks=1000,
        )

        assert mooncake_config.tp_size == parent_config.tp_size
        assert mooncake_config.ep_size == parent_config.ep_size
        # ... verify all fields match
```

**Key takeaways:**
- Verify inheritance relationship with `issubclass()`
- Test that child class has same behavior as parent
- Ensure no regression in inherited functionality

### Example 5: Integration Tests

Testing how multiple components work together (from `tests/pytorch/disagg/test_config.py`):

```python
def test_multiple_engine_configs_for_pd_pair(self):
    """Test creating configs for prefill-decode pair."""
    prefill_config = DistServeEngineConfig(
        tp_size=4, ep_size=1, dp_size=1, pp_size=None,
        dp_rank=0, block_size=16,
        num_cpu_blocks=100, num_gpu_blocks=1000,
    )

    decode_config = DistServeEngineConfig(
        tp_size=2, ep_size=1, dp_size=2, pp_size=2,
        dp_rank=0, block_size=16,
        num_cpu_blocks=200, num_gpu_blocks=2000,
    )

    # Prefill uses TP=4 for computation
    assert prefill_config.tp_size == 4
    assert prefill_config.pp_size is None

    # Decode uses TP=2, PP=2 for latency optimization
    assert decode_config.tp_size == 2
    assert decode_config.pp_size == 2
```

**Key takeaways:**
- Test realistic usage scenarios
- Document the reasoning behind configurations
- Verify interactions between components

## Common Testing Patterns

### Pattern 1: Arrange-Act-Assert

```python
def test_example():
    # Arrange: Set up test data
    config = DistServeRDMAConfig(with_gdr=False)

    # Act: Perform the action being tested
    result = config.with_gdr

    # Assert: Verify the result
    assert result is False
```

### Pattern 2: Test Edge Cases

```python
def test_dp_rank_can_be_nonzero(self):
    """Test that dp_rank can be non-zero."""
    for rank in [0, 1, 2, 3]:
        config = DistServeEngineConfig(
            tp_size=1, ep_size=1, dp_size=4, pp_size=None,
            dp_rank=rank, block_size=16,
            num_cpu_blocks=100, num_gpu_blocks=1000,
        )
        assert config.dp_rank == rank
```

### Pattern 3: Test Error Conditions

```python
def test_validation_rejects_invalid_types(self):
    """Test that invalid types are rejected."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        DistServeRDMAConfig(with_gdr="invalid")
```

## Testing Best Practices Checklist

- [ ] **Test names are descriptive**: `test_default_with_gdr_is_true` not `test_1`
- [ ] **Each test has a docstring**: Explains what is being tested
- [ ] **One assertion per concept**: Don't test too many things in one test
- [ ] **Test both success and failure cases**: Happy path + error conditions
- [ ] **Use fixtures for common setup**: Avoid code duplication
- [ ] **Mock external dependencies**: Don't rely on network/files/GPU
- [ ] **Keep tests fast**: Unit tests should run in milliseconds
- [ ] **Make tests deterministic**: No random values or timing-dependent logic
- [ ] **Test edge cases**: Empty inputs, boundary values, None values
- [ ] **Update tests when code changes**: Keep tests in sync with implementation

### Coverage Goals

Recommended coverage targets by module type:

| Module Type | Target Coverage |
|-------------|----------------|
| Core API | 80%+ |
| Configuration | 90%+ |
| Utilities | 70%+ |
| Model implementations | 60%+ |
| Kernels (Python wrappers) | 50%+ |
| CLI tools | 40%+ |

## Troubleshooting

### Issue: Coverage data not collected

**Solution:** Ensure `pytest-cov` is installed:
```bash
pip install pytest-cov
```

### Issue: Low coverage due to CUDA code

**Note:** C++/CUDA code in `src/turbomind/` cannot be measured by Python coverage tools. The reported coverage only includes Python code.

### Issue: Tests running too slowly

**Solution:** Use parallel execution:
```bash
pytest -n auto  # Auto-detect CPU cores
# or
NUM_WORKERS=8 ./scripts/run_tests.sh
```

### Issue: Coverage threshold too strict

**Solution:** Adjust threshold temporarily:
```bash
THRESHOLD=30 ./scripts/run_tests.sh
```

Then gradually increase as you add more tests.

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [Testing best practices](https://docs.python-guide.org/writing/tests/)

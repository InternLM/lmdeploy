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

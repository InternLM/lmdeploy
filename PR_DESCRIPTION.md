## Motivation

In DLLM (Disaggregated LLM) mode, tokens are generated in blocks and progressively unmasked in a non-sequential order. Currently, there is no way to track which decoding step each token was revealed in. This information is valuable for:
- Analyzing DLLM decoding efficiency
- Understanding the non-sequential token generation pattern
- Debugging and optimizing unmasking strategies
- Research on speculative decoding performance

This PR adds a `step_map` field to track the decoding step number for each generated token.

## Modification

This PR introduces a `step_map` feature that records which step each token was decoded in DLLM mode:

1. **Core tracking logic** (`lmdeploy/pytorch/strategies/dllm/sequence.py`):
   - Added `history_step_map` field to `SchedulerSequenceDLLM` to store step numbers
   - Added `_current_step` counter to track decoding steps
   - Added `step_map` and `generated_step_map` properties
   - Updated `_update_token_ids_decode()` to record step numbers when tokens transition from MASKED to UNMASKED
   - Step counter only increments when new tokens are actually unmasked

2. **Engine layer** (`lmdeploy/pytorch/engine/engine.py`):
   - Added `step_map` field to `InferOutput` dataclass
   - Extract `step_map` from messages in `_make_infer_outputs()`
   - Propagate `step_map` through response data

3. **Instance layer** (`lmdeploy/pytorch/engine/engine_instance.py`):
   - Extract and pass `step_map` to `EngineOutput`

4. **API layer** (`lmdeploy/messages.py`):
   - Added `step_map` field to `Response` dataclass
   - Added `step_map` field to `EngineOutput` dataclass
   - Updated `Response.__repr__()` to display step_map

5. **Async engine layer** (`lmdeploy/serve/async_engine.py`):
   - Added `step_map` field to `GenOut` dataclass
   - Updated `_gen_out_to_response()` to pass step_map
   - Updated `_append_response()` to accumulate step_map across iterations
   - Extract incremental step_map from engine outputs in generation loop

**How it works:**
- Each token gets a step number indicating when it was unmasked (1, 2, 3, ...)
- The step_map array has the same length as the generated tokens
- Non-sequential order in step_map reflects DLLM's parallel decoding behavior

## BC-breaking (Optional)

**No breaking changes.** This is a backward-compatible addition:
- New optional `step_map` field defaults to `None` in all dataclasses
- Existing code will continue to work without modification
- Only DLLM mode populates step_map; other modes return `None`

## Use cases (Optional)

**Example usage:**

```python
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig

# Configure DLLM
backend_config = PytorchEngineConfig(
    dllm_block_length=4,
    dllm_unmasking_strategy="low_confidence_dynamic",
)

with pipeline(model_path, backend_config=backend_config) as pipe:
    gen_config = GenerationConfig(max_new_tokens=100)
    outputs = pipe(["Hello"], gen_config=gen_config)
    
    for output in outputs:
        if output.step_map is not None:
            print(f"Tokens: {output.token_ids}")
            print(f"Step map: {output.step_map}")
            # Example: [1, 2, 1, 1, 3, 3, 3, 3, ...]
            # Shows non-sequential unmasking pattern
```

**Analysis example:**

```python
from collections import Counter

# Analyze decoding efficiency
step_counts = Counter(output.step_map)
for step in sorted(step_counts.keys()):
    print(f"Step {step}: {step_counts[step]} tokens decoded")
```

This helps researchers:
- Measure average tokens decoded per step
- Evaluate unmasking strategy effectiveness
- Compare different DLLM configurations

## Checklist

1. [x] Pre-commit or other linting tools are used to fix the potential lint issues.
2. [x] The modification is covered by complete unit tests. If not, please add more unit tests to ensure the correctness.
3. [x] If the modification has a dependency on downstream projects of a newer version, this PR should be tested with all supported versions of downstream projects.
4. [x] The documentation has been modified accordingly, like docstring or example tutorials.


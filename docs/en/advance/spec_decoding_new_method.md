# Add a New Speculative Decoding Method

This guide documents the extension contract that lets you plug a new draft-token proposer into
the PyTorch engine's speculative decoding pipeline without modifying the engine.

For an overview of the feature and ready-to-run examples of the four shipped methods
(`eagle`, `eagle3`, `deepseek_mtp`, `qwen3_5_mtp`), see [Speculative Decoding](./spec_decoding.md).

> \[!NOTE\]
> Speculative decoding is experimental in lmdeploy and the proposer contract may evolve.

## The plug-in contract

A speculative decoding "method" is a subclass of `BaseSpecProposer` registered into the
`SPEC_PROPOSERS` registry under a string name. At runtime, when you pass
`SpeculativeConfig(method='<name>', ...)` to the pipeline, the engine looks up `<name>` in
`SPEC_PROPOSERS` and instantiates the matching proposer.

There are three pieces:

1. **Registry** — `lmdeploy.pytorch.spec_decode.proposers.base.SPEC_PROPOSERS`, an `mmengine`
   `Registry`.
2. **Base class** — `BaseSpecProposer`, which encapsulates the draft-model build, forward, and
   decoding-input update logic that most methods can reuse as-is.
3. **`method` string** — what the user types as `SpeculativeConfig.method`. It must match the
   `name=` you pass to `@SPEC_PROPOSERS.register_module`.

The entry point that ties them together is `build_specdecode_proposer` in
`lmdeploy/pytorch/spec_decode/proposers/base.py`:

```python
def build_specdecode_proposer(specdecode_config: SpecDecodeConfig, device: str = 'cuda'):
    method = specdecode_config.method
    if method in SPEC_PROPOSERS.module_dict:
        spec_cls = SPEC_PROPOSERS.module_dict[method]
        return spec_cls(specdecode_config, device=device)
    raise ValueError(f'{method} not found in {SPEC_PROPOSERS.module_dict.keys()}')
```

If your proposer is not imported when this function runs, the lookup fails — which is why new
proposers must also be imported from `lmdeploy/pytorch/spec_decode/proposers/__init__.py`.

## What `BaseSpecProposer` already does

Read `lmdeploy/pytorch/spec_decode/proposers/base.py` before writing your own. The base class
already provides:

- `__init__` — stores config, sets `self.num_speculative_tokens`.
- `build_model` — patches and loads draft weights via the standard PyTorch engine path.
- `_forward` — runs one draft forward under `torch.inference_mode()` and the correct CUDA stream.
- `update_inputs_decoding` — rolls `ModelInputs` forward by one step (history lengths, KV
  positions, MRoPE indices, rejected-token bookkeeping).
- `embed_input_ids` / `get_logits` — fall back to the target model's embeddings / LM head if the
  draft does not expose its own.
- `get_target_hidden_size(model_config)` — defaults to the target's `hidden_size`. Override if
  your head expects a wider feature (Eagle3 concatenates three layers).

The one method you almost always need to implement is `get_outputs`.

## Minimal new proposer

```python
# lmdeploy/pytorch/spec_decode/proposers/my_method.py

import torch

from ...model_inputs import ModelInputs
from ...strategies.ar_spec.model_agent import ARSpecExtraInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer


@SPEC_PROPOSERS.register_module(name='my_method')
class MyMethod(BaseSpecProposer):
    """My speculative decoding proposer."""

    def get_outputs(self,
                    model_outputs: dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ARSpecExtraInputs = None):
        hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs['model_metas']
        if extra_inputs is not None:
            hidden_states = hidden_states[:, extra_inputs.last_token_indices]
        logits = self.get_logits(hidden_states)[0]
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        return draft_token_ids, model_metas, hidden_states
```

Then expose it so the registry sees it at import time:

```python
# lmdeploy/pytorch/spec_decode/proposers/__init__.py
from .my_method import MyMethod  # noqa F401
```

Users can then select your method:

```python
from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig

spec_cfg = SpeculativeConfig(method='my_method', num_speculative_tokens=3,
                             model='your-org/your-draft-checkpoint')
pipe = pipeline('meta-llama/Llama-3.1-8B-Instruct',
                backend_config=PytorchEngineConfig(max_batch_size=128),
                speculative_config=spec_cfg)
```

## What `get_outputs` must return

| Position | Type           | Meaning                                                                            |
| -------- | -------------- | ---------------------------------------------------------------------------------- |
| `[0]`    | `torch.Tensor` | Proposed draft token ids, shape `[batch, 1]` per step.                             |
| `[1]`    | `list[Any]`    | Per-sequence `model_metas` carried forward (cache / state hooks).                  |
| `[2]`    | `torch.Tensor` | `target_hidden_states` fed back into the next prefill (used by Eagle-style heads). |

`DeepseekMTP.get_outputs` is the canonical reference; `Eagle3.get_outputs` is the variant that
performs a `draft_id → target_id` remap.

## When to override `build_model`

If your method must share weights with the target model — typically token embeddings or a special
LM head — override `build_model` and patch the draft module after the base class finishes loading.
Two real examples ship today:

`Qwen3_5MTP` reuses the target model's embedding table to keep the draft compact:

```python
@SPEC_PROPOSERS.register_module(name='qwen3_5_mtp')
class Qwen3_5MTP(DeepseekMTP):

    def build_model(self, empty_init, target_model=None, build_model_ctx=None):
        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)
        self.model.set_input_embeddings(target_model.get_input_embeddings())
```

`Eagle3` swaps in the target embeddings only when the checkpoint was trained without its own, and
additionally overrides `get_target_hidden_size`:

```python
@SPEC_PROPOSERS.register_module(name='eagle3')
class Eagle3(DeepseekMTP):

    def build_model(self, empty_init, target_model=None, build_model_ctx=None):
        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)
        self.draft_id_to_target_id = self.model.draft_id_to_target_id
        if not self.model.include_embed_tokens:
            del self.model.model.embed_tokens
            self.model.model.embed_tokens = target_model.get_input_embeddings()

    def get_target_hidden_size(self, model_config):
        hf_config = self.specdecode_config.model_config.hf_config
        hidden_size = getattr(hf_config, 'target_hidden_size', hf_config.hidden_size)
        return hidden_size * 3
```

## Wire up the draft model architecture

`BaseSpecProposer.build_model` reuses the PyTorch engine's standard model-loading path, so the
draft checkpoint must be discoverable by the engine in the same way any other model is. In
practice this means two more touch-points outside `spec_decode/`:

1. **`lmdeploy/pytorch/configurations/`** — add (or extend) an `AutoModelConfigBuilder` so the
   engine can produce a `ModelConfig` for the draft. Each file under this directory subclasses
   `AutoModelConfigBuilder` (see `configurations/builder.py`) and is auto-registered at import
   time by `configurations/__init__.py`, which walks the package with `pkgutil.walk_packages`.
   The base builder picks the first subclass whose `condition(hf_config)` returns `True`, falling
   back to `DefaultModelConfigBuilder`. The `build(...)` classmethod receives `is_draft_model`
   and `spec_method` kwargs — use them to override the draft-side fields. Existing references:

   - `configurations/deepseek_v2.py` (`DeepseekV2ModelConfigBuilder`) shows the minimal
     `deepseek_mtp` wiring: switch `hf_config.architectures[0]` to `'DeepseekMTPModel'` and set
     `model_paradigm = 'ar_spec'` when `is_draft_model=True`.
   - `configurations/qwen3_5.py` (`Qwen3_5ModelConfigBuilder`) shows a `qwen3_5_mtp` variant
     that additionally adjusts `num_layers`, `states_shapes`, and `conv_kernel_size` for the
     draft model.
   - `configurations/llama.py` (`LlamaModelConfigBuilder`) shows the eagle/eagle3 pattern:
     prepending the `spec_method` to the original architecture name to produce a unique draft
     architecture string (e.g. `Eagle3LlamaForCausalLM`).

2. **`lmdeploy/pytorch/models/`** — add the draft model definition (an `nn.Module` subclass,
   typically also mixing in `CudaGraphMixin`) and register the architecture string from step 1
   in `lmdeploy/pytorch/models/module_map.py`. The engine's patcher (`models/patch.py`)
   resolves the architecture string via the `MODULE_MAP` dictionary to a fully-qualified class
   path. For example, after the qwen3.5 MTP draft was added, `module_map.py` gained:

   ```python
   MODULE_MAP.update({
       'Qwen3_5MTPModel': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.qwen3_5_mtp.Qwen3_5MTPModel',
   })
   ```

   Existing draft-model implementations worth using as templates:

   - `models/deepseek_mtp.py` — the canonical MTP draft module.
   - `models/qwen3_5_mtp.py` — `Qwen3_5MTPModel`, extends the deepseek MTP pattern.
   - `models/llama_eagle.py` / `models/llama_eagle3.py` — Eagle-family draft heads on top of Llama.
   - `models/glm4moe_mtp.py` — `Glm4MoeMTPModel`.

If you skip either of these, `build_specdecode_proposer` will still be able to instantiate your
proposer, but `BaseSpecProposer.build_model` will fail to materialise the draft network because
neither `AutoModelConfigBuilder` nor `MODULE_MAP` knows how to handle the draft architecture.

## Checklist

1. The proposer class is decorated with `@SPEC_PROPOSERS.register_module(name='<unique-method-name>')`
   and the name is not already in `SPEC_PROPOSERS.module_dict`.
2. The proposer module is imported from `lmdeploy/pytorch/spec_decode/proposers/__init__.py` so
   the decorator runs at engine startup.
3. `get_outputs` returns the three-tuple above; tensor shapes match existing proposers.
4. If your draft model needs target weights, override `build_model` rather than rewriting patching.
5. A config builder under `lmdeploy/pytorch/configurations/` recognises the draft `hf_config`
   (via `condition`) and sets `model_paradigm = 'ar_spec'` / the right `architectures[0]` when
   `is_draft_model=True`.
6. The draft architecture string is registered in `lmdeploy/pytorch/models/module_map.py` and
   resolves to a model class under `lmdeploy/pytorch/models/`.
7. Run end-to-end with `SpeculativeConfig(method='<name>', ...)` and confirm draft tokens are
   being accepted by the target model.

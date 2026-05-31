# Adding a New Speculative Decoding Method

The LMDeploy PyTorch engine provides a plugin system for speculative decoding
proposers. A proposer is the component that generates draft tokens that the
target model then verifies. This page explains how to add a new method.

If you just want to *use* the existing shipped methods (`eagle`, `eagle3`,
`deepseek_mtp`, `qwen3_5_mtp`), see the
[Speculative Decoding](./spec_decoding.md) user guide instead.

## Overview of the Plugin System

Three symbols govern every speculative decoding method:

| Symbol | Location | Purpose |
|---|---|---|
| `SPEC_PROPOSERS` | `lmdeploy/pytorch/spec_decode/proposers/base.py` | The [MMEngine Registry](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html) that maps method names to proposer classes |
| `BaseSpecProposer` | Same module | Base class that implements weight loading, draft forward, logit extraction, input management, and fallback logic |
| `method` string | The value you pass in `SpeculativeConfig(method='...')` or `--speculative-algorithm` on the CLI | Must match the `name` argument used in the `@register_module` decorator |

The entry point is `build_specdecode_proposer` (same module):

```python
def build_specdecode_proposer(specdecode_config, device='cuda'):
    method = specdecode_config.method
    if method in SPEC_PROPOSERS.module_dict:
        spec_cls = SPEC_PROPOSERS.module_dict[method]
        obj = spec_cls(specdecode_config, device=device)
        return obj
    raise ValueError(...)
```

It looks up the method string inside `SPEC_PROPOSERS.module_dict`, instantiates
the registered class, and returns it. For this lookup to succeed, your class
must be imported before `build_specdecode_proposer` runs. This is why every new
proposer needs an import line in `proposers/__init__.py`.

## What BaseSpecProposer Provides

Subclassing `BaseSpecProposer` gives you the following methods for free.
In simple cases you only need to override `get_outputs`.

| Method / Attribute | What it does |
|---|---|
| `build_model(...)` | Loads the draft model from `specdecode_config.model`, builds patched model, loads weights. Override only if you need custom embedding sharing or model surgery. |
| `_forward(model_inputs)` | Runs a single forward pass of the draft model with `draft_model_forward`. |
| `update_inputs_decoding(...)` | Clones the model inputs and updates history lengths, position ids, and related metadata for the next decoding step. |
| `get_logits(hidden_states)` | Extracts logits from draft hidden states (falls back to the target model's `get_logits`). |
| `embed_input_ids(input_ids)` | Embeds token ids via the draft or target model's `get_input_embeddings`. |
| `get_target_hidden_size(model_config)` | Returns the hidden size expected by the target model. Override if your proposer produces wider or narrower hidden states. |

## Implementing a Custom Proposer

### 1. Create the proposer file

Create a file in `lmdeploy/pytorch/spec_decode/proposers/`. Here is a minimal
skeleton:

```python
# Copyright (c) OpenMMLab. All rights reserved.
# lmdeploy/pytorch/spec_decode/proposers/my_method.py

import torch

from lmdeploy.utils import get_logger

from ...model_inputs import ModelInputs
from ...strategies.base.model_agent import ExtraInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='my_method')
class MyMethod(BaseSpecProposer):

    def get_outputs(self,
                    model_outputs: dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ExtraInputs = None):
        """Extract draft tokens from the model outputs.

        Must return a 3-tuple of:
            (draft_token_ids, model_metas, target_hidden_states)
        """
        # model_outputs contains the draft model's forward return value.
        # The keys depend on your model, but 'hidden_states' and
        # 'model_metas' are always present.
        hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs['model_metas']

        # When extra_inputs is available, only the last token in the
        # sequence is needed for decoding.
        if extra_inputs is not None:
            last_token_loc = extra_inputs.last_token_indices
            hidden_states = hidden_states[:, last_token_loc]
            target_hidden_states = hidden_states
        else:
            target_hidden_states = hidden_states

        # Produce draft token ids from hidden states.
        logits = self.get_logits(hidden_states)[0]
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)

        return draft_token_ids, model_metas, target_hidden_states
```

#### Return contract of `get_outputs`

The method must return a 3-tuple:

1. **draft_token_ids** (`torch.Tensor`): Integer token ids predicted by the
   draft model, shape `(batch_size, num_speculative_tokens)`.
2. **model_metas**: Opaque metadata produced by the model's
   `update_model_metas`. Passed through to the target model's verification
   step.
3. **target_hidden_states** (`torch.Tensor` | `None`): Hidden states that the
   target model can reuse. If your method does not produce shareable hidden
   states, pass `None`.

### 2. Register the module

The `@SPEC_PROPOSERS.register_module(name='my_method')` decorator registers
your class under the string `'my_method'`. This string is what the user passes
in `SpeculativeConfig(method='my_method')` or via the
`--speculative-algorithm my_method` CLI flag.

### 3. Add the import to `__init__.py`

Open `lmdeploy/pytorch/spec_decode/proposers/__init__.py` and add an import for
your new module:

```python
from .my_method import MyMethod  # noqa: F401
```

Without this import, `SPEC_PROPOSERS.module_dict` will not contain your method
at runtime and `build_specdecode_proposer` will raise a `ValueError`.

## When to Override `build_model`

The default `build_model` builds a patched model from the Hugging Face
checkpoint and loads its weights. Override it when your proposer needs to
share or replace the draft model's embedding layer with the target model's.

**Example: sharing target embeddings (Qwen3_5MTP)**

The `Qwen3_5MTP` proposer replaces the draft model's input embeddings with the
target model's to save memory and ensure token embeddings are identical:

```python
def build_model(self, empty_init, target_model=None, build_model_ctx=None):
    super().build_model(empty_init, target_model=target_model,
                        build_model_ctx=build_model_ctx)
    self.model.set_input_embeddings(target_model.get_input_embeddings())
```

**Example: conditional embedding swap and wider hidden size (Eagle3)**

`Eagle3` conditionally deletes the draft model's embedding layer and widens
the hidden size that the target model must accept:

```python
def build_model(self, empty_init, target_model=None, build_model_ctx=None):
    super().build_model(empty_init, target_model=target_model,
                        build_model_ctx=build_model_ctx)
    if not self.model.include_embed_tokens:
        del self.model.model.embed_tokens
        self.model.model.embed_tokens = target_model.get_input_embeddings()

def get_target_hidden_size(self, model_config):
    hf_config = self.specdecode_config.model_config.hf_config
    hidden_size = getattr(hf_config, 'target_hidden_size',
                          hf_config.hidden_size)
    return hidden_size * 3  # Eagle3 concatenates 3x hidden states
```

Override `get_target_hidden_size` whenever your draft model's output dimension
differs from the target model's `hidden_size`.

## Shipping Checklist

1. **Proposer class**: Implement `get_outputs` (and optionally `build_model` /
   `get_target_hidden_size`) in a new file under `proposers/`.
2. **Registration**: The class must carry the
   `@SPEC_PROPOSERS.register_module(name='<your_method>')` decorator.
3. **Import**: Add `from .<your_method> import <YourClass>` to
   `proposers/__init__.py`.
4. **Documentation**: Add a usage example (pipeline + serve) to
   `spec_decoding.md`, mirroring the existing Eagle3 and DeepseekMTP sections.
5. **End-to-end test**: Run a short inference with
   `SpeculativeConfig(method='<your_method>')` and verify that throughput
   improves and generations are valid.

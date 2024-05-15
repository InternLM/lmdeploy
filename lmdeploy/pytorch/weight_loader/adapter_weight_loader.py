# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager


class AdapterWeightLoader:
    """adapter weight loader."""

    def __init__(self, adapter_name: str, adapter_path: str):
        from peft.utils.save_and_load import load_peft_weights
        self._adapter_name = adapter_name
        self._adapter_path = adapter_path
        self._state_dict = load_peft_weights(adapter_path)
        self._prefix = 'base_model.model.'

    def pop(self, key: str):
        """pop weight."""
        key = self._prefix + key
        return self._state_dict.pop(key)

    def get(self, key: str):
        """get weight."""
        key = self._prefix + key
        return self._state_dict.get(key)

    @contextmanager
    def prefix_context(self, mod_name: str):
        """update prefix by mod name."""
        old_prefix = self._prefix
        if len(old_prefix) == 0:
            new_prefix = f'{mod_name}.'
        else:
            new_prefix = f'{old_prefix}{mod_name}.'
        self._prefix = new_prefix
        yield new_prefix
        self._prefix = old_prefix

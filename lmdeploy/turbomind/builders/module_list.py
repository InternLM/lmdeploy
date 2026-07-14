# Copyright (c) OpenMMLab. All rights reserved.
import _turbomind as _tm

from ._base import Builder, BuiltModule

ModuleListConfig = _tm.ModuleListConfig


class ModuleListBuilder(Builder):
    """Builder for ModuleList containers."""

    def __setitem__(self, index: int, value):
        if isinstance(value, Builder):
            raise TypeError(
                f'{type(self).__name__}[{index}]: call .build() first')
        if isinstance(value, BuiltModule):
            if self._built:
                raise RuntimeError(
                    f'{type(self).__name__} is built; '
                    f'cannot set index {index}')
            self._add_child(str(index), value.handles)
            return
        raise TypeError(
            f'{type(self).__name__}[{index}] requires a BuiltModule')

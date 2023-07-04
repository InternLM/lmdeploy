# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Callable


class Observer:
    """The Observer class applies a user-specified function on its inputs and
    stores the results in a buffer.

    Args:
        observe_fn (Callable[..., Any]): The function to apply on inputs.
    """

    def __init__(self, observe_fn: Callable[..., Any]) -> None:
        super().__init__()
        self.fn = observe_fn
        self.buffer = list()
        self.enabled = False

    def enable_observer(self, enabled: bool = True) -> None:
        """Enable or disable the observer.

        Args:
            enabled (bool, optional): Whether to enable the observer.
                Defaults to True.
        """
        self.enabled = enabled

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Apply the observer function on the input if the observer is enabled.

        Args:
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments.
        """
        if self.enabled:
            self.buffer.append(self.fn(*args, **kwds))

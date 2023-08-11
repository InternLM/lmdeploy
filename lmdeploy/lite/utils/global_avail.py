# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Union

from torch import nn


class GlobalAvailMixin:
    """Mixin class to make instances globally available."""

    _instances: Dict[str, Dict[Union[str, nn.Module], 'GlobalAvailMixin']] = {
        'default': {}
    }

    def global_available(self,
                         key: Union[str, nn.Module] = 'default',
                         group: str = 'default') -> None:
        """Make the instance globally available.

        Args:
            key (Union[str, nn.Module], optional): Key to save the instance.
                Defaults to 'default'.
            group (str, optional): Group to save the instance.
                Defaults to 'default'.
        """
        self._save_instance(self, key, group)

    @classmethod
    def _save_instance(cls,
                       instance: 'GlobalAvailMixin',
                       key: Union[str, nn.Module] = 'default',
                       group: str = 'default') -> None:
        """Save the instance.

        Args:
            instance (GlobalAvailMixin): Instance to save.
            key (Union[str, nn.Module], optional): Key to save the instance.
                Defaults to 'default'.
            group (str, optional): Group to save the instance.
                Defaults to 'default'.
        """
        if group not in cls._instances:
            assert isinstance(group, str)
            cls._instances[group] = {}

        cls._instances[group][key] = instance

    @classmethod
    def find(cls,
             key: Union[str, nn.Module] = 'default',
             group: str = 'default') -> Union[None, 'GlobalAvailMixin']:
        """Find an instance by its key and group.

        Args:
            key (Union[str, nn.Module], optional): Key of the instance.
                Defaults to 'default'.
            group (str, optional): Group of the instance.
                Defaults to 'default'.

        Returns:
            Union[None, GlobalAvailMixin]: The found instance, or None if
                it does not exist.
        """
        return cls._instances.get(group, {}).get(key)

    @classmethod
    def find_group(
            cls,
            group: str) -> Dict[Union[str, nn.Module], 'GlobalAvailMixin']:
        """Find all instances in a group.

        Args:
            group (str): Group of the instances.

        Returns:
            Dict[Union[str, nn.Module], GlobalAvailMixin]: All instances in
                the group.
        """
        return cls._instances.get(group, {})

    @classmethod
    def instances(
            cls) -> Dict[str, Dict[Union[str, nn.Module], 'GlobalAvailMixin']]:
        """Get all instances."""
        return cls._instances

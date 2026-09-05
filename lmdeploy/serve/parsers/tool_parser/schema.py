# Copyright (c) OpenMMLab. All rights reserved.
"""Offline JSON Schema validation for tool arguments."""

import json
from functools import lru_cache
from typing import Any

from jsonschema.validators import validator_for
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import specification_with


def create_schema_validator(schema: dict | bool) -> Any:
    """Create a request-local offline validator after cached schema
    preflight."""
    _check_schema(json.dumps(schema, separators=(',', ':')))
    return validator_for(schema)(schema, registry=Registry())


@lru_cache(maxsize=128)
def _check_schema(serialized_schema: str) -> None:
    """Check an immutable schema snapshot and cache only successful results."""
    schema = json.loads(serialized_schema)
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    specification = specification_with(validator_cls.ID_OF(validator_cls.META_SCHEMA))
    resource = specification.create_resource(schema)
    registry = Registry()
    pending = [(resource, registry.resolver_with_root(resource))]
    visited = set()
    while pending:
        resource, resolver = pending.pop()
        if id(resource.contents) in visited:
            continue
        visited.add(id(resource.contents))
        if isinstance(resource.contents, dict):
            for keyword in ('$ref', '$dynamicRef', '$recursiveRef'):
                ref = resource.contents.get(keyword)
                if ref is None:
                    continue
                if not ref.startswith('#'):
                    raise ValueError(f'Only local tool schema references are supported: {ref!r}')
                try:
                    resolved = resolver.lookup(ref)
                except Unresolvable as err:
                    raise ValueError(f'Unresolvable tool schema reference: {ref!r}') from err
                if id(resolved.contents) not in visited:
                    validator_cls.check_schema(resolved.contents)
                    target = Resource.from_contents(resolved.contents, default_specification=specification)
                    pending.append((target, resolved.resolver))
        # Walk schema locations, not arbitrary JSON in defaults, examples or enums.
        pending.extend((child, resolver.in_subresource(child)) for child in resource.subresources())

# Copyright (c) OpenMMLab. All rights reserved.
"""Served-model name parsing and matching helpers."""


def split_model_name(model_name: str) -> tuple[str, list[str]]:
    """Split a served-model name into a primary name and aliases.

    ``model_name`` may be a comma-separated list of names. The first name is
    the primary one, returned by ``/v1/models`` and echoed in responses. The
    remaining names are aliases that are also accepted in requests. Request
    matching against these names is case-insensitive.

    Args:
        model_name (str): the served-model name, possibly comma-separated.

    Returns:
        tuple[str, list[str]]: the primary name and the full list of names.
    """
    names = [name.strip() for name in model_name.split(',') if name.strip()]
    return names[0], names


def name_matches_any(requested: str | None, candidates: list[str]) -> bool:
    """Return whether a requested name matches any candidate.

    Matching is case-insensitive, so downstream clients may use arbitrary casing for the served model, its aliases and
    adapters.
    """
    normalized = (requested or '').lower()
    return any((candidate or '').lower() == normalized for candidate in candidates)


def model_matches(requested: str | None, async_engine) -> bool:
    """Return whether a requested name is the served model or one of its
    aliases."""
    aliases = getattr(async_engine, 'model_aliases', None)
    return name_matches_any(requested, aliases or [async_engine.model_name])


def model_or_adapter_matches(requested: str | None, async_engine, adapters: list[str] | None) -> bool:
    """Return whether a requested name matches the model, an alias or an
    adapter.

    Model and alias matching is case-insensitive; adapter matching is exact, mirroring the adapter routing behavior of
    the engine.
    """
    return model_matches(requested, async_engine) or requested in (adapters or [])

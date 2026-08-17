"""Shared test helpers for the abstraction capsule."""

from __future__ import annotations

import dataclasses

import pytest


def _walk(value, seen=None):
    """Yield `value` and every object nested inside it.

    Walks mappings, sequences, sets and dataclasses. Used by the
    profile and card tests, which must decide structurally whether a retained
    environment or a state-card term still carries source history.
    """
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))

    yield value

    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(key, seen)
            yield from _walk(item, seen)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _walk(item, seen)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _walk(getattr(value, field.name), seen)
        return
    mapping = getattr(value, "__dict__", None)
    if isinstance(mapping, dict):
        for item in mapping.values():
            yield from _walk(item, seen)


@pytest.fixture
def walk_values():
    return _walk

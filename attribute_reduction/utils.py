"""Shared helper utilities for attribute reduction algorithms."""

from __future__ import annotations

import numpy as np


def unique_preserve_order(items):
    """Drop duplicates while preserving first-seen order."""
    seen = set()
    out = []
    for item in items:
        key = tuple(item) if isinstance(item, (list, tuple, np.ndarray)) else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def to_index_list(columns):
    """Normalize a columns-like input to python list[int]."""
    if columns is None:
        return []
    if isinstance(columns, np.ndarray):
        return columns.tolist()
    return list(columns)


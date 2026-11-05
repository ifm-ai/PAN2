# Copyright 2026 Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
# SPDX-License-Identifier: Apache-2.0
"""Option parsing shared by ``train.py`` and ``generate.py``."""
import argparse
import json
import os
from typing import Any, Dict, Optional

_TRUE_WORDS = {"true", "1", "yes", "on"}
_FALSE_WORDS = {"false", "0", "no", "off"}


def str_to_bool(value) -> bool:
    """argparse ``type`` for on/off flags; a bare flag (``nargs='?'`` with no value) arrives as ``None`` and means on."""
    if value is None or isinstance(value, bool):
        return value is None or value
    word = str(value).strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean (true/false, 1/0, yes/no, on/off), got {value!r}")


def parse_json_object(raw: Any, name: str) -> Optional[Dict[str, Any]]:
    """A mapping given as a dict, a JSON object string or a path to a JSON file; ``None`` / blank -> ``None``."""
    if raw is None or isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a JSON object, a JSON file path or a YAML mapping")
    text = raw.strip()
    if not text:
        return None
    try:
        value = json.load(open(text, encoding="utf-8")) if os.path.isfile(text) else json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name}: not valid JSON (for shell wrappers, a JSON file path avoids quoting issues)") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value

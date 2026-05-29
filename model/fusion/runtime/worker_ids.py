"""Worker ID conversion helpers shared by alert and DB logging paths."""

from __future__ import annotations

import re


def worker_label_to_int(worker_id: str | int | None, default: int = 1) -> int:
    """Convert runtime labels like W01/worker_2 into DB worker ids 1/2."""
    if worker_id is None:
        return default
    if isinstance(worker_id, int):
        return worker_id

    text = str(worker_id).strip()
    if not text:
        return default

    match = re.search(r"(\d+)$", text)
    if match:
        return int(match.group(1))
    return default


def worker_label_to_topic_id(worker_id: str | int | None, default: str = "1") -> str:
    """Convert a runtime worker label into the MQTT/API topic id string."""
    return str(worker_label_to_int(worker_id, int(default)))

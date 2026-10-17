"""Lightweight diagnostics for Fluxon CUDA host registration.

The Fluxon storage path emits these records from scheduler and worker
processes while registering the owner's host segments.  Keep this helper
dependency-free so it is available during the earliest scheduler import and
so diagnostics cannot turn a cache operation into a failure.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_WRITE_LOCK = threading.Lock()


def _diagnostic_path() -> Path | None:
    value = os.environ.get("SGLANG_FLUXON_REGISTRATION_DIAGNOSTIC_PATH", "").strip()
    return Path(value) if value else None


def append_registration_diagnostic(event: str, **fields: Any) -> None:
    """Record a registration event without affecting the storage operation.

    ``include_snapshot`` is accepted by callers for compatibility with the
    deployed Fluxon instrumentation.  Snapshot collection is intentionally
    left to the caller's normal logs; this helper only records the supplied
    event fields and remains safe during exception handling.
    """

    fields.pop("include_snapshot", None)
    row = {
        "at_ns": time.time_ns(),
        "event": str(event),
        **fields,
    }
    try:
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        encoded = repr(row)
    logger.info("Fluxon registration diagnostic: %s", encoded)

    path = _diagnostic_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
    except Exception:
        # Diagnostics must never break scheduler startup or cleanup.
        logger.debug("Unable to persist Fluxon registration diagnostic", exc_info=True)


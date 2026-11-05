from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_payload_hash(payload: Any, length: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]

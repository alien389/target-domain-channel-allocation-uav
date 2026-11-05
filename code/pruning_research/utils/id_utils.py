from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional


def sanitize_id_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    return normalized or "item"


def utc_timestamp_compact(now: Optional[datetime] = None) -> str:
    current = now or datetime.now(timezone.utc)
    return current.strftime("%Y%m%d_%H%M%S")


def build_experiment_id(experiment_type: str, model_id: str, now: Optional[datetime] = None) -> str:
    return f"{sanitize_id_component(experiment_type)}_{sanitize_id_component(model_id)}_{utc_timestamp_compact(now)}"

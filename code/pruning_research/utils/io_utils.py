from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_csv_rows(path: str | Path) -> List[Dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    with open(target, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_json(payload: Any, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return target


def save_csv(rows: List[Dict[str, Any]], path: str | Path, fieldnames: Optional[Iterable[str]] = None) -> Path:
    target = Path(path)
    ensure_dir(target.parent)

    headers = list(fieldnames) if fieldnames is not None else []
    if not headers and rows:
        headers = list(rows[0].keys())

    with open(target, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return target

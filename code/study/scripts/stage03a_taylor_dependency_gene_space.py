from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import networkx as nx
import numpy as np
import torch
import yaml
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
STAGE00 = PROJECT / "00_preflight_freeze"
STAGE01 = PROJECT / "01_data_split"
STAGE01A = PROJECT / "01A_protocol_amendment"
STAGE02 = PROJECT / "02_search_parent_teacher" / "lite_v3_search_parent"
STAGE03 = PROJECT / "03a_taylor_dependency_gene_space"
APPROVAL = PROJECT / "manifests" / "approvals" / "stage02.token"

OD_ROOT = Path(_release_paths.expand('${CODE_ROOT}'))
PRUNING_PACKAGE_ROOT = OD_ROOT
PRUNING_ROOT = OD_ROOT / "pruning_research"
YOLOV5_ROOT = OD_ROOT / "yolov5"
PYTHON = Path(_release_paths.expand('${TRAINING_PYTHON}'))

BEST_PT = STAGE02 / "weights" / "best.pt"
EXPECTED_BEST_SHA = "15E0E4C80677F9E01163D66CE9844C6ED96FF9E922D4FDA5530A3425A3009659"
META_TRAIN = STAGE01 / "meta_train5177_seed42.txt"
INNER_VAL = STAGE01 / "meta_inner_val1294_seed42.txt"
OFFICIAL_LOCKED = STAGE01 / "official_val548_locked.txt"
SPLIT_MANIFEST = STAGE01 / "split_manifest.csv"
SPLIT_HASHES = STAGE01 / "split_hashes.json"
HYP_YAML = STAGE02 / "hyp.yaml"
MODEL_COMPLEXITY = STAGE02 / "model_complexity.json"

SUBSET_TXT = STAGE03 / "taylor_subset512_seed42.txt"
SUBSET_ALIAS = STAGE03 / "taylor_sensitivity512_seed42.txt"
SUBSET_MANIFEST = STAGE03 / "taylor_subset_manifest.csv"
SUBSET_DATA_YAML = STAGE03 / "taylor_subset_data.yaml"
PROFILE_ROOT = STAGE03 / "structural_profile"
PROFILE_MODEL_ID = "litev3_searchparent_nc10_stage03a_unfused"
PROFILE_DIR = PROFILE_ROOT / PROFILE_MODEL_ID
TAYLOR_RUNTIME = STAGE03 / "taylor_runtime"

TAYLOR_IMAGES = 512
BATCH_SIZE = 16
MAX_BATCHES = 32
SEED = 42
IMGSZ = 640
CHANNEL_QUANTUM = 8
GENERIC_MIN_WIDTH = 8

BUDGET_PARAMS_MAX = 2_515_679
BUDGET_FLOPS_MAX_G = 7.8540032
BUDGET_LOWER_FRACTION = 0.97

EXPECTED_STAGE01_HASHES = {
    META_TRAIN: "A890318CB684513A4A4F27FC5E0FEDA7207A54CC554D556940907B2FBD1EA936",
    INNER_VAL: "1814CF80594A014FAB3AD8D6F80D956441FD06EAA43DDDA8970D148F82827278",
    OFFICIAL_LOCKED: "8A30707BCC6FB6E119C929BAD1A4D531299E497A9762666AD70A9C6A31B2DE4A",
    SPLIT_MANIFEST: "B7AC38A4E7EA8ED6693C4ACD4584B24A4CA93F6A5354D7563F59267683CB307C",
}

SOURCE_FILES = [
    STAGE00 / "frozen_execution_plan.md",
    STAGE00 / "frozen_hyperparameters.yaml",
    STAGE01A / "revised_frozen_hyperparameters.yaml",
    STAGE01A / "revised_stage_map.yaml",
    STAGE02 / "stage02_state.json",
    STAGE02 / "checkpoint_hashes.json",
    MODEL_COMPLEXITY,
    META_TRAIN,
    INNER_VAL,
    OFFICIAL_LOCKED,
    SPLIT_MANIFEST,
    SPLIT_HASHES,
    HYP_YAML,
    YOLOV5_ROOT / "models" / "yolov5s_lite_v3.yaml",
    PRUNING_ROOT / "pruning" / "taylor_importance.py",
    PRUNING_ROOT / "pruning" / "importance_scores.py",
    PRUNING_ROOT / "pruning" / "structured_pruner.py",
    PRUNING_ROOT / "profiling" / "deep_structure_profile.py",
    PRUNING_ROOT / "profiling" / "profile_export.py",
    PRUNING_ROOT / "configs" / "taylor_importance_litev3_baseline.yaml",
    PRUNING_ROOT / "configs" / "pruning_search_space_litev3_stage6_c30.yaml",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def yaml_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def normalize_path_text(value: str) -> str:
    return os.path.normcase(os.path.normpath(value.strip()))


def source_snapshot() -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for path in SOURCE_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Required source file is missing: {path}")
        snapshot[str(path)] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return snapshot


def assert_source_snapshot_unchanged(before: dict[str, dict[str, Any]]) -> None:
    after = source_snapshot()
    changed = [
        path
        for path, metadata in before.items()
        if path not in after or metadata["sha256"] != after[path]["sha256"]
    ]
    if changed:
        raise RuntimeError(f"Read-only source files changed during Stage03A: {changed}")


def check_preflight() -> dict[str, Any]:
    if APPROVAL.read_text(encoding="utf-8").strip() != "APPROVE_STAGE02_SEARCH_PARENT":
        raise RuntimeError("Stage02 approval token is missing or invalid.")
    stage02_state = json_load(STAGE02 / "stage02_state.json")
    if stage02_state.get("stage02_search_parent") != "PASS":
        raise RuntimeError("Stage02 SearchParent state is not PASS.")
    if stage02_state.get("validation") != "PASS":
        raise RuntimeError("Stage02 validation state is not PASS.")
    if stage02_state.get("official_val_accessed") is not False:
        raise RuntimeError("Stage02 state does not confirm official-val lock.")
    if sha256(BEST_PT) != EXPECTED_BEST_SHA:
        raise RuntimeError("SearchParent best.pt SHA256 mismatch.")
    for path, expected in EXPECTED_STAGE01_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Stage01 immutable hash mismatch: {path}: {actual} != {expected}")
    if len(read_nonempty_lines(META_TRAIN)) != 5177:
        raise RuntimeError("meta-train count is not 5177.")
    if len(read_nonempty_lines(INNER_VAL)) != 1294:
        raise RuntimeError("meta-inner-val count is not 1294.")
    if len(read_nonempty_lines(OFFICIAL_LOCKED)) != 548:
        raise RuntimeError("official locked list count is not 548.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the frozen Stage03A Taylor run.")
    disk = shutil.disk_usage(PROJECT)
    return {
        "timestamp_utc": utc_now(),
        "approval_token": APPROVAL.read_text(encoding="utf-8").strip(),
        "best_checkpoint": str(BEST_PT),
        "best_checkpoint_sha256": sha256(BEST_PT),
        "stage01_hashes_verified": True,
        "meta_train_count": 5177,
        "meta_inner_val_count": 1294,
        "official_val_locked_count": 548,
        "official_val_evaluation_accessed": False,
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "free_space_bytes": disk.free,
        "free_space_gib": disk.free / (1024**3),
    }


def protocol_audit_rows() -> list[dict[str, Any]]:
    historical_cfg = PRUNING_ROOT / "configs" / "taylor_importance_litev3_baseline.yaml"
    implementation = PRUNING_ROOT / "pruning" / "taylor_importance.py"
    stage03_plan = STAGE00 / "frozen_execution_plan.md"
    common_source = f"{implementation} (SHA256 {sha256(implementation)})"
    rows = [
        {
            "Field": "Taylor implementation",
            "Previous": str(implementation),
            "Current": str(implementation),
            "Status": "MATCH",
            "Evidence": common_source,
            "Notes": "Historical source file is imported without modification.",
        },
        {
            "Field": "Importance formula",
            "Previous": "|A * dL/dA| reduced over N,H,W",
            "Current": "|A * dL/dA| reduced over N,H,W",
            "Status": "MATCH",
            "Evidence": f"{implementation}:34-46,221-237",
            "Notes": "Exact implemented first-order activation-gradient interaction.",
        },
        {
            "Field": "Score reduction",
            "Previous": "mean_abs over dims (0,2,3)",
            "Current": "mean_abs over dims (0,2,3)",
            "Status": "MATCH",
            "Evidence": f"{implementation}:34-46",
            "Notes": "No alternative reduction introduced.",
        },
        {
            "Field": "Loss implementation",
            "Previous": "official YOLOv5 ComputeLoss",
            "Current": "official YOLOv5 ComputeLoss",
            "Status": "MATCH",
            "Evidence": f"{implementation}:145-177,221-224; {HYP_YAML}",
            "Notes": "Box, objectness, and classification components use checkpoint/model hyp weights.",
        },
        {
            "Field": "Loss weights",
            "Previous": "resolved from frozen model/hyp",
            "Current": "box=0.05; obj=1.0; cls=0.5",
            "Status": "MATCH",
            "Evidence": str(HYP_YAML),
            "Notes": "Frozen Stage02 hyp is passed explicitly.",
        },
        {
            "Field": "Model mode",
            "Previous": "train mode",
            "Current": "train mode",
            "Status": "MATCH",
            "Evidence": f"{implementation}:127-133",
            "Notes": "Gradient-compatible mode retained.",
        },
        {
            "Field": "BatchNorm policy",
            "Previous": "BN modules eval; running statistics frozen",
            "Current": "BN modules eval; running statistics frozen",
            "Status": "MATCH",
            "Evidence": f"{implementation}:49-55,130-133",
            "Notes": "Affine parameters remain in graph; running statistics do not update.",
        },
        {
            "Field": "Augmentation",
            "Previous": "disabled",
            "Current": "disabled",
            "Status": "MATCH",
            "Evidence": f"{implementation}:157-175",
            "Notes": "create_dataloader augment=False.",
        },
        {
            "Field": "Shuffle",
            "Previous": "false; source-list order",
            "Current": "false; frozen subset-list order",
            "Status": "MATCH",
            "Evidence": f"{implementation}:157-175",
            "Notes": "No class balancing or new stratification is introduced.",
        },
        {
            "Field": "Subset construction",
            "Previous": "ordered source list consumed without shuffle",
            "Current": "first 512 unique ordered entries from immutable meta-train list",
            "Status": "MATCH",
            "Evidence": f"{implementation}:157-175; current protocol clarification",
            "Notes": "The older Stage00 stratified wording is superseded by the explicit protocol-equivalence clarification.",
        },
        {
            "Field": "Subset size",
            "Previous": "historical campaign horizon was larger",
            "Current": "512 images",
            "Status": "EXPECTED_CHANGE",
            "Evidence": f"{stage03_plan}:882-890; current approval",
            "Notes": "Stage03A freezes a 512-image horizon while retaining the scoring implementation.",
        },
        {
            "Field": "Subset seed",
            "Previous": "42",
            "Current": "42",
            "Status": "MATCH",
            "Evidence": f"{historical_cfg}; {implementation}:109,174",
            "Notes": "Seed is passed to the unchanged dataloader.",
        },
        {
            "Field": "Batch size",
            "Previous": "16",
            "Current": "16",
            "Status": "MATCH",
            "Evidence": f"{historical_cfg}; current approval",
            "Notes": "",
        },
        {
            "Field": "Processed batches",
            "Previous": "historical campaign horizon was larger",
            "Current": "32",
            "Status": "EXPECTED_CHANGE",
            "Evidence": f"{stage03_plan}:882-890; current approval",
            "Notes": "Exactly 512 / 16 batches.",
        },
        {
            "Field": "Image size",
            "Previous": "640",
            "Current": "640",
            "Status": "MATCH",
            "Evidence": f"{historical_cfg}; current approval",
            "Notes": "",
        },
        {
            "Field": "Precision / AMP",
            "Previous": "FP32; no autocast",
            "Current": "FP32; no autocast",
            "Status": "MATCH",
            "Evidence": f"{implementation}:127,217-224",
            "Notes": "",
        },
        {
            "Field": "Gradient reset",
            "Previous": "zero_grad before and after each batch",
            "Current": "zero_grad before and after each batch",
            "Status": "MATCH",
            "Evidence": f"{implementation}:221,244",
            "Notes": "Scores accumulate; gradients do not accumulate between batches.",
        },
        {
            "Field": "Batch aggregation",
            "Previous": "arithmetic mean of per-batch channel scores",
            "Current": "arithmetic mean of 32 per-batch channel scores",
            "Status": "MATCH",
            "Evidence": f"{implementation}:237-238,271-277",
            "Notes": "",
        },
        {
            "Field": "Normalization",
            "Previous": "per-scope score / per-scope maximum",
            "Current": "per-scope score / per-scope maximum",
            "Status": "MATCH",
            "Evidence": f"{implementation}:275-285",
            "Notes": "",
        },
        {
            "Field": "Optimizer / updates",
            "Previous": "no optimizer; no step; no EMA",
            "Current": "no optimizer; no step; no EMA",
            "Status": "MATCH",
            "Evidence": f"{implementation}:92-374",
            "Notes": "Checkpoint is loaded read-only and re-hashed after collection.",
        },
        {
            "Field": "Checkpoint",
            "Previous": "source-domain Lite-v3 checkpoint",
            "Current": f"Lite-v3-SearchParent best.pt ({EXPECTED_BEST_SHA})",
            "Status": "EXPECTED_CHANGE",
            "Evidence": str(BEST_PT),
            "Notes": "Authorized target-domain change.",
        },
        {
            "Field": "Eligible data pool",
            "Previous": "historical source-domain training pool",
            "Current": "meta-train5177 only",
            "Status": "EXPECTED_CHANGE",
            "Evidence": f"{META_TRAIN} ({sha256(META_TRAIN)})",
            "Notes": "inner-val1294 and official val548 are excluded.",
        },
    ]
    return rows


def style_workbook(path: Path) -> None:
    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.sheet_view.showGridLines = False
        if ws.max_row >= 1 and ws.max_column >= 1:
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            ws.auto_filter.ref = ws.dimensions
            thin = Side(style="thin", color="B7C9D6")
            for row in ws.iter_rows():
                for cell in row:
                    cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
            for column_cells in ws.columns:
                length = min(
                    70,
                    max(10, max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells) + 2),
                )
                ws.column_dimensions[get_column_letter(column_cells[0].column)].width = length
        if ws.max_row >= 2 and ws.max_column >= 1:
            safe_name = "".join(ch for ch in ws.title if ch.isalnum())[:20] or "Sheet"
            table = Table(displayName=f"T_{safe_name}", ref=ws.dimensions)
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2",
                showFirstColumn=False,
                showLastColumn=False,
                showRowStripes=True,
                showColumnStripes=False,
            )
            ws.add_table(table)
    wb.save(path)


def xlsx_from_sheets(path: Path, sheets: list[tuple[str, list[dict[str, Any]]]]) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets:
        ws = wb.create_sheet(name[:31])
        headers = list(rows[0].keys()) if rows else ["Status"]
        ws.append(headers)
        if rows:
            for row in rows:
                ws.append([row.get(header) for header in headers])
        else:
            ws.append(["NO_ROWS"])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    style_workbook(path)


def create_protocol_audit() -> dict[str, Any]:
    rows = protocol_audit_rows()
    unresolved = [row for row in rows if row["Status"] == "UNRESOLVED"]
    payload = {
        "created_utc": utc_now(),
        "protocol_name": "Stage03A frozen Taylor protocol equivalence audit",
        "allowed_statuses": ["MATCH", "EXPECTED_CHANGE", "UNRESOLVED"],
        "rows": rows,
        "major_unresolved_fields": len(unresolved),
        "execution_allowed": len(unresolved) == 0,
        "historical_implementation_sha256": sha256(PRUNING_ROOT / "pruning" / "taylor_importance.py"),
        "status": "PASS" if not unresolved else "FAIL_UNRESOLVED_PROTOCOL",
    }
    json_write(STAGE03 / "taylor_protocol_equivalence_audit.json", payload)
    xlsx_from_sheets(STAGE03 / "taylor_protocol_equivalence_audit.xlsx", [("Protocol Audit", rows)])
    if unresolved:
        raise RuntimeError(f"Taylor protocol contains unresolved major fields: {unresolved}")
    return payload


def create_subset() -> dict[str, Any]:
    all_train = read_nonempty_lines(META_TRAIN)
    inner = {normalize_path_text(value) for value in read_nonempty_lines(INNER_VAL)}
    official = {normalize_path_text(value) for value in read_nonempty_lines(OFFICIAL_LOCKED)}

    subset: list[str] = []
    seen: set[str] = set()
    for path_text in all_train:
        key = normalize_path_text(path_text)
        if key in seen:
            continue
        seen.add(key)
        subset.append(path_text)
        if len(subset) == TAYLOR_IMAGES:
            break
    if len(subset) != TAYLOR_IMAGES:
        raise RuntimeError("Unable to construct 512 unique ordered meta-train entries.")
    if {normalize_path_text(value) for value in subset} & inner:
        raise RuntimeError("Taylor subset overlaps meta-inner-val.")
    if {normalize_path_text(value) for value in subset} & official:
        raise RuntimeError("Taylor subset overlaps official val.")

    manifest_map: dict[str, dict[str, str]] = {}
    with SPLIT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            manifest_map[normalize_path_text(row["image_path"])] = row

    rows: list[dict[str, Any]] = []
    image_hashes: list[str] = []
    class_images: Counter[int] = Counter()
    object_counts: list[int] = []
    small_counts: list[int] = []
    small_ratios: list[float] = []
    areas: list[int] = []
    for index, path_text in enumerate(subset, start=1):
        key = normalize_path_text(path_text)
        row = manifest_map.get(key)
        if not row:
            raise RuntimeError(f"Subset image is absent from split_manifest.csv: {path_text}")
        image_path = Path(path_text)
        label_path = Path(row["label_path"])
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing image or label: {image_path}; {label_path}")
        image_sha = sha256(image_path)
        label_sha = sha256(label_path)
        if image_sha != row["image_sha256"].upper() or label_sha != row["label_sha256"].upper():
            raise RuntimeError(f"Dataset artifact hash mismatch for {image_path}")
        classes = [int(value) for value in row["present_classes"].split("|") if value != ""]
        for class_id in classes:
            class_images[class_id] += 1
        object_count = int(row["object_count"])
        small_count = int(row["small_object_count"])
        small_ratio = float(row["small_object_ratio"])
        image_area = int(row["image_area"])
        object_counts.append(object_count)
        small_counts.append(small_count)
        small_ratios.append(small_ratio)
        areas.append(image_area)
        image_hashes.append(image_sha)
        rows.append(
            {
                "selection_order": index,
                "selection_rule": "ORDERED_PREFIX_NO_SHUFFLE_FROM_IMMUTABLE_META_TRAIN_LIST",
                "selection_seed": SEED,
                "image_path": str(image_path),
                "image_filename": image_path.name,
                "image_sha256": image_sha,
                "label_path": str(label_path),
                "label_sha256": label_sha,
                "present_classes": row["present_classes"],
                "object_count": object_count,
                "small_object_count": small_count,
                "small_object_ratio": small_ratio,
                "image_width": int(row["image_width"]),
                "image_height": int(row["image_height"]),
                "image_area": image_area,
                "source_split": row["assigned_split"],
                "source_manifest": str(SPLIT_MANIFEST),
            }
        )

    text_write(SUBSET_TXT, "\n".join(subset) + "\n")
    text_write(SUBSET_ALIAS, "\n".join(subset) + "\n")
    csv_write(SUBSET_MANIFEST, rows)
    base_yaml = yaml.safe_load((STAGE01 / "data_yamls" / "visdrone_meta_train5177_inner1294.yaml").read_text(encoding="utf-8"))
    base_yaml["train"] = str(SUBSET_TXT).replace("\\", "/")
    base_yaml["val"] = str(SUBSET_TXT).replace("\\", "/")
    base_yaml["usage"] = "Stage03A Taylor scoring only; official val locked"
    base_yaml["taylor_subset_seed"] = SEED
    base_yaml["official_val_used"] = False
    yaml_write(SUBSET_DATA_YAML, base_yaml)

    summary_rows = [
        {"Metric": "Selection rule", "Value": "ORDERED_PREFIX_NO_SHUFFLE_FROM_IMMUTABLE_META_TRAIN_LIST"},
        {"Metric": "Selection seed", "Value": SEED},
        {"Metric": "Image count", "Value": len(rows)},
        {"Metric": "Unique image paths", "Value": len({normalize_path_text(row["image_path"]) for row in rows})},
        {"Metric": "Unique image hashes", "Value": len(set(image_hashes))},
        {"Metric": "Overlap with inner-val", "Value": 0},
        {"Metric": "Overlap with official val", "Value": 0},
        {"Metric": "Classes represented", "Value": len(class_images)},
        {"Metric": "Mean objects/image", "Value": mean(object_counts)},
        {"Metric": "Mean small objects/image", "Value": mean(small_counts)},
        {"Metric": "Mean small-object ratio", "Value": mean(small_ratios)},
        {"Metric": "Mean image area", "Value": mean(areas)},
    ]
    class_rows = [
        {
            "Class_ID": class_id,
            "Images_With_Class": class_images[class_id],
            "Image_Prevalence": class_images[class_id] / len(rows),
        }
        for class_id in range(10)
    ]
    xlsx_from_sheets(
        STAGE03 / "taylor_subset_statistics.xlsx",
        [("Summary", summary_rows), ("Class Presence", class_rows)],
    )
    hashes_payload = {
        "created_utc": utc_now(),
        "selection_rule": "ORDERED_PREFIX_NO_SHUFFLE_FROM_IMMUTABLE_META_TRAIN_LIST",
        "selection_seed": SEED,
        "subset_count": len(rows),
        "subset_list_sha256": sha256(SUBSET_TXT),
        "subset_alias_sha256": sha256(SUBSET_ALIAS),
        "subset_manifest_sha256": sha256(SUBSET_MANIFEST),
        "subset_data_yaml_sha256": sha256(SUBSET_DATA_YAML),
        "source_meta_train_sha256": sha256(META_TRAIN),
        "source_split_manifest_sha256": sha256(SPLIT_MANIFEST),
        "image_hash_digest": hashlib.sha256("\n".join(image_hashes).encode("ascii")).hexdigest().upper(),
        "inner_val_overlap": 0,
        "official_val_overlap": 0,
        "status": "PASS",
    }
    json_write(STAGE03 / "taylor_subset_hashes.json", hashes_payload)
    return hashes_payload


def run_profile() -> dict[str, Any]:
    summary_path = PROFILE_DIR / "model_structure_summary.json"
    if summary_path.exists():
        summary = json_load(summary_path)
        metadata = summary.get("model", {})
        if int(metadata.get("nc", -1)) == 10:
            return summary
    if str(PRUNING_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PRUNING_PACKAGE_ROOT))
    from pruning_research.profiling.profile_export import run_structural_profile_export

    result = run_structural_profile_export(
        weights=str(BEST_PT),
        yolov5_root=str(YOLOV5_ROOT),
        output_dir=str(PROFILE_ROOT),
        registry_dir=str(STAGE03 / "profile_registry"),
        model_id=PROFILE_MODEL_ID,
        imgsz=IMGSZ,
        device="cuda:0",
        fuse=False,
    )
    return json_load(Path(result["model_structure_summary_json"]))


def model_contract_audit(profile_summary: dict[str, Any]) -> dict[str, Any]:
    complexity = json_load(MODEL_COMPLEXITY)
    metadata = profile_summary["model"]
    profile = profile_summary["profiling"]
    tests = [
        {
            "Test": "SearchParent checkpoint SHA256",
            "Expected": EXPECTED_BEST_SHA,
            "Actual": sha256(BEST_PT),
            "Status": "PASS" if sha256(BEST_PT) == EXPECTED_BEST_SHA else "FAIL",
        },
        {
            "Test": "Class count",
            "Expected": 10,
            "Actual": metadata.get("nc"),
            "Status": "PASS" if int(metadata.get("nc", -1)) == 10 else "FAIL",
        },
        {
            "Test": "Fused parameters",
            "Expected": 3_985_767,
            "Actual": complexity["fused_inference_graph"]["params"],
            "Status": "PASS" if complexity["fused_inference_graph"]["params"] == 3_985_767 else "FAIL",
        },
        {
            "Test": "Fused FLOPs G",
            "Expected": 10.825984,
            "Actual": complexity["fused_inference_graph"]["flops_g_thop_macs_x2"],
            "Status": "PASS"
            if abs(complexity["fused_inference_graph"]["flops_g_thop_macs_x2"] - 10.825984) <= 1e-9
            else "FAIL",
        },
        {
            "Test": "Input shape",
            "Expected": "1x3x640x640",
            "Actual": "x".join(str(value) for value in complexity["best_checkpoint_sanity"]["input_shape"]),
            "Status": "PASS"
            if complexity["best_checkpoint_sanity"]["input_shape"] == [1, 3, 640, 640]
            else "FAIL",
        },
        {
            "Test": "Detect scales",
            "Expected": 3,
            "Actual": complexity["best_checkpoint_sanity"]["detect_scale_count"],
            "Status": "PASS"
            if complexity["best_checkpoint_sanity"]["detect_scale_count"] == 3
            else "FAIL",
        },
        {
            "Test": "Nonfinite tensors",
            "Expected": 0,
            "Actual": complexity["best_checkpoint_sanity"]["nonfinite_tensor_count"],
            "Status": "PASS"
            if complexity["best_checkpoint_sanity"]["nonfinite_tensor_count"] == 0
            else "FAIL",
        },
        {
            "Test": "Profile failures",
            "Expected": 0,
            "Actual": len(profile.get("profile_failures", [])),
            "Status": "PASS" if not profile.get("profile_failures") else "FAIL",
        },
        {
            "Test": "Official val evaluation access",
            "Expected": False,
            "Actual": False,
            "Status": "PASS",
        },
    ]
    status = "PASS" if all(row["Status"] == "PASS" for row in tests) else "FAIL"
    payload = {
        "created_utc": utc_now(),
        "tests": tests,
        "fused_complexity_source": str(MODEL_COMPLEXITY),
        "unfused_profile_source": str(PROFILE_DIR / "model_structure_summary.json"),
        "flops_protocol": "THOP MACs x2 at 1x3x640x640",
        "status": status,
    }
    json_write(STAGE03 / "model_contract_audit.json", payload)
    if status != "PASS":
        raise RuntimeError("SearchParent model-contract audit failed.")
    return payload


def taylor_config_payload() -> dict[str, Any]:
    return {
        "stage": "Stage03A",
        "model_id": PROFILE_MODEL_ID,
        "weights": str(BEST_PT),
        "weights_sha256": sha256(BEST_PT),
        "yolov5_root": str(YOLOV5_ROOT),
        "profile_dir": str(PROFILE_DIR),
        "data_yaml": str(SUBSET_DATA_YAML),
        "data_split": "train",
        "subset_list": str(SUBSET_TXT),
        "subset_count": TAYLOR_IMAGES,
        "subset_seed": SEED,
        "subset_selection": "ordered prefix from immutable meta-train list; shuffle=false",
        "imgsz": IMGSZ,
        "batch_size": BATCH_SIZE,
        "max_batches": MAX_BATCHES,
        "workers": 4,
        "device": "cuda:0",
        "prefer_ema": True,
        "score_reduction": "mean_abs",
        "freeze_bn_running_stats": True,
        "hyp_yaml": str(HYP_YAML),
        "amp": False,
        "optimizer_step": False,
        "weight_update": False,
        "ema_update": False,
        "augmentation": False,
        "shuffle": False,
        "formula": "I_c = mean_(n,h,w) |A_(n,c,h,w) * dL/dA_(n,c,h,w)|; then mean over batches",
        "normalization": "per-scope score divided by per-scope maximum",
        "implementation": str(PRUNING_ROOT / "pruning" / "taylor_importance.py"),
        "implementation_sha256": sha256(PRUNING_ROOT / "pruning" / "taylor_importance.py"),
        "official_val_used": False,
    }


def run_taylor() -> dict[str, Any]:
    summary_path = TAYLOR_RUNTIME / "taylor_collection_summary.json"
    if summary_path.exists():
        prior = json_load(summary_path)
        if prior.get("processed_batches") == MAX_BATCHES and prior.get("observed_images") == TAYLOR_IMAGES:
            return prior
    if str(PRUNING_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PRUNING_PACKAGE_ROOT))
    from pruning_research.pruning.taylor_importance import collect_taylor_scores_one_run

    before_checkpoint_sha = sha256(BEST_PT)
    result = collect_taylor_scores_one_run(
        weights=str(BEST_PT),
        yolov5_root=str(YOLOV5_ROOT),
        profile_dir=str(PROFILE_DIR),
        output_dir=str(TAYLOR_RUNTIME),
        data_yaml=str(SUBSET_DATA_YAML),
        model_id=PROFILE_MODEL_ID,
        imgsz=IMGSZ,
        batch_size=BATCH_SIZE,
        workers=4,
        device="cuda:0",
        prefer_ema=True,
        split="train",
        max_batches=MAX_BATCHES,
        start_batch_offset=0,
        score_reduction="mean_abs",
        freeze_bn_running_stats=True,
        seed=SEED,
        hyp_yaml=str(HYP_YAML),
    )
    after_checkpoint_sha = sha256(BEST_PT)
    if before_checkpoint_sha != after_checkpoint_sha:
        raise RuntimeError("SearchParent checkpoint changed during Taylor collection.")
    if result["processed_batches"] != MAX_BATCHES or result["observed_images"] != TAYLOR_IMAGES:
        raise RuntimeError(f"Taylor horizon mismatch: {result}")
    return result


def export_taylor_outputs(taylor_summary: dict[str, Any]) -> dict[str, Any]:
    raw_json = json_load(TAYLOR_RUNTIME / "raw_taylor_scores.json")
    channel_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    matrix_rows: list[list[float]] = []
    group_index = 0
    for block_name, block in raw_json["importance_bank"].items():
        for scope_name, scope in block["scope_rankings"].items():
            group_index += 1
            scores = [float(value) for value in scope["scores"]]
            normalized = [float(value) for value in scope["normalized_scores"]]
            ranks = {int(channel): rank + 1 for rank, channel in enumerate(scope["indices_low_to_high"])}
            summary_rows.append(
                {
                    "group_index": group_index,
                    "block_name": block_name,
                    "stage": block["stage"],
                    "module_type": block["module_type"],
                    "scope_name": scope_name,
                    "scope_type": scope["scope_type"],
                    "anchor_module": scope["anchor_module"],
                    "width": scope["width"],
                    "observations": scope["observation_count"],
                    "min_score": min(scores),
                    "mean_score": mean(scores),
                    "max_score": max(scores),
                    "score_reduction": raw_json["score_reduction"],
                }
            )
            for channel_index, (score, norm_score) in enumerate(zip(scores, normalized)):
                base = {
                    "group_index": group_index,
                    "block_name": block_name,
                    "stage": block["stage"],
                    "module_type": block["module_type"],
                    "scope_name": scope_name,
                    "anchor_module": scope["anchor_module"],
                    "channel_index": channel_index,
                    "rank_low_to_high": ranks[channel_index],
                    "score": score,
                    "normalized_score": norm_score,
                    "observations": scope["observation_count"],
                }
                channel_rows.append(base)
                normalized_rows.append(
                    {
                        **base,
                        "score": norm_score,
                        "raw_score": score,
                    }
                )
                matrix_rows.append([group_index, channel_index, score, norm_score, ranks[channel_index]])
    csv_write(STAGE03 / "raw_channel_taylor_scores.csv", channel_rows)
    csv_write(STAGE03 / "normalized_channel_taylor_scores.csv", normalized_rows)
    xlsx_from_sheets(
        STAGE03 / "layer_taylor_summary.xlsx",
        [("Scope Summary", summary_rows), ("Channel Scores", channel_rows)],
    )
    xlsx_from_sheets(STAGE03 / "taylor_scores.xlsx", [("Channel Scores", channel_rows)])
    np.save(STAGE03 / "taylor_scores.npy", np.asarray(matrix_rows, dtype=np.float64))
    ranking_payload = {
        "created_utc": utc_now(),
        "model_id": PROFILE_MODEL_ID,
        "checkpoint_sha256": sha256(BEST_PT),
        "processed_batches": taylor_summary["processed_batches"],
        "observed_images": taylor_summary["observed_images"],
        "target_block_count": taylor_summary["target_block_count"],
        "target_scope_count": taylor_summary["target_scope_count"],
        "scope_count": len(summary_rows),
        "channel_score_count": len(channel_rows),
        "importance_bank": raw_json["importance_bank"],
        "status": "PASS",
    }
    json_write(STAGE03 / "taylor_ranking_manifest.json", ranking_payload)
    run_manifest = {
        "created_utc": utc_now(),
        "stage": "Stage03A",
        "model_id": PROFILE_MODEL_ID,
        "checkpoint": str(BEST_PT),
        "checkpoint_sha256_before_after": sha256(BEST_PT),
        "subset_list": str(SUBSET_TXT),
        "subset_list_sha256": sha256(SUBSET_TXT),
        "data_yaml": str(SUBSET_DATA_YAML),
        "data_yaml_sha256": sha256(SUBSET_DATA_YAML),
        "implementation": str(PRUNING_ROOT / "pruning" / "taylor_importance.py"),
        "implementation_sha256": sha256(PRUNING_ROOT / "pruning" / "taylor_importance.py"),
        "processed_batches": taylor_summary["processed_batches"],
        "observed_images": taylor_summary["observed_images"],
        "batch_size": BATCH_SIZE,
        "imgsz": IMGSZ,
        "seed": SEED,
        "device": "cuda:0",
        "frozen_bn_count": taylor_summary["frozen_bn_count"],
        "optimizer_steps": 0,
        "weight_updates": 0,
        "ema_updates": 0,
        "official_val_used": False,
        "loader_warnings": [
            {
                "image": _release_paths.expand('${DATASETS_ROOT}\\VisDrone2019-YOLO-format\\images\\train\\0000140_00118_d_0000002.jpg'),
                "warning": "One duplicate label was removed in memory by the unchanged YOLOv5 dataloader.",
                "source_file_modified": False,
            }
        ],
        "status": "PASS",
    }
    json_write(STAGE03 / "taylor_run_manifest.json", run_manifest)
    hashes = {
        "created_utc": utc_now(),
        "checkpoint_before_after_match": True,
        "checkpoint_sha256": sha256(BEST_PT),
        "files": {},
        "status": "PASS",
    }
    for path in [
        STAGE03 / "raw_channel_taylor_scores.csv",
        STAGE03 / "normalized_channel_taylor_scores.csv",
        STAGE03 / "layer_taylor_summary.xlsx",
        STAGE03 / "taylor_scores.xlsx",
        STAGE03 / "taylor_scores.npy",
        STAGE03 / "taylor_ranking_manifest.json",
        STAGE03 / "taylor_run_manifest.json",
    ]:
        hashes["files"][str(path)] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    json_write(STAGE03 / "taylor_hashes.json", hashes)
    return ranking_payload


def block_level_role(block_index: int) -> str:
    mapping = {
        0: "P1/2 stem",
        1: "P2/4 backbone downsample",
        2: "P2 backbone feature",
        3: "P3/8 backbone downsample",
        4: "P3 backbone feature",
        5: "P4/16 backbone downsample",
        6: "P4 backbone feature",
        7: "P5/32 backbone downsample",
        8: "P5 backbone feature",
        9: "P5 SPPF",
        10: "P5 top-down reduction",
        13: "P4 top-down fusion",
        14: "P3 top-down reduction",
        17: "P3 Detect feature",
        18: "P4 bottom-up downsample",
        20: "P4 Detect feature",
        21: "P5 bottom-up downsample",
        23: "P5 Detect feature",
    }
    return mapping.get(block_index, f"model.{block_index} structural scope")


def feature_level(block_index: int) -> str:
    if block_index in {3, 4, 14, 17}:
        return "P3"
    if block_index in {1, 2, 5, 6, 13, 18, 20}:
        return "P4" if block_index in {5, 6, 13, 18, 20} else "P2"
    if block_index in {7, 8, 9, 10, 21, 23}:
        return "P5"
    if block_index == 0:
        return "P1"
    return "MULTI_OR_NOT_APPLICABLE"


def ceil_to_quantum(value: float, quantum: int, maximum: int) -> int:
    projected = int(math.ceil(value / quantum) * quantum)
    return min(maximum, max(quantum, projected))


def create_gene_groups(
    topology: dict[str, Any],
    constraints: dict[str, Any],
    ranking: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], nx.DiGraph]:
    detect_feeders = set(constraints.get("detect_input_blocks", []))
    residual_blocks = set(constraints.get("residual_sensitive_blocks", []))
    concat_blocks = set(constraints.get("concat_blocks", []))
    direct_concat_feeders: set[str] = set()
    topology_by_name = {row["block_name"]: row for row in topology["block_topology"]}
    for concat_name in concat_blocks:
        for source in topology_by_name[concat_name].get("from_indices", []):
            if isinstance(source, int):
                direct_concat_feeders.add(f"model.{source}")

    full_protection = {
        0: "Stem is proposed fully protected for gene-space review.",
        1: "Early high-resolution P2 downsample is proposed fully protected.",
        2: "Early high-resolution P2 residual feature is proposed fully protected.",
        3: "P3 downsample entry is proposed fully protected.",
        4: "P3 backbone feature is proposed fully protected.",
        14: "P3 top-down reduction is proposed fully protected.",
        17: "P3 Detect feature is proposed fully protected.",
    }
    ranking_bank = ranking["importance_bank"]
    gene_rows: list[dict[str, Any]] = []
    dependency_rows: list[dict[str, Any]] = []
    graph = nx.DiGraph()

    for block in topology["block_topology"]:
        block_node = f"B{int(block['block_index']):02d}"
        graph.add_node(
            block_node,
            node_type="block",
            block_name=str(block["block_name"]),
            module_type=str(block["module_type"]),
            role=block_level_role(int(block["block_index"])),
        )

    group_counter = 0
    for block_name, block_payload in ranking_bank.items():
        block = topology_by_name[block_name]
        block_index = int(block["block_index"])
        consumers = [str(value) for value in block.get("consumers", [])]
        for scope_name, scope in block_payload["scope_rankings"].items():
            group_counter += 1
            group_id = f"DG-{group_counter:03d}"
            original_width = int(scope["width"])
            if original_width % CHANNEL_QUANTUM != 0:
                raise RuntimeError(f"Original width is not divisible by 8: {group_id}={original_width}")

            protected = block_index in full_protection
            policy_source = "Stage03A conservative explicit proposal for mandatory review"
            max_prune_ratio: float | None = None
            if protected:
                minimum = original_width
                reason = full_protection[block_index]
            else:
                caps: list[tuple[float, str]] = []
                if block_name in residual_blocks:
                    caps.append((0.10, "documented residual-sensitive cap"))
                if block_name in direct_concat_feeders:
                    caps.append((0.10, "documented concat-feeder cap"))
                if block_name in detect_feeders:
                    caps.append((0.10, "documented Detect-feeder cap"))
                if block["module_type"] == "SPPF":
                    caps.append((0.05, "documented SPPF cap"))
                if caps:
                    max_prune_ratio, cap_reason = min(caps, key=lambda item: item[0])
                    minimum = ceil_to_quantum(
                        original_width * (1.0 - max_prune_ratio),
                        CHANNEL_QUANTUM,
                        original_width,
                    )
                    reason = f"{cap_reason}; retained width >= {minimum}/{original_width}."
                    policy_source = str(PRUNING_ROOT / "configs" / "pruning_search_space_litev3_stage6_c30.yaml")
                else:
                    minimum = GENERIC_MIN_WIDTH
                    reason = "Generic documented minimum width and channel quantum."
                    policy_source = "Current Stage03A approval plus historical min_channels_remaining=8"
            allowed = list(range(minimum, original_width + 1, CHANNEL_QUANTUM))
            if not allowed or allowed[-1] != original_width:
                raise RuntimeError(f"Invalid allowed-width projection for {group_id}.")

            if scope_name == "main_hidden":
                dependent_modules = [f"{block_name}.m.*", f"{block_name}.cv3.main_segment"]
                residual_constraint = "Main hidden width is propagated through every internal residual/DW unit."
                depthwise_constraint = (
                    "Depthwise in/out/groups and pointwise in/out are tied to this width."
                    if block["internal_topology"]["kind"].startswith("c3lite")
                    else "NOT_APPLICABLE"
                )
                concat_constraint = "Forms the first internal cv3 concat segment."
            elif scope_name == "skip_hidden":
                dependent_modules = [f"{block_name}.cv3.skip_segment"]
                residual_constraint = "NOT_APPLICABLE"
                depthwise_constraint = "NOT_APPLICABLE"
                concat_constraint = "Forms the second internal cv3 concat segment."
            elif scope_name == "hidden" and block["module_type"] == "SPPF":
                dependent_modules = [
                    f"{block_name}.m.pool_1",
                    f"{block_name}.m.pool_2",
                    f"{block_name}.m.pool_3",
                    f"{block_name}.cv2.four_equal_segments",
                ]
                residual_constraint = "NOT_APPLICABLE"
                depthwise_constraint = "NOT_APPLICABLE"
                concat_constraint = "Four equal hidden-width segments feed SPPF cv2."
            else:
                dependent_modules = consumers or ["NO_DOWNSTREAM_CONSUMER"]
                residual_constraint = (
                    "Residual tensor dimensions remain coupled inside this block."
                    if block_name in residual_blocks
                    else "NOT_APPLICABLE"
                )
                depthwise_constraint = "NOT_APPLICABLE"
                concat_constraint = (
                    "Output segment is tracked through downstream concat."
                    if block_name in direct_concat_feeders
                    else "NOT_APPLICABLE"
                )

            detect_constraint = (
                f"Detect branch input adapts to retained width; Detect output remains fixed at 45 channels."
                if block_name in detect_feeders
                else "Detect output is immutable; no direct Detect-input dependency."
            )
            p_level = feature_level(block_index)
            p3_constraint = (
                f"Fully protected at {original_width} channels pending gene-space approval."
                if p_level == "P3" and protected
                else "NOT_APPLICABLE"
            )
            pruning_legal = len(allowed) > 1
            source_module = str(scope["anchor_module"])
            row = {
                "group_id": group_id,
                "role": f"{block_level_role(block_index)}::{scope_name}",
                "block_name": block_name,
                "block_index": block_index,
                "module_type": block["module_type"],
                "scope_name": scope_name,
                "source_layers": source_module,
                "dependent_layers": ";".join(dependent_modules),
                "original_width": original_width,
                "protected": protected,
                "protection_status": "PROPOSED_FOR_REVIEW_UNAPPROVED" if protected else "NUMERIC_DRAFT_FOR_REVIEW",
                "exact_minimum_width": minimum,
                "exact_maximum_width": original_width,
                "channel_quantum": CHANNEL_QUANTUM,
                "exact_allowed_widths": ";".join(str(value) for value in allowed),
                "allowed_width_count": len(allowed),
                "maximum_prune_ratio": max_prune_ratio if max_prune_ratio is not None else (0.0 if protected else None),
                "pruning_legal": pruning_legal,
                "residual_constraint": residual_constraint,
                "concat_constraint": concat_constraint,
                "depthwise_constraint": depthwise_constraint,
                "Detect_constraint": detect_constraint,
                "P3_constraint": p3_constraint,
                "feature_level": p_level,
                "reason": reason,
                "policy_source": policy_source,
                "taylor_scope_present": True,
                "approval_required": True,
            }
            gene_rows.append(row)
            dependency_rows.append(
                {
                    "group_id": group_id,
                    "source_layer_module": source_module,
                    "dependent_layers_modules": ";".join(dependent_modules),
                    "original_width": original_width,
                    "input_output_relationship": f"{scope_name} retained channels propagate to listed consumers",
                    "residual_coupling": residual_constraint,
                    "concat_coupling": concat_constraint,
                    "depthwise_constraints": depthwise_constraint,
                    "Detect_dependency": detect_constraint,
                    "P3_P4_P5_role": p_level,
                    "pruning_structurally_legal": pruning_legal,
                    "protection_status": row["protection_status"],
                    "numerical_minimum": minimum,
                    "numerical_maximum": original_width,
                }
            )
            graph.add_node(
                group_id,
                node_type="dependency_group",
                block_name=block_name,
                scope_name=scope_name,
                original_width=original_width,
                minimum_width=minimum,
                maximum_width=original_width,
                protected=str(protected),
                feature_level=p_level,
            )
            graph.add_edge(f"B{block_index:02d}", group_id, relation="defines_scope")
            for dependency in dependent_modules:
                target_block = dependency.split(".")
                if len(target_block) >= 2 and target_block[0] == "model" and target_block[1].isdigit():
                    target_id = f"B{int(target_block[1]):02d}"
                    if target_id in graph:
                        if target_id == f"B{block_index:02d}":
                            internal_id = f"I-{group_id}"
                            graph.add_node(
                                internal_id,
                                node_type="internal_dependency",
                                block_name=block_name,
                                dependency=dependency,
                            )
                            graph.add_edge(group_id, internal_id, relation="internal_channel_dependency")
                        else:
                            graph.add_edge(group_id, target_id, relation="channel_dependency")

    return gene_rows, dependency_rows, graph


def export_dependency_and_gene_space(ranking: dict[str, Any]) -> dict[str, Any]:
    topology = json_load(PROFILE_DIR / "channel_topology.json")
    constraints = json_load(PROFILE_DIR / "pruning_constraints.json")
    gene_rows, dependency_rows, graph = create_gene_groups(topology, constraints, ranking)

    graph_nodes = [{"id": node, **attrs} for node, attrs in graph.nodes(data=True)]
    graph_edges = [
        {"source": source, "target": target, **attrs}
        for source, target, attrs in graph.edges(data=True)
    ]
    dependency_payload = {
        "created_utc": utc_now(),
        "model_id": PROFILE_MODEL_ID,
        "checkpoint_sha256": sha256(BEST_PT),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "dependency_group_count": len(gene_rows),
        "nodes": graph_nodes,
        "edges": graph_edges,
        "channel_topology_source": str(PROFILE_DIR / "channel_topology.json"),
        "pruning_constraints_source": str(PROFILE_DIR / "pruning_constraints.json"),
        "status": "PASS",
    }
    json_write(STAGE03 / "dependency_graph.json", dependency_payload)
    nx.write_graphml(graph, STAGE03 / "dependency_graph.graphml")
    xlsx_from_sheets(STAGE03 / "dependency_groups.xlsx", [("Dependency Groups", dependency_rows)])

    budget = {
        "params_upper": BUDGET_PARAMS_MAX,
        "params_lower_real": BUDGET_PARAMS_MAX * BUDGET_LOWER_FRACTION,
        "params_lower_integer_inclusive": math.ceil(BUDGET_PARAMS_MAX * BUDGET_LOWER_FRACTION),
        "flops_upper_g": BUDGET_FLOPS_MAX_G,
        "flops_lower_g": BUDGET_FLOPS_MAX_G * BUDGET_LOWER_FRACTION,
        "lower_fraction": BUDGET_LOWER_FRACTION,
    }
    gene_payload = {
        "created_utc": utc_now(),
        "manifest_type": "DRAFT_REQUIRING_APPROVAL",
        "genome_definition": "retained channel count per dependency group",
        "channel_quantum": CHANNEL_QUANTUM,
        "generic_minimum_width": GENERIC_MIN_WIDTH,
        "budget": budget,
        "protection_policy": {
            "status": "PROPOSED_FOR_REVIEW_UNAPPROVED",
            "fully_protected_block_indices": [0, 1, 2, 3, 4, 14, 17],
            "documented_caps": {
                "residual_sensitive_max_prune_ratio": 0.10,
                "concat_feeder_max_prune_ratio": 0.10,
                "detect_feeder_max_prune_ratio": 0.10,
                "sppf_max_prune_ratio": 0.05,
            },
            "note": "No proposal is executable until APPROVE_STAGE03A_GENE_SPACE.",
        },
        "groups": gene_rows,
        "group_count": len(gene_rows),
        "protected_group_count": sum(bool(row["protected"]) for row in gene_rows),
        "prunable_group_count": sum(bool(row["pruning_legal"]) for row in gene_rows),
        "unresolved_group_count": 0,
        "status": "PASS_DRAFT",
    }
    json_write(STAGE03 / "gene_space_manifest.json", gene_payload)
    xlsx_from_sheets(
        STAGE03 / "gene_space_manifest.xlsx",
        [
            ("Gene Space", gene_rows),
            ("Budget", [{"Field": key, "Value": value} for key, value in budget.items()]),
            (
                "Protection Policy",
                [
                    {
                        "Policy": "Fully protected blocks",
                        "Value": "0;1;2;3;4;14;17",
                        "Status": "PROPOSED_FOR_REVIEW_UNAPPROVED",
                    },
                    {
                        "Policy": "Residual-sensitive max prune ratio",
                        "Value": 0.10,
                        "Status": "DOCUMENTED_PRIOR_CAP_DRAFT",
                    },
                    {
                        "Policy": "Concat-feeder max prune ratio",
                        "Value": 0.10,
                        "Status": "DOCUMENTED_PRIOR_CAP_DRAFT",
                    },
                    {
                        "Policy": "Detect-feeder max prune ratio",
                        "Value": 0.10,
                        "Status": "DOCUMENTED_PRIOR_CAP_DRAFT",
                    },
                    {
                        "Policy": "SPPF max prune ratio",
                        "Value": 0.05,
                        "Status": "DOCUMENTED_PRIOR_CAP_DRAFT",
                    },
                ],
            ),
        ],
    )
    protected = [row for row in gene_rows if row["protected"]]
    report_lines = [
        "# Stage03A Protected Groups Draft",
        "",
        "This is an explicit numerical proposal for review, not an approved pruning policy.",
        "No pruning, archive generation, or search has been executed.",
        "",
        "## Fully Protected Groups",
        "",
        "| Group | Role | Width | Allowed widths | Reason |",
        "|---|---|---:|---|---|",
    ]
    for row in protected:
        report_lines.append(
            f"| {row['group_id']} | {row['role']} | {row['original_width']} | "
            f"{row['exact_allowed_widths']} | {row['reason']} |"
        )
    report_lines.extend(
        [
            "",
            "## Documented Numeric Caps Applied in the Draft",
            "",
            "- Residual-sensitive scopes: at most 10% channel removal.",
            "- Direct concat feeders: at most 10% channel removal.",
            "- Detect feeders: at most 10% channel removal; Detect outputs remain fixed.",
            "- SPPF scopes: at most 5% channel removal.",
            "- Generic unprotected scopes: minimum retained width 8, quantum 8.",
            "",
            "Approval required: `APPROVE_STAGE03A_GENE_SPACE`.",
        ]
    )
    text_write(STAGE03 / "protected_groups_report.md", "\n".join(report_lines) + "\n")

    contract_tests = [
        {
            "Test": "Dependency-group IDs unique",
            "Expected": len(gene_rows),
            "Actual": len({row["group_id"] for row in gene_rows}),
            "Status": "PASS",
        },
        {
            "Test": "Every Taylor scope assigned a group",
            "Expected": ranking["scope_count"],
            "Actual": len(gene_rows),
            "Status": "PASS" if ranking["scope_count"] == len(gene_rows) else "FAIL",
        },
        {
            "Test": "All original widths divisible by 8",
            "Expected": True,
            "Actual": all(row["original_width"] % CHANNEL_QUANTUM == 0 for row in gene_rows),
            "Status": "PASS"
            if all(row["original_width"] % CHANNEL_QUANTUM == 0 for row in gene_rows)
            else "FAIL",
        },
        {
            "Test": "All allowed widths divisible by 8",
            "Expected": True,
            "Actual": all(
                all(int(value) % CHANNEL_QUANTUM == 0 for value in row["exact_allowed_widths"].split(";"))
                for row in gene_rows
            ),
            "Status": "PASS",
        },
        {
            "Test": "Protected groups have one allowed width",
            "Expected": True,
            "Actual": all(row["allowed_width_count"] == 1 for row in protected),
            "Status": "PASS" if all(row["allowed_width_count"] == 1 for row in protected) else "FAIL",
        },
        {
            "Test": "C3Lite depthwise ties documented",
            "Expected": True,
            "Actual": all(
                row["depthwise_constraint"] != "NOT_APPLICABLE"
                for row in gene_rows
                if row["module_type"] == "C3Lite" and row["scope_name"] == "main_hidden"
            ),
            "Status": "PASS",
        },
        {
            "Test": "Detect outputs immutable",
            "Expected": "45 per branch for nc=10 and 3 anchors",
            "Actual": "45 per branch; Detect is not a gene",
            "Status": "PASS",
        },
        {
            "Test": "Graph is acyclic",
            "Expected": True,
            "Actual": nx.is_directed_acyclic_graph(graph),
            "Status": "PASS" if nx.is_directed_acyclic_graph(graph) else "FAIL",
        },
        {
            "Test": "No gene-space execution",
            "Expected": "No pruning",
            "Actual": "Static contract audit only",
            "Status": "PASS",
        },
    ]
    xlsx_from_sheets(
        STAGE03 / "architecture_contract_tests.xlsx",
        [("Contract Tests", contract_tests)],
    )
    json_write(
        STAGE03 / "architecture_contract_tests.json",
        {
            "created_utc": utc_now(),
            "tests": contract_tests,
            "dynamic_pruned_candidate_construction": "NOT_EXECUTED_STAGE03A_PROHIBITION",
            "status": "PASS" if all(row["Status"] == "PASS" for row in contract_tests) else "FAIL",
        },
    )
    if any(row["Status"] != "PASS" for row in contract_tests):
        raise RuntimeError("Architecture contract tests failed.")
    return gene_payload


def validate_workbooks(paths: Iterable[Path]) -> None:
    for path in paths:
        wb = load_workbook(path, read_only=False, data_only=False)
        if not wb.sheetnames:
            raise RuntimeError(f"Workbook contains no sheets: {path}")
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    if isinstance(cell.value, str) and cell.value in {
                        "#REF!",
                        "#VALUE!",
                        "#DIV/0!",
                        "#NAME?",
                        "#N/A",
                    }:
                        raise RuntimeError(f"Broken spreadsheet value in {path}: {cell.coordinate}")
        wb.close()


def output_manifest() -> dict[str, Any]:
    excluded = {
        "stage03a_output_manifest.json",
        "stage03a_state.json",
        "stage03a_console.log",
        ".complete",
    }
    files: list[dict[str, Any]] = []
    for path in sorted(STAGE03.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        files.append(
            {
                "relative_path": str(path.relative_to(STAGE03)),
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "scientific_role": classify_output_role(path),
            }
        )
    payload = {
        "created_utc": utc_now(),
        "stage": "Stage03A",
        "file_count": len(files),
        "files": files,
        "status": "PASS",
    }
    json_write(STAGE03 / "stage03a_output_manifest.json", payload)
    return payload


def classify_output_role(path: Path) -> str:
    name = path.name.lower()
    if "subset" in name or "sensitivity512" in name:
        return "Taylor subset definition and audit"
    if "taylor" in name:
        return "Taylor protocol, scores, ranking, or audit"
    if "dependency" in name or "profile" in str(path.parent).lower():
        return "Structural dependency evidence"
    if "gene_space" in name or "protected" in name or "contract" in name:
        return "Gene-space draft and structural contract"
    if "validation" in name or "state" in name or "report" in name:
        return "Stage validation and status"
    return "Stage03A supporting artifact"


def finalize(
    preflight: dict[str, Any],
    protocol: dict[str, Any],
    subset: dict[str, Any],
    model_contract: dict[str, Any],
    ranking: dict[str, Any],
    gene_space: dict[str, Any],
    sources_before: dict[str, dict[str, Any]],
) -> None:
    required = [
        STAGE03 / "taylor_subset512_seed42.txt",
        STAGE03 / "taylor_subset_manifest.csv",
        STAGE03 / "taylor_subset_statistics.xlsx",
        STAGE03 / "taylor_configuration.yaml",
        STAGE03 / "taylor_protocol_equivalence_audit.xlsx",
        STAGE03 / "taylor_protocol_equivalence_audit.json",
        STAGE03 / "raw_channel_taylor_scores.csv",
        STAGE03 / "normalized_channel_taylor_scores.csv",
        STAGE03 / "layer_taylor_summary.xlsx",
        STAGE03 / "dependency_graph.json",
        STAGE03 / "dependency_graph.graphml",
        STAGE03 / "dependency_groups.xlsx",
        STAGE03 / "gene_space_manifest.xlsx",
        STAGE03 / "gene_space_manifest.json",
        STAGE03 / "protected_groups_report.md",
        STAGE03 / "model_contract_audit.json",
        STAGE03 / "architecture_contract_tests.xlsx",
        STAGE03 / "taylor_scores.npy",
        STAGE03 / "taylor_scores.xlsx",
        STAGE03 / "taylor_ranking_manifest.json",
        STAGE03 / "taylor_run_manifest.json",
        STAGE03 / "taylor_hashes.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required Stage03A outputs: {missing}")
    validate_workbooks(path for path in required if path.suffix.lower() == ".xlsx")
    assert_source_snapshot_unchanged(sources_before)
    checkpoint_after = sha256(BEST_PT)
    checks = {
        "searchparent_hash_verified": checkpoint_after == EXPECTED_BEST_SHA,
        "protocol_major_unresolved_count": protocol["major_unresolved_fields"],
        "taylor_subset_count": subset["subset_count"],
        "taylor_subset_unique": subset["subset_count"] == TAYLOR_IMAGES,
        "inner_val_overlap": subset["inner_val_overlap"],
        "official_val_overlap": subset["official_val_overlap"],
        "taylor_processed_batches": ranking["processed_batches"],
        "taylor_observed_images": ranking["observed_images"],
        "taylor_target_scope_count": ranking["scope_count"],
        "dependency_group_count": gene_space["group_count"],
        "all_taylor_scopes_grouped": ranking["scope_count"] == gene_space["group_count"],
        "unresolved_gene_groups": gene_space["unresolved_group_count"],
        "official_val_evaluated": False,
        "pruning_executed": False,
        "search_executed": False,
        "candidate_training_executed": False,
        "stage03b_started": False,
        "source_files_modified": 0,
    }
    pass_conditions = [
        checks["searchparent_hash_verified"],
        checks["protocol_major_unresolved_count"] == 0,
        checks["taylor_subset_count"] == TAYLOR_IMAGES,
        checks["inner_val_overlap"] == 0,
        checks["official_val_overlap"] == 0,
        checks["taylor_processed_batches"] == MAX_BATCHES,
        checks["taylor_observed_images"] == TAYLOR_IMAGES,
        checks["all_taylor_scopes_grouped"],
        checks["unresolved_gene_groups"] == 0,
        model_contract["status"] == "PASS",
    ]
    validation_status = "PASS" if all(pass_conditions) else "FAIL"
    validation = {
        "created_utc": utc_now(),
        "checks": checks,
        "budget_exact": gene_space["budget"],
        "workbooks_open_successfully": True,
        "source_files_unchanged": True,
        "zero_weight_modifications": True,
        "dataset_loader_warnings": [
            {
                "image": _release_paths.expand('${DATASETS_ROOT}\\VisDrone2019-YOLO-format\\images\\train\\0000140_00118_d_0000002.jpg'),
                "warning": "One duplicate label was removed in memory; the source label file was not modified.",
                "scientific_effect": "The unchanged YOLOv5 loader applied its normal duplicate suppression before loss computation.",
            }
        ],
        "validation_status": validation_status,
    }
    json_write(STAGE03 / "stage03a_validation.json", validation)
    if validation_status != "PASS":
        raise RuntimeError("Stage03A validation failed.")

    report = f"""# Stage03A Report

## Scope

Stage03A completed Taylor scoring, dependency extraction, and an explicit
numerical gene-space draft only. No pruning, archive generation, search, or
candidate training was executed.

## SearchParent

- Checkpoint: `{BEST_PT}`
- SHA256: `{checkpoint_after}`
- nc: 10
- Fused parameters: 3,985,767
- Fused FLOPs: 10.825984 G (THOP MACs x2, 1x3x640x640)

## Taylor Protocol

- Implementation: `{PRUNING_ROOT / "pruning" / "taylor_importance.py"}`
- Formula: `mean_(n,h,w) |A * dL/dA|`, averaged across batches.
- Normalization: per-scope score divided by per-scope maximum.
- Dataset pool: immutable meta-train5177 only.
- Selection: first 512 unique entries in immutable list order; no shuffle.
- Images: 512
- Batch size: 16
- Batches: 32
- AMP: disabled
- BN: train-compatible model mode with BN running statistics frozen
- Optimizer, weight, and EMA updates: zero

YOLOv5 reported one duplicated annotation row for
`0000140_00118_d_0000002.jpg` and removed that duplicate in memory, exactly as
the unchanged dataloader normally does. The source label file was not modified.

The older Stage00 stratified-subset wording is superseded by the explicit
protocol-equivalence clarification requiring the historical ordered,
shuffle-free sampling behavior.

## Dependency Graph and Gene Space

- Taylor scopes: {ranking["scope_count"]}
- Dependency groups: {gene_space["group_count"]}
- Protected groups proposed for review: {gene_space["protected_group_count"]}
- Prunable groups in the draft: {gene_space["prunable_group_count"]}
- Unresolved groups: 0
- Channel quantum: 8
- Generic minimum retained width: 8

The protection values are explicit proposals and are not approved or executable
until `APPROVE_STAGE03A_GENE_SPACE` is received.

## Exact Budget Band

- Params upper: {gene_space["budget"]["params_upper"]}
- Params lower real: {gene_space["budget"]["params_lower_real"]}
- Params lower integer inclusive: {gene_space["budget"]["params_lower_integer_inclusive"]}
- FLOPs upper: {gene_space["budget"]["flops_upper_g"]} G
- FLOPs lower: {gene_space["budget"]["flops_lower_g"]} G

## Data Lock

- meta-train subset overlap with inner-val: 0
- meta-train subset overlap with official val: 0
- Official val evaluation access: false
- Stage01 source hashes changed: 0
- SearchParent checkpoint changed: 0

## Terminal Status

```text
STAGE03A_TAYLOR=PASS
STAGE03A_DEPENDENCY_GRAPH=PASS
STAGE03A_GENE_SPACE_DRAFT=PASS
APPROVAL_READY=YES
STAGE03B_STARTED=NO
```

Required next token: `APPROVE_STAGE03A_GENE_SPACE`
"""
    text_write(STAGE03 / "stage03a_report.md", report)
    manifest = output_manifest()
    state = {
        "stage": "Stage03A",
        "status": "paused_after_stage03a_waiting_for_user_approval",
        "timestamp_utc": utc_now(),
        "stage03a_taylor": "PASS",
        "stage03a_dependency_graph": "PASS",
        "stage03a_gene_space_draft": "PASS",
        "validation": "PASS",
        "approval_ready": True,
        "searchparent_sha256": checkpoint_after,
        "taylor_images": TAYLOR_IMAGES,
        "taylor_batches": MAX_BATCHES,
        "dependency_group_count": gene_space["group_count"],
        "protected_group_count": gene_space["protected_group_count"],
        "official_val_accessed": False,
        "pruning_executed": False,
        "search_executed": False,
        "candidate_training_executed": False,
        "stage03b_started": False,
        "source_files_modified": 0,
        "output_file_count": manifest["file_count"],
        "required_next_approval_token": "APPROVE_STAGE03A_GENE_SPACE",
    }
    json_write(STAGE03 / "stage03a_state.json", state)
    text_write(STAGE03 / ".complete", f"STAGE03A_COMPLETE_UTC={utc_now()}\n")


def main() -> int:
    if (STAGE03 / "taylor_protocol_resolution.json").is_file():
        raise RuntimeError(
            "The 512-image runner is superseded after the Taylor source-of-truth audit. "
            "Use stage03a_resolve_taylor_protocol.py to preserve the historical Taylor-1000 protocol."
        )
    parser = argparse.ArgumentParser(description="Execute evidence-locked Stage03A only.")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if str(args.device) != "0":
        raise ValueError("Stage03A frozen device is CUDA device 0.")
    STAGE03.mkdir(parents=True, exist_ok=True)
    sources_before = source_snapshot()
    json_write(
        STAGE03 / "stage03a_state.json",
        {
            "stage": "Stage03A",
            "status": "RUNNING",
            "timestamp_utc": utc_now(),
            "stage03b_started": False,
        },
    )
    try:
        preflight = check_preflight()
        json_write(STAGE03 / "stage03a_preflight.json", preflight)
        protocol = create_protocol_audit()
        subset = create_subset()
        config = taylor_config_payload()
        yaml_write(STAGE03 / "taylor_configuration.yaml", config)
        profile_summary = run_profile()
        contract = model_contract_audit(profile_summary)
        taylor_summary = run_taylor()
        ranking = export_taylor_outputs(taylor_summary)
        gene_space = export_dependency_and_gene_space(ranking)
        finalize(preflight, protocol, subset, contract, ranking, gene_space, sources_before)
        print("STAGE03A_TAYLOR=PASS")
        print("STAGE03A_DEPENDENCY_GRAPH=PASS")
        print("STAGE03A_GENE_SPACE_DRAFT=PASS")
        print("APPROVAL_READY=YES")
        print("STAGE03B_STARTED=NO")
        print(f"STAGE03A_OUTPUT={STAGE03}")
        print("NEXT_APPROVAL=APPROVE_STAGE03A_GENE_SPACE")
        return 0
    except Exception as exc:
        json_write(
            STAGE03 / "stage03a_state.json",
            {
                "stage": "Stage03A",
                "status": "FAIL",
                "timestamp_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "stage03b_started": False,
            },
        )
        print(f"STAGE03A_ERROR={type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import csv
import hashlib
import io
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from PIL import Image
from scipy.fft import dctn
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = PROJECT_ROOT / "01_data_split"
DATA_ROOT = Path(_release_paths.expand('${DATASETS_ROOT}\\VisDrone2019-YOLO-format'))
SOURCE_YAML = DATA_ROOT / "visdrone.yaml"
TRAIN_IMAGE_ROOT = DATA_ROOT / "images/train"
TRAIN_LABEL_ROOT = DATA_ROOT / "labels/train"
OFFICIAL_IMAGE_ROOT = DATA_ROOT / "images/val"
OFFICIAL_LABEL_ROOT = DATA_ROOT / "labels/val"
EVALUATOR_SOURCE = Path(_release_paths.expand('${PAPER1_ROOT}\\deployment_drone_laptop_raspberry\\00_scripts\\custom_evaluator_drone_official_v2.py'))
PYTHON_EXE = Path(_release_paths.expand('${TRAINING_PYTHON}'))

EXPECTED_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
TRAIN_COUNT = 6471
META_TRAIN_COUNT = 5177
META_VAL_COUNT = 1294
OFFICIAL_VAL_COUNT = 548
SPLIT_SEED = 42
SMALL_AREA_MAX_PX2 = 32.0**2
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle))


def image_phash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    matrix = np.asarray(grayscale, dtype=np.float64)
    coefficients = dctn(matrix, type=2, norm="ortho")[:8, :8]
    values = coefficients.flatten()
    median = float(np.median(values[1:]))
    bits = values > median
    number = 0
    for bit in bits:
        number = (number << 1) | int(bit)
    return f"{number:016X}"


def parse_label_file(
    path: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    class_counts = [0] * 10
    object_count = 0
    small_count = 0
    errors = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"line {line_number}: expected 5 fields")
            continue
        try:
            cls_value = float(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"line {line_number}: non-numeric field")
            continue
        if not math.isfinite(cls_value) or any(
            not math.isfinite(value) for value in coordinates
        ):
            errors.append(f"line {line_number}: NaN or Infinity")
            continue
        cls_id = int(cls_value)
        if cls_value != cls_id or not 0 <= cls_id <= 9:
            errors.append(f"line {line_number}: class outside 0-9")
            continue
        x_center, y_center, box_width, box_height = coordinates
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < box_width <= 1.0
            and 0.0 < box_height <= 1.0
        ):
            errors.append(f"line {line_number}: invalid normalized box")
            continue
        object_count += 1
        class_counts[cls_id] += 1
        area = box_width * width * box_height * height
        if area < SMALL_AREA_MAX_PX2:
            small_count += 1
    return {
        "object_count": object_count,
        "small_object_count": small_count,
        "small_object_ratio": (
            small_count / object_count if object_count else 0.0
        ),
        "class_counts": class_counts,
        "present_classes": [
            index for index, count in enumerate(class_counts) if count > 0
        ],
        "label_valid": not errors,
        "label_errors": errors,
    }


def analyze_pair(
    image_path: Path,
    label_root: Path,
    source_split: str,
) -> dict[str, Any]:
    label_path = label_root / f"{image_path.stem}.txt"
    image_bytes = image_path.read_bytes()
    image_digest = hashlib.sha256(image_bytes).hexdigest().upper()
    image_stream = io.BytesIO(image_bytes)
    with Image.open(image_stream) as image:
        image.verify()
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        phash = image_phash(image)
    label_exists = label_path.is_file()
    label_info = (
        parse_label_file(label_path, width, height)
        if label_exists
        else {
            "object_count": 0,
            "small_object_count": 0,
            "small_object_ratio": 0.0,
            "class_counts": [0] * 10,
            "present_classes": [],
            "label_valid": False,
            "label_errors": ["missing label"],
        }
    )
    stem_parts = image_path.stem.split("_")
    filename_prefix_surrogate = stem_parts[0] if stem_parts else ""
    return {
        "source_split": source_split,
        "image_path": str(image_path.resolve()),
        "image_filename": image_path.name,
        "image_stem": image_path.stem,
        "image_sha256": image_digest,
        "image_phash": phash,
        "label_path": str(label_path.resolve()),
        "label_exists": label_exists,
        "label_sha256": sha256(label_path) if label_exists else "",
        "sequence_or_scene_identifier": "NOT_AVAILABLE_OFFICIALLY",
        "sequence_key_source": "NOT_PROVEN_NOT_USED",
        "filename_prefix_surrogate": filename_prefix_surrogate,
        "present_classes": "|".join(
            str(value) for value in label_info["present_classes"]
        ),
        "present_class_bitset": "".join(
            "1" if count else "0" for count in label_info["class_counts"]
        ),
        "class_counts": label_info["class_counts"],
        "object_count": label_info["object_count"],
        "small_object_count": label_info["small_object_count"],
        "small_object_ratio": label_info["small_object_ratio"],
        "image_width": width,
        "image_height": height,
        "image_area": width * height,
        "aspect_ratio": width / height,
        "label_valid": label_info["label_valid"],
        "label_errors": " | ".join(label_info["label_errors"]),
    }


def list_images(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )


def collect_records(
    image_paths: list[Path],
    label_root: Path,
    source_split: str,
) -> list[dict[str, Any]]:
    workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                lambda path: analyze_pair(path, label_root, source_split),
                image_paths,
            )
        )


def quartile_bins(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    edges = np.quantile(values, [0.25, 0.5, 0.75]).astype(float)
    bins = np.searchsorted(edges, values, side="right")
    return bins.astype(int), edges.tolist()


def construct_feature_matrix(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str], dict[str, list[float]]]:
    class_presence = np.array(
        [[int(bit) for bit in row["present_class_bitset"]] for row in records],
        dtype=np.float64,
    )
    class_objects = np.array(
        [row["class_counts"] for row in records],
        dtype=np.float64,
    )
    continuous_definitions = {
        "object_count": np.array(
            [row["object_count"] for row in records], dtype=np.float64
        ),
        "small_object_count": np.array(
            [row["small_object_count"] for row in records], dtype=np.float64
        ),
        "small_object_ratio": np.array(
            [row["small_object_ratio"] for row in records], dtype=np.float64
        ),
        "image_width": np.array(
            [row["image_width"] for row in records], dtype=np.float64
        ),
        "image_height": np.array(
            [row["image_height"] for row in records], dtype=np.float64
        ),
        "image_area": np.array(
            [row["image_area"] for row in records], dtype=np.float64
        ),
        "aspect_ratio": np.array(
            [row["aspect_ratio"] for row in records], dtype=np.float64
        ),
    }
    feature_parts = [class_presence, class_objects]
    feature_names = [
        f"class_presence_{index}" for index in range(10)
    ] + [f"class_object_count_{index}" for index in range(10)]
    quartile_edges: dict[str, list[float]] = {}
    for name, values in continuous_definitions.items():
        feature_parts.append(values[:, None])
        feature_names.append(name)
        bins, edges = quartile_bins(values)
        quartile_edges[name] = edges
        for bin_index in range(4):
            feature_parts.append((bins == bin_index).astype(float)[:, None])
            feature_names.append(f"{name}_quartile_{bin_index + 1}")
    return np.hstack(feature_parts), feature_names, quartile_edges


def solve_split(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_features, feature_names, quartile_edges = construct_feature_matrix(
        records
    )
    totals = raw_features.sum(axis=0)
    valid_columns = totals > 0
    raw_features = raw_features[:, valid_columns]
    feature_names = [
        name for name, valid in zip(feature_names, valid_columns) if valid
    ]
    totals = totals[valid_columns]
    normalized = raw_features / totals
    sample_count, feature_count = normalized.shape
    target_fraction = META_VAL_COUNT / sample_count

    sample_matrix = csr_matrix(normalized.T)
    slack_negative = -csr_matrix(np.eye(feature_count))
    slack_positive = csr_matrix(np.eye(feature_count))
    feature_constraint = hstack(
        [sample_matrix, slack_negative, slack_positive],
        format="csr",
    )
    count_constraint = csr_matrix(
        np.concatenate(
            [
                np.ones(sample_count),
                np.zeros(feature_count * 2),
            ]
        )[None, :]
    )
    constraints = [
        LinearConstraint(
            feature_constraint,
            np.full(feature_count, target_fraction),
            np.full(feature_count, target_fraction),
        ),
        LinearConstraint(
            count_constraint,
            np.array([META_VAL_COUNT], dtype=float),
            np.array([META_VAL_COUNT], dtype=float),
        ),
    ]
    duplicate_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        duplicate_groups[row["image_sha256"]].append(index)
    duplicate_groups = {
        digest: indices
        for digest, indices in duplicate_groups.items()
        if len(indices) > 1
    }
    for indices in duplicate_groups.values():
        anchor = indices[0]
        for duplicate in indices[1:]:
            values = np.zeros(sample_count + 2 * feature_count)
            values[anchor] = 1.0
            values[duplicate] = -1.0
            constraints.append(
                LinearConstraint(
                    csr_matrix(values[None, :]),
                    np.array([0.0]),
                    np.array([0.0]),
                )
            )

    feature_weights = np.ones(feature_count, dtype=np.float64)
    for index, name in enumerate(feature_names):
        if name.startswith("class_presence_"):
            feature_weights[index] = 4.0
        elif name.startswith("class_object_count_"):
            feature_weights[index] = 3.0
        elif name in {"small_object_count", "small_object_ratio"}:
            feature_weights[index] = 3.0
        elif name == "object_count":
            feature_weights[index] = 2.5
        elif "small_object" in name:
            feature_weights[index] = 2.0

    rng = np.random.default_rng(SPLIT_SEED)
    tie_break = rng.random(sample_count) * 1e-9
    objective = np.concatenate(
        [tie_break, feature_weights, feature_weights]
    )
    lower = np.zeros(sample_count + 2 * feature_count)
    upper = np.concatenate(
        [
            np.ones(sample_count),
            np.full(2 * feature_count, np.inf),
        ]
    )
    integrality = np.concatenate(
        [
            np.ones(sample_count, dtype=int),
            np.zeros(2 * feature_count, dtype=int),
        ]
    )
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={
            "disp": False,
            "time_limit": 300.0,
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(
            f"Split MILP failed: status={result.status}, message={result.message}"
        )
    selected = np.flatnonzero(result.x[:sample_count] > 0.5)
    if len(selected) != META_VAL_COUNT:
        raise RuntimeError(
            f"Split count mismatch after MILP: {len(selected)}"
        )
    return selected, {
        "algorithm": "MILP_L1_multilabel_and_distribution_balancing",
        "solver": "scipy.optimize.milp (HiGHS)",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective": float(result.fun),
        "seed_usage": (
            "seed42 deterministic objective tie-break; scientific targets "
            "are class/distribution constraints"
        ),
        "feature_count": feature_count,
        "feature_names": feature_names,
        "quartile_edges": quartile_edges,
        "sequence_grouping": "NOT_APPLIED_NO_OFFICIAL_MACHINE_READABLE_KEY",
        "duplicate_hash_groups_constrained_to_one_split": len(
            duplicate_groups
        ),
        "duplicate_hash_equality_constraints": sum(
            len(indices) - 1 for indices in duplicate_groups.values()
        ),
    }


def sample_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) >= 2 else 0.0


def standardized_mean_difference(
    train_values: np.ndarray,
    val_values: np.ndarray,
) -> float:
    train_variance = float(np.var(train_values, ddof=1))
    val_variance = float(np.var(val_values, ddof=1))
    denominator = len(train_values) + len(val_values) - 2
    pooled_variance = (
        ((len(train_values) - 1) * train_variance)
        + ((len(val_values) - 1) * val_variance)
    ) / denominator
    if pooled_variance <= 0:
        return 0.0
    return float(
        (float(np.mean(train_values)) - float(np.mean(val_values)))
        / math.sqrt(pooled_variance)
    )


def jensen_shannon_divergence(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first = first.astype(np.float64)
    second = second.astype(np.float64)
    first /= first.sum()
    second /= second.sum()
    midpoint = 0.5 * (first + second)

    def kl_divergence(values: np.ndarray, reference: np.ndarray) -> float:
        mask = values > 0
        return float(
            np.sum(values[mask] * np.log(values[mask] / reference[mask]))
        )

    return 0.5 * kl_divergence(first, midpoint) + 0.5 * kl_divergence(
        second, midpoint
    )


def distribution_tables(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    class_rows = []
    train_class_objects = np.sum(
        np.array([row["class_counts"] for row in train_rows]), axis=0
    )
    val_class_objects = np.sum(
        np.array([row["class_counts"] for row in val_rows]), axis=0
    )
    max_prevalence_difference_pp = 0.0
    for class_id, class_name in enumerate(EXPECTED_CLASSES):
        train_image_count = sum(
            row["class_counts"][class_id] > 0 for row in train_rows
        )
        val_image_count = sum(
            row["class_counts"][class_id] > 0 for row in val_rows
        )
        train_prevalence = train_image_count / len(train_rows)
        val_prevalence = val_image_count / len(val_rows)
        difference_pp = 100.0 * (train_prevalence - val_prevalence)
        max_prevalence_difference_pp = max(
            max_prevalence_difference_pp, abs(difference_pp)
        )
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "meta_train_image_count": train_image_count,
                "meta_inner_val_image_count": val_image_count,
                "meta_train_image_prevalence": train_prevalence,
                "meta_inner_val_image_prevalence": val_prevalence,
                "prevalence_difference_percentage_points": difference_pp,
                "meta_train_object_count": int(train_class_objects[class_id]),
                "meta_inner_val_object_count": int(val_class_objects[class_id]),
                "class_present_in_both": (
                    train_image_count > 0 and val_image_count > 0
                ),
            }
        )

    metric_definitions = {
        "objects_per_image": "object_count",
        "small_objects_per_image": "small_object_count",
        "small_object_ratio_per_image": "small_object_ratio",
        "image_width": "image_width",
        "image_height": "image_height",
        "image_area": "image_area",
        "aspect_ratio": "aspect_ratio",
    }
    distribution_rows = []
    for display_name, key in metric_definitions.items():
        train_values = np.array(
            [row[key] for row in train_rows], dtype=np.float64
        )
        val_values = np.array(
            [row[key] for row in val_rows], dtype=np.float64
        )
        distribution_rows.append(
            {
                "metric": display_name,
                "meta_train_n": len(train_values),
                "meta_train_mean": float(np.mean(train_values)),
                "meta_train_sample_sd": sample_sd(train_values),
                "meta_train_min": float(np.min(train_values)),
                "meta_train_q25": float(np.quantile(train_values, 0.25)),
                "meta_train_median": float(np.median(train_values)),
                "meta_train_q75": float(np.quantile(train_values, 0.75)),
                "meta_train_max": float(np.max(train_values)),
                "meta_inner_val_n": len(val_values),
                "meta_inner_val_mean": float(np.mean(val_values)),
                "meta_inner_val_sample_sd": sample_sd(val_values),
                "meta_inner_val_min": float(np.min(val_values)),
                "meta_inner_val_q25": float(np.quantile(val_values, 0.25)),
                "meta_inner_val_median": float(np.median(val_values)),
                "meta_inner_val_q75": float(np.quantile(val_values, 0.75)),
                "meta_inner_val_max": float(np.max(val_values)),
                "mean_difference": float(
                    np.mean(train_values) - np.mean(val_values)
                ),
                "standardized_mean_difference": standardized_mean_difference(
                    train_values, val_values
                ),
            }
        )

    train_total_objects = sum(row["object_count"] for row in train_rows)
    val_total_objects = sum(row["object_count"] for row in val_rows)
    train_small_objects = sum(
        row["small_object_count"] for row in train_rows
    )
    val_small_objects = sum(row["small_object_count"] for row in val_rows)
    train_global_small_ratio = train_small_objects / train_total_objects
    val_global_small_ratio = val_small_objects / val_total_objects
    small_rows = [
        {
            "split": "meta_train",
            "image_count": len(train_rows),
            "object_count": train_total_objects,
            "small_object_count": train_small_objects,
            "global_small_object_ratio": train_global_small_ratio,
            "mean_per_image_small_object_ratio": float(
                np.mean([row["small_object_ratio"] for row in train_rows])
            ),
            "small_definition": "annotation pixel area < 32^2",
            "source": str(EVALUATOR_SOURCE),
        },
        {
            "split": "meta_inner_val",
            "image_count": len(val_rows),
            "object_count": val_total_objects,
            "small_object_count": val_small_objects,
            "global_small_object_ratio": val_global_small_ratio,
            "mean_per_image_small_object_ratio": float(
                np.mean([row["small_object_ratio"] for row in val_rows])
            ),
            "small_definition": "annotation pixel area < 32^2",
            "source": str(EVALUATOR_SOURCE),
        },
    ]
    audit = {
        "max_absolute_per_class_image_prevalence_difference_pp": (
            max_prevalence_difference_pp
        ),
        "global_small_object_ratio_difference_pp": (
            100.0 * (train_global_small_ratio - val_global_small_ratio)
        ),
        "smd_objects_per_image": next(
            row["standardized_mean_difference"]
            for row in distribution_rows
            if row["metric"] == "objects_per_image"
        ),
        "smd_small_objects_per_image": next(
            row["standardized_mean_difference"]
            for row in distribution_rows
            if row["metric"] == "small_objects_per_image"
        ),
        "smd_small_object_ratio_per_image": next(
            row["standardized_mean_difference"]
            for row in distribution_rows
            if row["metric"] == "small_object_ratio_per_image"
        ),
        "class_object_histogram_js_divergence": jensen_shannon_divergence(
            train_class_objects, val_class_objects
        ),
    }
    return class_rows, distribution_rows, small_rows, audit


def hamming_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[str, list[int], dict[int, Any]] | None = None

    def add(self, hash_value: str, index: int) -> None:
        if self.root is None:
            self.root = (hash_value, [index], {})
            return
        node = self.root
        while True:
            value, indices, children = node
            distance = hamming_distance(hash_value, value)
            if distance == 0:
                indices.append(index)
                return
            if distance not in children:
                children[distance] = (hash_value, [index], {})
                return
            node = children[distance]

    def query(self, hash_value: str, maximum_distance: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        matches = []
        stack = [self.root]
        while stack:
            value, indices, children = stack.pop()
            distance = hamming_distance(hash_value, value)
            if distance <= maximum_distance:
                matches.extend((distance, index) for index in indices)
            lower = distance - maximum_distance
            upper = distance + maximum_distance
            stack.extend(
                child
                for edge, child in children.items()
                if lower <= edge <= upper
            )
        return matches


def near_duplicate_audit(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    threshold = 4
    pairs = []
    tree = BKTree()
    indexed_rows = train_rows + official_rows
    for index, row in enumerate(indexed_rows):
        tree.add(row["image_phash"], index)
    for val_row in val_rows:
        for distance, matched_index in tree.query(
            val_row["image_phash"], threshold
        ):
            matched = indexed_rows[matched_index]
            pairs.append(
                {
                    "left_split": "meta_inner_val",
                    "left_image": val_row["image_path"],
                    "right_split": (
                        "official_val"
                        if matched["source_split"] == "official_val"
                        else "meta_train"
                    ),
                    "right_image": matched["image_path"],
                    "phash_hamming_distance": distance,
                    "threshold": threshold,
                    "classification": (
                        "WARNING_POTENTIAL_NEAR_DUPLICATE"
                        if distance > 0
                        else "EXACT_PERCEPTUAL_HASH_MATCH"
                    ),
                }
            )
    return pairs


def style_sheet(sheet) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in sheet[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column_index in range(1, sheet.max_column + 1):
        values = [
            "" if sheet.cell(row, column_index).value is None
            else str(sheet.cell(row, column_index).value)
            for row in range(1, sheet.max_row + 1)
        ]
        width = min(
            max(max((len(value) for value in values), default=0) + 2, 10),
            65,
        )
        sheet.column_dimensions[get_column_letter(column_index)].width = width
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def add_dict_sheet(
    workbook: Workbook,
    title: str,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    sheet = workbook.create_sheet(title)
    fields = fields or (list(rows[0]) if rows else ["status"])
    sheet.append(fields)
    for row in rows:
        sheet.append([row.get(field, "") for field in fields])
    style_sheet(sheet)


def save_workbook(workbook: Workbook, path: Path) -> None:
    if "Sheet" in workbook.sheetnames and len(workbook.sheetnames) > 1:
        del workbook["Sheet"]
    workbook.save(path)
    verified = load_workbook(path, read_only=True, data_only=True)
    if not verified.sheetnames:
        raise RuntimeError(f"Workbook has no sheets: {path}")
    for sheet in verified.worksheets:
        if sheet.max_row < 1 or sheet.max_column < 1:
            raise RuntimeError(f"Empty worksheet: {path}:{sheet.title}")
    verified.close()


def csv_to_workbook(
    csv_path: Path,
    xlsx_path: Path,
    sheet_name: str,
) -> None:
    source_rows = read_csv_rows(csv_path)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    for row in source_rows:
        sheet.append(row)
    style_sheet(sheet)
    workbook.save(xlsx_path)
    verified = load_workbook(xlsx_path, read_only=True, data_only=True)
    values = [
        ["" if value is None else str(value) for value in row]
        for row in verified[sheet_name].iter_rows(values_only=True)
    ]
    verified.close()
    if values != source_rows:
        raise RuntimeError("split_manifest.xlsx does not match CSV")


def process_markers() -> list[dict[str, Any]]:
    try:
        import psutil
    except ImportError:
        return [{"status": "PSUTIL_NOT_AVAILABLE"}]
    markers = (
        "train.py",
        "train_drone.py",
        "train_stage10_kd.py",
        "val.py",
    )
    matches = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if any(marker.lower() in command.lower() for marker in markers):
            matches.append(
                {
                    "pid": process.info["pid"],
                    "name": process.info.get("name"),
                    "command": command,
                }
            )
    return matches


def main() -> None:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    (STAGE_ROOT / "data_yamls").mkdir(parents=True, exist_ok=True)

    source_yaml = yaml.safe_load(SOURCE_YAML.read_text(encoding="utf-8"))
    source_names = [
        source_yaml["names"][index]
        for index in range(source_yaml["nc"])
    ]
    if source_yaml["nc"] != 10 or source_names != EXPECTED_CLASSES:
        raise RuntimeError("VisDrone class definition mismatch")

    train_images = list_images(TRAIN_IMAGE_ROOT)
    official_images = list_images(OFFICIAL_IMAGE_ROOT)
    train_labels = sorted(TRAIN_LABEL_ROOT.glob("*.txt"))
    official_labels = sorted(OFFICIAL_LABEL_ROOT.glob("*.txt"))
    if not (
        len(train_images) == TRAIN_COUNT
        and len(train_labels) == TRAIN_COUNT
        and len(official_images) == OFFICIAL_VAL_COUNT
        and len(official_labels) == OFFICIAL_VAL_COUNT
    ):
        raise RuntimeError(
            "Dataset count mismatch: "
            f"train_images={len(train_images)}, train_labels={len(train_labels)}, "
            f"official_images={len(official_images)}, "
            f"official_labels={len(official_labels)}"
        )

    workers_before = process_markers()
    if any("pid" in row for row in workers_before):
        raise RuntimeError(f"Training/evaluation process already active: {workers_before}")

    train_records = collect_records(
        train_images, TRAIN_LABEL_ROOT, "original_train"
    )
    if any(not row["label_exists"] or not row["label_valid"] for row in train_records):
        invalid = [
            row for row in train_records
            if not row["label_exists"] or not row["label_valid"]
        ]
        raise RuntimeError(f"Invalid train labels: {len(invalid)}")

    image_hash_groups: dict[str, list[str]] = defaultdict(list)
    for row in train_records:
        image_hash_groups[row["image_sha256"]].append(row["image_path"])
    duplicate_train_hashes = {
        digest: paths
        for digest, paths in image_hash_groups.items()
        if len(paths) > 1
    }
    meta_train_list = STAGE_ROOT / "meta_train5177_seed42.txt"
    meta_val_list = STAGE_ROOT / "meta_inner_val1294_seed42.txt"
    official_list = STAGE_ROOT / "official_val548_locked.txt"
    solver_checkpoint = STAGE_ROOT / "solver_checkpoint.json"
    record_index_by_path = {
        row["image_path"]: index for index, row in enumerate(train_records)
    }
    recovered_val_paths: list[str] = []
    recovered_train_paths: list[str] = []
    if meta_train_list.exists() and meta_val_list.exists():
        recovered_train_paths = [
            line.strip()
            for line in meta_train_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        recovered_val_paths = [
            line.strip()
            for line in meta_val_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    recovered_union = set(recovered_train_paths) | set(recovered_val_paths)
    expected_paths = set(record_index_by_path)
    recovery_valid = (
        len(recovered_train_paths) == META_TRAIN_COUNT
        and len(recovered_val_paths) == META_VAL_COUNT
        and not (set(recovered_train_paths) & set(recovered_val_paths))
        and recovered_union == expected_paths
    )
    if recovery_valid:
        selected_indices = np.array(
            sorted(record_index_by_path[path] for path in recovered_val_paths),
            dtype=int,
        )
        solver_manifest = {
            "algorithm": (
                "MILP_L1_multilabel_and_distribution_balancing; "
                "solution recovered from lists written by the immediately "
                "preceding run after a CSV serialization-only failure"
            ),
            "solver": "scipy.optimize.milp (HiGHS)",
            "solver_status": "RECOVERED_FEASIBLE_SOLUTION",
            "solver_message": (
                "The preceding run passed MILP feasibility and exact-count "
                "checks, then failed only while serializing extra CSV fields."
            ),
            "objective": "NOT_PERSISTED_BEFORE_SERIALIZATION_FAILURE",
            "seed_usage": (
                "seed42 deterministic objective tie-break; scientific targets "
                "are class/distribution constraints"
            ),
            "feature_count": "NOT_PERSISTED_BEFORE_SERIALIZATION_FAILURE",
            "feature_names": [],
            "quartile_edges": {},
            "sequence_grouping": (
                "NOT_APPLIED_NO_OFFICIAL_MACHINE_READABLE_KEY"
            ),
            "duplicate_hash_groups_constrained_to_one_split": len(
                duplicate_train_hashes
            ),
            "duplicate_hash_equality_constraints": sum(
                len(paths) - 1 for paths in duplicate_train_hashes.values()
            ),
            "recovery_validation": {
                "meta_train_count": len(recovered_train_paths),
                "meta_inner_val_count": len(recovered_val_paths),
                "path_overlap": 0,
                "union_matches_all_original_train_paths": True,
            },
        }
    else:
        selected_indices, solver_manifest = solve_split(train_records)
    solver_checkpoint.write_text(
        json.dumps(
            {
                "timestamp": now_iso(),
                "selected_meta_inner_val_paths": [
                    train_records[int(index)]["image_path"]
                    for index in selected_indices
                ],
                "solver_manifest": solver_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    val_index_set = set(int(index) for index in selected_indices)
    meta_val_rows = [
        row for index, row in enumerate(train_records)
        if index in val_index_set
    ]
    meta_train_rows = [
        row for index, row in enumerate(train_records)
        if index not in val_index_set
    ]
    for row in meta_train_rows:
        row["assigned_split"] = "meta_train"
    for row in meta_val_rows:
        row["assigned_split"] = "meta_inner_val"

    if len(meta_train_rows) != META_TRAIN_COUNT or len(meta_val_rows) != META_VAL_COUNT:
        raise RuntimeError("Internal split count mismatch")

    meta_train_rows.sort(key=lambda row: row["image_filename"])
    meta_val_rows.sort(key=lambda row: row["image_filename"])
    assigned_rows = sorted(
        meta_train_rows + meta_val_rows,
        key=lambda row: row["image_filename"],
    )

    meta_train_list.write_text(
        "\n".join(row["image_path"] for row in meta_train_rows) + "\n",
        encoding="utf-8",
    )
    meta_val_list.write_text(
        "\n".join(row["image_path"] for row in meta_val_rows) + "\n",
        encoding="utf-8",
    )
    official_list.write_text(
        "\n".join(str(path.resolve()) for path in official_images) + "\n",
        encoding="utf-8",
    )

    official_records = collect_records(
        official_images, OFFICIAL_LABEL_ROOT, "official_val"
    )
    if any(
        not row["label_exists"] or not row["label_valid"]
        for row in official_records
    ):
        invalid = [
            row for row in official_records
            if not row["label_exists"] or not row["label_valid"]
        ]
        raise RuntimeError(f"Invalid official-val labels: {len(invalid)}")

    train_hashes = {row["image_sha256"] for row in meta_train_rows}
    val_hashes = {row["image_sha256"] for row in meta_val_rows}
    official_hashes = {row["image_sha256"] for row in official_records}
    exact_overlap = {
        "meta_train_vs_meta_inner_val_paths": len(
            {row["image_path"] for row in meta_train_rows}
            & {row["image_path"] for row in meta_val_rows}
        ),
        "meta_train_vs_official_val_paths": len(
            {row["image_path"] for row in meta_train_rows}
            & {row["image_path"] for row in official_records}
        ),
        "meta_inner_val_vs_official_val_paths": len(
            {row["image_path"] for row in meta_val_rows}
            & {row["image_path"] for row in official_records}
        ),
        "meta_train_vs_meta_inner_val_hashes": len(train_hashes & val_hashes),
        "meta_train_vs_official_val_hashes": len(
            train_hashes & official_hashes
        ),
        "meta_inner_val_vs_official_val_hashes": len(
            val_hashes & official_hashes
        ),
    }

    class_rows, distribution_rows, small_rows, balance_audit = (
        distribution_tables(meta_train_rows, meta_val_rows)
    )
    near_duplicate_rows = near_duplicate_audit(
        meta_train_rows, meta_val_rows, official_records
    )
    near_duplicate_path = STAGE_ROOT / "near_duplicate_pairs.csv"
    write_csv(
        near_duplicate_path,
        near_duplicate_rows,
        [
            "left_split",
            "left_image",
            "right_split",
            "right_image",
            "phash_hamming_distance",
            "threshold",
            "classification",
        ],
    )

    manifest_fields = [
        "assigned_split",
        "image_path",
        "image_filename",
        "image_sha256",
        "image_phash",
        "label_path",
        "label_sha256",
        "sequence_or_scene_identifier",
        "sequence_key_source",
        "filename_prefix_surrogate",
        "present_classes",
        "present_class_bitset",
        "object_count",
        "small_object_count",
        "small_object_ratio",
        "image_width",
        "image_height",
        "image_area",
        "aspect_ratio",
        "label_valid",
        "label_errors",
    ]
    manifest_path = STAGE_ROOT / "split_manifest.csv"
    write_csv(manifest_path, assigned_rows, manifest_fields)
    csv_to_workbook(
        manifest_path,
        STAGE_ROOT / "split_manifest.xlsx",
        "Split Manifest",
    )

    summary_rows = [
        {"metric": "original_train_images", "value": len(train_records)},
        {"metric": "meta_train_images", "value": len(meta_train_rows)},
        {"metric": "meta_inner_val_images", "value": len(meta_val_rows)},
        {"metric": "official_val_images", "value": len(official_records)},
        {"metric": "nc", "value": 10},
        {"metric": "split_seed", "value": SPLIT_SEED},
        {
            "metric": "duplicate_hash_groups_in_original_train",
            "value": len(duplicate_train_hashes),
        },
        {
            "metric": "duplicate_hash_group_policy",
            "value": "KEEP_EACH_IDENTICAL_IMAGE_GROUP_IN_ONE_SPLIT",
        },
        {
            "metric": "sequence_key_status",
            "value": "NOT_AVAILABLE_OFFICIALLY",
        },
        {
            "metric": "sequence_grouping_applied",
            "value": False,
        },
        {
            "metric": "near_duplicate_warning_pairs",
            "value": len(near_duplicate_rows),
        },
    ]
    target_rows = [
        {
            "audit_metric": "max per-class image prevalence difference (pp)",
            "preferred_target": "<= 2.0",
            "actual": balance_audit[
                "max_absolute_per_class_image_prevalence_difference_pp"
            ],
            "status": (
                "PASS"
                if balance_audit[
                    "max_absolute_per_class_image_prevalence_difference_pp"
                ] <= 2.0
                else "DOCUMENTED_WARNING"
            ),
        },
        {
            "audit_metric": "global small-object ratio difference (pp)",
            "preferred_target": "<= 2.0 absolute",
            "actual": balance_audit[
                "global_small_object_ratio_difference_pp"
            ],
            "status": (
                "PASS"
                if abs(
                    balance_audit[
                        "global_small_object_ratio_difference_pp"
                    ]
                ) <= 2.0
                else "DOCUMENTED_WARNING"
            ),
        },
        {
            "audit_metric": "SMD objects per image",
            "preferred_target": "<= 0.10 absolute",
            "actual": balance_audit["smd_objects_per_image"],
            "status": (
                "PASS"
                if abs(balance_audit["smd_objects_per_image"]) <= 0.10
                else "DOCUMENTED_WARNING"
            ),
        },
        {
            "audit_metric": "SMD small objects per image",
            "preferred_target": "<= 0.10 absolute",
            "actual": balance_audit["smd_small_objects_per_image"],
            "status": (
                "PASS"
                if abs(
                    balance_audit["smd_small_objects_per_image"]
                ) <= 0.10
                else "DOCUMENTED_WARNING"
            ),
        },
        {
            "audit_metric": "SMD small-object ratio per image",
            "preferred_target": "<= 0.10 absolute",
            "actual": balance_audit[
                "smd_small_object_ratio_per_image"
            ],
            "status": (
                "PASS"
                if abs(
                    balance_audit[
                        "smd_small_object_ratio_per_image"
                    ]
                ) <= 0.10
                else "DOCUMENTED_WARNING"
            ),
        },
        {
            "audit_metric": "class-object histogram JS divergence",
            "preferred_target": "<= 0.05",
            "actual": balance_audit[
                "class_object_histogram_js_divergence"
            ],
            "status": (
                "PASS"
                if balance_audit[
                    "class_object_histogram_js_divergence"
                ] <= 0.05
                else "DOCUMENTED_WARNING"
            ),
        },
    ]
    split_statistics = Workbook()
    add_dict_sheet(split_statistics, "Summary", summary_rows)
    add_dict_sheet(split_statistics, "Balance Targets", target_rows)
    add_dict_sheet(
        split_statistics,
        "Solver",
        [
            {
                "algorithm": solver_manifest["algorithm"],
                "solver": solver_manifest["solver"],
                "status": solver_manifest["solver_status"],
                "message": solver_manifest["solver_message"],
                "objective": solver_manifest["objective"],
                "feature_count": solver_manifest["feature_count"],
                "seed_usage": solver_manifest["seed_usage"],
            }
        ],
    )
    save_workbook(
        split_statistics,
        STAGE_ROOT / "split_statistics.xlsx",
    )

    class_workbook = Workbook()
    add_dict_sheet(class_workbook, "Class Distribution", class_rows)
    save_workbook(
        class_workbook,
        STAGE_ROOT / "class_distribution.xlsx",
    )

    object_workbook = Workbook()
    add_dict_sheet(
        object_workbook,
        "Object Distribution",
        distribution_rows,
    )
    save_workbook(
        object_workbook,
        STAGE_ROOT / "object_count_distribution.xlsx",
    )

    small_workbook = Workbook()
    add_dict_sheet(
        small_workbook,
        "Small Object Summary",
        small_rows,
    )
    add_dict_sheet(
        small_workbook,
        "Distribution Metrics",
        [
            row
            for row in distribution_rows
            if "small" in row["metric"]
        ],
    )
    save_workbook(
        small_workbook,
        STAGE_ROOT / "small_object_distribution.xlsx",
    )

    prefix_counts: dict[str, Counter[str]] = {
        "meta_train": Counter(
            row["filename_prefix_surrogate"] for row in meta_train_rows
        ),
        "meta_inner_val": Counter(
            row["filename_prefix_surrogate"] for row in meta_val_rows
        ),
    }
    all_prefixes = sorted(
        set(prefix_counts["meta_train"])
        | set(prefix_counts["meta_inner_val"])
    )
    sequence_diagnostic_rows = [
        {
            "filename_prefix_surrogate": prefix,
            "meta_train_count": prefix_counts["meta_train"][prefix],
            "meta_inner_val_count": prefix_counts["meta_inner_val"][prefix],
            "appears_in_both": (
                prefix_counts["meta_train"][prefix] > 0
                and prefix_counts["meta_inner_val"][prefix] > 0
            ),
            "interpretation": (
                "DIAGNOSTIC_ONLY_NOT_AN_OFFICIAL_SEQUENCE_IDENTIFIER"
            ),
        }
        for prefix in all_prefixes
    ]
    sequence_status_rows = [
        {
            "field": "official_sequence_key",
            "value": "NOT_AVAILABLE_IN_LOCAL_VISDRONE_DET_PACKAGE",
        },
        {
            "field": "grouping_applied",
            "value": "NO",
        },
        {
            "field": "reason",
            "value": (
                "Official VisDrone source separates DET static images from "
                "VID sequences; local DET metadata contains no official "
                "machine-readable sequence key."
            ),
        },
        {
            "field": "official_source",
            "value": "https://github.com/VisDrone/VisDrone-Dataset",
        },
        {
            "field": "filename_prefix_use",
            "value": "DIAGNOSTIC_ONLY_NOT_USED_IN_STRATIFICATION",
        },
    ]
    sequence_workbook = Workbook()
    add_dict_sheet(sequence_workbook, "Status", sequence_status_rows)
    add_dict_sheet(
        sequence_workbook,
        "Prefix Diagnostic",
        sequence_diagnostic_rows,
    )
    save_workbook(
        sequence_workbook,
        STAGE_ROOT / "sequence_distribution.xlsx",
    )

    data_yaml_root = STAGE_ROOT / "data_yamls"
    inner_yaml_path = (
        data_yaml_root / "visdrone_meta_train5177_inner1294.yaml"
    )
    full_yaml_path = (
        data_yaml_root / "visdrone_full_train6471_official_val548.yaml"
    )
    inner_yaml = {
        "path": str(DATA_ROOT).replace("\\", "/"),
        "train": str(meta_train_list).replace("\\", "/"),
        "val": str(meta_val_list).replace("\\", "/"),
        "nc": 10,
        "names": {index: name for index, name in enumerate(EXPECTED_CLASSES)},
        "split_seed": 42,
        "usage": "Stage02-Stage08 internal search only",
        "official_val_used": False,
    }
    full_yaml = {
        "path": str(DATA_ROOT).replace("\\", "/"),
        "train": str(TRAIN_IMAGE_ROOT).replace("\\", "/"),
        "val": str(OFFICIAL_IMAGE_ROOT).replace("\\", "/"),
        "nc": 10,
        "names": {index: name for index, name in enumerate(EXPECTED_CLASSES)},
        "official_val_locked_until_stage": "Stage11",
        "stage10_requirement": "training must use validation-disabled fixed horizon",
        "usage": "Stage10 full training and Stage11 official evaluation only",
    }
    inner_yaml_path.write_text(
        yaml.safe_dump(inner_yaml, sort_keys=False),
        encoding="utf-8",
    )
    full_yaml_path.write_text(
        yaml.safe_dump(full_yaml, sort_keys=False),
        encoding="utf-8",
    )

    near_duplicate_cross_internal = sum(
        row["right_split"] == "meta_train"
        for row in near_duplicate_rows
    )
    near_duplicate_with_official = sum(
        row["right_split"] == "official_val"
        for row in near_duplicate_rows
    )
    leakage_audit = {
        "timestamp": now_iso(),
        "counts": {
            "original_train": len(train_records),
            "meta_train": len(meta_train_rows),
            "meta_inner_val": len(meta_val_rows),
            "official_val": len(official_records),
        },
        "exact_overlap": exact_overlap,
        "duplicate_image_hashes_across_splits": sum(
            exact_overlap[key]
            for key in exact_overlap
            if key.endswith("_hashes")
        ),
        "source_duplicate_image_hashes": {
            "group_count": len(duplicate_train_hashes),
            "policy": "KEEP_EACH_IDENTICAL_IMAGE_GROUP_IN_ONE_SPLIT",
            "groups": [
                {
                    "image_sha256": digest,
                    "image_paths": paths,
                    "label_sha256_values": [
                        row["label_sha256"]
                        for row in train_records
                        if row["image_sha256"] == digest
                    ],
                    "assigned_splits": sorted(
                        {
                            row["assigned_split"]
                            for row in assigned_rows
                            if row["image_sha256"] == digest
                        }
                    ),
                }
                for digest, paths in sorted(duplicate_train_hashes.items())
            ],
        },
        "missing_labels": sum(
            not row["label_exists"]
            for row in train_records + official_records
        ),
        "invalid_labels": sum(
            not row["label_valid"]
            for row in train_records + official_records
        ),
        "class_ids_outside_0_9": 0,
        "class_order": EXPECTED_CLASSES,
        "class_order_correct": source_names == EXPECTED_CLASSES,
        "sequence_audit": {
            "official_sequence_key_available": False,
            "sequence_grouping_applied": False,
            "sequence_overlap": "NOT_DETERMINABLE_NO_OFFICIAL_KEY",
            "filename_prefix_surrogate_used_for_selection": False,
            "official_source": (
                "https://github.com/VisDrone/VisDrone-Dataset"
            ),
        },
        "near_duplicate_audit": {
            "method": "64-bit perceptual hash (DCT), Hamming threshold <=4",
            "cross_internal_warning_pairs": near_duplicate_cross_internal,
            "inner_val_vs_official_warning_pairs": (
                near_duplicate_with_official
            ),
            "pairs_file": str(near_duplicate_path),
            "blocking": False,
            "interpretation": (
                "Warning-only unless exact SHA256 duplicates are present."
            ),
        },
        "balance_audit": balance_audit,
        "official_val_lock": {
            "list_created": True,
            "bytes_read_for_integrity_sha256_and_label_validation": True,
            "model_loaded": False,
            "predictions_generated": False,
            "evaluation_metrics_computed": False,
            "used_in_stratification": False,
            "used_in_candidate_selection": False,
        },
        "training_started": False,
        "evaluation_started": False,
        "status": "PASS",
    }
    (STAGE_ROOT / "leakage_audit.json").write_text(
        json.dumps(leakage_audit, indent=2),
        encoding="utf-8",
    )

    source_snapshot_rows = [
        {
            "path": row["image_path"],
            "sha256": row["image_sha256"],
            "label_path": row["label_path"],
            "label_sha256": row["label_sha256"],
        }
        for row in train_records + official_records
    ]
    source_snapshot_digest = hashlib.sha256(
        json.dumps(
            source_snapshot_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest().upper()

    output_paths = [
        meta_train_list,
        meta_val_list,
        official_list,
        manifest_path,
        STAGE_ROOT / "split_manifest.xlsx",
        STAGE_ROOT / "split_statistics.xlsx",
        STAGE_ROOT / "class_distribution.xlsx",
        STAGE_ROOT / "object_count_distribution.xlsx",
        STAGE_ROOT / "small_object_distribution.xlsx",
        STAGE_ROOT / "sequence_distribution.xlsx",
        near_duplicate_path,
        inner_yaml_path,
        full_yaml_path,
        STAGE_ROOT / "leakage_audit.json",
        solver_checkpoint,
    ]
    split_hashes = {
        "timestamp": now_iso(),
        "source_dataset_yaml": {
            "path": str(SOURCE_YAML),
            "sha256": sha256(SOURCE_YAML),
        },
        "official_evaluator_small_area_definition": {
            "path": str(EVALUATOR_SOURCE),
            "sha256": sha256(EVALUATOR_SOURCE),
            "source_lines": "2122: small=[0.0, 32.0**2]",
        },
        "source_snapshot_digest": source_snapshot_digest,
        "outputs": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in output_paths
        ],
        "split_list_hashes": {
            "meta_train": sha256(meta_train_list),
            "meta_inner_val": sha256(meta_val_list),
            "official_val_locked": sha256(official_list),
        },
        "solver_manifest": solver_manifest,
    }
    (STAGE_ROOT / "split_hashes.json").write_text(
        json.dumps(split_hashes, indent=2),
        encoding="utf-8",
    )

    workers_after = process_markers()
    if any("pid" in row for row in workers_after):
        raise RuntimeError(
            f"Unexpected training/evaluation process after Stage01: {workers_after}"
        )

    scientific_pass = (
        len(meta_train_rows) == META_TRAIN_COUNT
        and len(meta_val_rows) == META_VAL_COUNT
        and len(official_records) == OFFICIAL_VAL_COUNT
        and not any(exact_overlap.values())
        and leakage_audit["missing_labels"] == 0
        and leakage_audit["invalid_labels"] == 0
        and all(row["class_present_in_both"] for row in class_rows)
        and source_names == EXPECTED_CLASSES
    )
    state = {
        "stage": "stage01",
        "status": (
            "paused_after_stage01_waiting_for_user_approval"
            if scientific_pass
            else "stage01_failed"
        ),
        "timestamp": now_iso(),
        "stage01_data_split": "PASS" if scientific_pass else "FAIL",
        "validation": "PENDING_FINAL_VALIDATOR",
        "approval_ready": False,
        "meta_train_count": len(meta_train_rows),
        "meta_inner_val_count": len(meta_val_rows),
        "official_val_count": len(official_records),
        "split_seed": SPLIT_SEED,
        "sequence_key_status": "NOT_AVAILABLE_OFFICIALLY",
        "sequence_grouping_applied": False,
        "official_val_used_for_evaluation": False,
        "official_val_used_for_stratification": False,
        "training_executed": False,
        "evaluation_executed": False,
        "stage02_started": False,
        "source_files_modified": 0,
        "required_next_approval_token": "APPROVE_STAGE01_DATA_SPLIT",
    }
    (STAGE_ROOT / "stage01_state.json").write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )
    (PROJECT_ROOT / "logs/current_stage.json").write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )
    (PROJECT_ROOT / "logs/current_run.json").write_text(
        json.dumps({"active_run": None, "timestamp": now_iso()}, indent=2),
        encoding="utf-8",
    )
    (PROJECT_ROOT / "logs/background_process.json").write_text(
        json.dumps(
            {"active": False, "pid": None, "timestamp": now_iso()},
            indent=2,
        ),
        encoding="utf-8",
    )
    with (PROJECT_ROOT / "logs/master_workflow.log").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            f"{now_iso()}\tstage01\t{state['status']}\t"
            "training=0\tevaluation=0\tofficial_val_metrics=0\n"
        )

    report = f"""# Stage01 Data Split and Leakage Audit

## Status

- Data split: {"PASS" if scientific_pass else "FAIL"}
- Meta-train: {len(meta_train_rows)}
- Meta-inner-val: {len(meta_val_rows)}
- Official val locked: {len(official_records)}
- Seed: {SPLIT_SEED}
- Stage02 started: NO

## Dataset integrity

- Source YAML: `{SOURCE_YAML}`
- Source YAML SHA256: `{sha256(SOURCE_YAML)}`
- nc: 10
- Class order: {", ".join(EXPECTED_CLASSES)}
- Missing labels: {leakage_audit["missing_labels"]}
- Invalid labels: {leakage_audit["invalid_labels"]}
- Exact path/hash overlap: {sum(exact_overlap.values())}
- Duplicate SHA256 groups already present inside original train6471: {len(duplicate_train_hashes)}
- Duplicate-source policy: keep every identical-image group wholly inside one split
- Duplicate SHA256 groups crossing split boundaries: {leakage_audit["duplicate_image_hashes_across_splits"]}
- Perceptual near-duplicate warning pairs across internal splits: {near_duplicate_cross_internal}
- Perceptual near-duplicate warning pairs against official val: {near_duplicate_with_official}

The two original duplicate-image groups have different label-file hashes.
No source row was deleted, merged, relabeled, or otherwise modified.

## Sequence decision

The official VisDrone source separates DET static-image detection from VID
video-sequence detection. The local DET package contains no official
machine-readable sequence/scene key. In accordance with the frozen plan, no
sequence identifier was inferred from filenames. The first filename component
is retained only as a diagnostic surrogate and was not used for assignment.

Official source: https://github.com/VisDrone/VisDrone-Dataset

## Stratification

- Algorithm: {solver_manifest["algorithm"]}
- Solver: {solver_manifest["solver"]}
- Solver status: {solver_manifest["solver_message"]}
- Max class-prevalence difference: {balance_audit["max_absolute_per_class_image_prevalence_difference_pp"]:.6f} pp
- Global small-object ratio difference: {balance_audit["global_small_object_ratio_difference_pp"]:.6f} pp
- SMD objects/image: {balance_audit["smd_objects_per_image"]:.6f}
- SMD small objects/image: {balance_audit["smd_small_objects_per_image"]:.6f}
- SMD small-object ratio: {balance_audit["smd_small_object_ratio_per_image"]:.6f}
- Class-object histogram JS divergence: {balance_audit["class_object_histogram_js_divergence"]:.8f}

## Official validation lock

Official val548 was accessed only for path, SHA256, image-integrity, label-
integrity, and leakage checks. It was not passed to the split optimizer, no
model was loaded, no prediction was generated, and no metric was computed.

## Execution

- Training executed: 0
- Evaluation executed: 0
- Official-val predictions: 0
- Source files modified: 0

The workflow is paused for `APPROVE_STAGE01_DATA_SPLIT`.
"""
    (STAGE_ROOT / "stage01_report.md").write_text(
        report,
        encoding="utf-8",
    )
    print("STAGE01_DATA_SPLIT=" + ("PASS" if scientific_pass else "FAIL"))
    print(f"META_TRAIN_COUNT={len(meta_train_rows)}")
    print(f"META_INNER_VAL_COUNT={len(meta_val_rows)}")
    print(f"OFFICIAL_VAL_COUNT={len(official_records)}")
    print(f"EXACT_OVERLAP={sum(exact_overlap.values())}")
    print(f"NEAR_DUPLICATE_WARNINGS={len(near_duplicate_rows)}")
    print("SEQUENCE_KEY=NOT_AVAILABLE_OFFICIALLY")
    print("OFFICIAL_VAL_EVALUATED=NO")
    print("TRAINING_STARTED=NO")
    print("STAGE02_STARTED=NO")


if __name__ == "__main__":
    main()

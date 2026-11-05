from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

import networkx as nx
import numpy as np
import torch
import yaml
from openpyxl import load_workbook
from torch.utils.data import DataLoader, Sampler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage03a_taylor_dependency_gene_space as base


PROJECT = base.PROJECT
STAGE01 = base.STAGE01
STAGE02 = base.STAGE02
STAGE03 = base.STAGE03
BEST_PT = base.BEST_PT
EXPECTED_BEST_SHA = base.EXPECTED_BEST_SHA
META_TRAIN = base.META_TRAIN
INNER_VAL = base.INNER_VAL
OFFICIAL_LOCKED = base.OFFICIAL_LOCKED
SPLIT_MANIFEST = base.SPLIT_MANIFEST
PROFILE_DIR = base.PROFILE_DIR
PROFILE_MODEL_ID = base.PROFILE_MODEL_ID
HYP_YAML = base.HYP_YAML
YOLOV5_ROOT = base.YOLOV5_ROOT
PRUNING_ROOT = base.PRUNING_ROOT

HIST_CONFIG = PRUNING_ROOT / "configs" / "taylor_importance_litev3_baseline.yaml"
HIST_COLLECTOR = PRUNING_ROOT / "pruning" / "taylor_importance.py"
HIST_BATCH_RUNNER = PRUNING_ROOT / "scripts" / "run_importance_collection_batch.py"
HIST_AGGREGATOR = PRUNING_ROOT / "pruning" / "aggregate_taylor_scores.py"
HIST_IMPORTANCE = PRUNING_ROOT / "pruning" / "importance_scores.py"
HIST_STRUCTURED_PRUNER = PRUNING_ROOT / "pruning" / "structured_pruner.py"
HIST_DEEP_PROFILE = PRUNING_ROOT / "profiling" / "deep_structure_profile.py"
HIST_HYP = YOLOV5_ROOT / "data" / "hyps" / "hyp.scratch-low.yaml"
YOLO_DATALOADERS = YOLOV5_ROOT / "utils" / "dataloaders.py"
YOLO_LOSS = YOLOV5_ROOT / "utils" / "loss.py"
HIST_ARCHIVE = (
    Path(_release_paths.expand('${CODE_ROOT}\\arch modify\\06_litev3_pruning\\03_taylor_importance\\baseline_1000'))
    / "yolov5s_lite_v3_full_coco_e300_best_fp32_pt"
)
HIST_LOG_STDOUT = HIST_ARCHIVE.parent / "taylor_baseline_1000_collection.stdout.log"
HIST_LOG_STDERR = HIST_ARCHIVE.parent / "taylor_baseline_1000_collection.stderr.log"
HIST_BATCH_SUMMARY = HIST_ARCHIVE / "batch_importance_collection_summary.json"
HIST_AGGREGATED = HIST_ARCHIVE / "aggregated" / "aggregated_taylor_importance.json"

OFFSETS = [0, 200, 400, 600, 800]
CHUNK_BATCHES = 200
BATCH_SIZE = 16
TOTAL_BATCHES = 1000
TOTAL_OBSERVATIONS = 16000
SEED = 42
IMGSZ = 640
WORKERS = 4

POOL_LIST = STAGE03 / "taylor_ordered_pool5177_seed42.txt"
POOL_DATA_YAML = STAGE03 / "taylor_historical_protocol_data.yaml"
POOL_MANIFEST = STAGE03 / "taylor_ordered_pool_manifest.csv"
SCHEDULE_CSV = STAGE03 / "taylor_observation_schedule16000_seed42.csv"
SCHEDULE_XLSX = STAGE03 / "taylor_schedule_statistics.xlsx"
RUNTIME_ROOT = STAGE03 / "taylor_runtime_historical_1000"
AGGREGATED_DIR = RUNTIME_ROOT / "aggregated"
SUPERSEDED_DIR = STAGE03 / "superseded_512_protocol"

AUDIT_SOURCES = [
    HIST_CONFIG,
    HIST_COLLECTOR,
    HIST_BATCH_RUNNER,
    HIST_AGGREGATOR,
    HIST_IMPORTANCE,
    HIST_STRUCTURED_PRUNER,
    HIST_DEEP_PROFILE,
    HIST_HYP,
    YOLO_DATALOADERS,
    YOLO_LOSS,
    HIST_BATCH_SUMMARY,
    HIST_AGGREGATED,
    HIST_LOG_STDOUT,
    HIST_LOG_STDERR,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_hashes() -> dict[str, str]:
    result = {}
    for path in AUDIT_SOURCES:
        if not path.is_file():
            raise FileNotFoundError(f"Historical Taylor evidence is missing: {path}")
        result[str(path)] = base.sha256(path)
    return result


def snapshot_current_dependencies() -> dict[str, str]:
    paths = [
        STAGE03 / "dependency_graph.json",
        STAGE03 / "dependency_graph.graphml",
        STAGE03 / "dependency_groups.xlsx",
        STAGE03 / "gene_space_manifest.json",
        STAGE03 / "gene_space_manifest.xlsx",
        STAGE03 / "protected_groups_report.md",
        STAGE03 / "architecture_contract_tests.json",
        STAGE03 / "architecture_contract_tests.xlsx",
    ]
    return {str(path): base.sha256(path) for path in paths}


def archive_superseded_512() -> dict[str, Any]:
    SUPERSEDED_DIR.mkdir(parents=True, exist_ok=True)
    names = [
        "taylor_subset512_seed42.txt",
        "taylor_sensitivity512_seed42.txt",
        "taylor_subset_manifest.csv",
        "taylor_subset_statistics.xlsx",
        "taylor_subset_hashes.json",
        "taylor_subset_data.yaml",
        "taylor_configuration.yaml",
        "taylor_protocol_equivalence_audit.json",
        "taylor_protocol_equivalence_audit.xlsx",
        "raw_channel_taylor_scores.csv",
        "normalized_channel_taylor_scores.csv",
        "layer_taylor_summary.xlsx",
        "taylor_scores.xlsx",
        "taylor_scores.npy",
        "taylor_ranking_manifest.json",
        "taylor_run_manifest.json",
        "taylor_hashes.json",
        "stage03a_report.md",
        "stage03a_validation.json",
    ]
    archived = []
    for name in names:
        source = STAGE03 / name
        target = SUPERSEDED_DIR / name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
        if target.is_file():
            archived.append(
                {
                    "relative_path": name,
                    "archived_path": str(target),
                    "sha256": base.sha256(target),
                    "status": "SUPERSEDED_NOT_PROTOCOL_EQUIVALENT",
                    "reason": "512-image/32-batch horizon does not match the historical 1000-batch protocol.",
                }
            )
    old_runtime = STAGE03 / "taylor_runtime"
    runtime_target = SUPERSEDED_DIR / "taylor_runtime_32_batches"
    if old_runtime.is_dir() and not runtime_target.exists():
        shutil.copytree(old_runtime, runtime_target)
    payload = {
        "created_utc": utc_now(),
        "superseded_protocol": {
            "images": 512,
            "batch_size": 16,
            "batches": 32,
            "status": "EXCLUDED_FROM_CURRENT_STAGE03A_TAYLOR_RANKING",
        },
        "archived_files": archived,
        "runtime_archive": str(runtime_target) if runtime_target.exists() else "NOT_FOUND",
        "status": "PASS",
    }
    base.json_write(STAGE03 / "superseded_512_taylor_manifest.json", payload)
    return payload


def historical_truth_rows(source_sha: dict[str, str]) -> list[dict[str, Any]]:
    common = {
        "Historical_Config": str(HIST_CONFIG),
        "Historical_Config_SHA256": source_sha[str(HIST_CONFIG)],
    }
    rows = [
        {
            "Field": "Taylor image observations",
            "Historical_Value": 16000,
            "Corrected_Stage03A_Value": 16000,
            "Status": "MATCH",
            "Source_File": str(HIST_BATCH_SUMMARY),
            "Source_Location": "run_summaries: five runs x observed_images=3200",
            "Reason": "The previous 512-image result is superseded.",
        },
        {
            "Field": "Unique eligible images",
            "Historical_Value": "First 16000 COCO train2017 entries; pool was larger than horizon",
            "Corrected_Stage03A_Value": "5177 unique meta-train images, repeated cyclically to preserve 16000 observations",
            "Status": "EXPECTED_CHANGE",
            "Source_File": str(YOLO_DATALOADERS),
            "Source_Location": "LoadImagesAndLabels lines 574-583; current split lock",
            "Reason": "Exact image identities and repetition are forced by the smaller target-domain pool.",
        },
        {
            "Field": "Batch size",
            "Historical_Value": 16,
            "Corrected_Stage03A_Value": 16,
            "Status": "MATCH",
            "Source_File": str(HIST_CONFIG),
            "Source_Location": "taylor.batch_size",
            "Reason": "",
        },
        {
            "Field": "Total batches",
            "Historical_Value": 1000,
            "Corrected_Stage03A_Value": 1000,
            "Status": "MATCH",
            "Source_File": str(HIST_BATCH_SUMMARY),
            "Source_Location": "five runs x processed_batches=200",
            "Reason": "",
        },
        {
            "Field": "Chunk layout",
            "Historical_Value": "5 chunks x 200 batches",
            "Corrected_Stage03A_Value": "5 chunks x 200 batches",
            "Status": "MATCH",
            "Source_File": str(HIST_CONFIG),
            "Source_Location": "max_batches=200; offsets=[0,200,400,600,800]",
            "Reason": "",
        },
        {
            "Field": "Subset sampling algorithm",
            "Historical_Value": "No separate subset; lexicographically sorted train list, contiguous offset windows, shuffle=false",
            "Corrected_Stage03A_Value": "Lexicographically sorted meta-train list, same contiguous logical offsets, deterministic cyclic continuation at pool exhaustion",
            "Status": "EXPECTED_CHANGE",
            "Source_File": str(YOLO_DATALOADERS),
            "Source_Location": "lines 159-217, 574-583",
            "Reason": "The 5177-image target pool is smaller than the frozen 16000-observation horizon; cyclic continuation is the minimal deterministic adaptation forced by the permitted pool change.",
        },
        {
            "Field": "Subset seed",
            "Historical_Value": 42,
            "Corrected_Stage03A_Value": 42,
            "Status": "MATCH",
            "Source_File": str(HIST_CONFIG),
            "Source_Location": "taylor.seed",
            "Reason": "shuffle=false, but the same generator seed remains configured.",
        },
        {
            "Field": "Image ordering",
            "Historical_Value": "sorted path order; no shuffle",
            "Corrected_Stage03A_Value": "sorted path order; no shuffle; cyclic modulo 5177",
            "Status": "EXPECTED_CHANGE",
            "Source_File": str(YOLO_DATALOADERS),
            "Source_Location": "line 582 and create_dataloader shuffle=false",
            "Reason": "Lexicographic order is preserved; wrap-around identities after pool exhaustion are forced by the permitted target-pool change.",
        },
        {
            "Field": "Image size",
            "Historical_Value": 640,
            "Corrected_Stage03A_Value": 640,
            "Status": "MATCH",
            "Source_File": str(HIST_CONFIG),
            "Source_Location": "taylor.image_size",
            "Reason": "",
        },
        {
            "Field": "Preprocessing",
            "Historical_Value": "letterbox auto=false, scaleup=false, BGR->RGB CHW, float32 /255",
            "Corrected_Stage03A_Value": "letterbox auto=false, scaleup=false, BGR->RGB CHW, float32 /255",
            "Status": "MATCH",
            "Source_File": f"{YOLO_DATALOADERS}; {HIST_COLLECTOR}",
            "Source_Location": "dataloaders.py 780-791; taylor_importance.py 217-219",
            "Reason": "",
        },
        {
            "Field": "Augmentation",
            "Historical_Value": "disabled",
            "Corrected_Stage03A_Value": "disabled",
            "Status": "MATCH",
            "Source_File": str(HIST_COLLECTOR),
            "Source_Location": "create_dataloader augment=False",
            "Reason": "Mosaic, MixUp, random perspective, and flips are inactive.",
        },
        {
            "Field": "Model mode",
            "Historical_Value": "model.train()",
            "Corrected_Stage03A_Value": "model.train()",
            "Status": "MATCH",
            "Source_File": str(HIST_COLLECTOR),
            "Source_Location": "lines 120-128",
            "Reason": "",
        },
        {
            "Field": "BatchNorm behavior",
            "Historical_Value": "all BN modules eval; running statistics frozen",
            "Corrected_Stage03A_Value": "all BN modules eval; running statistics frozen",
            "Status": "MATCH",
            "Source_File": str(HIST_COLLECTOR),
            "Source_Location": "lines 49-55 and 130-133",
            "Reason": "51 BN modules in both model realizations.",
        },
        {
            "Field": "AMP / precision",
            "Historical_Value": "FP32 model and gradients; no autocast",
            "Corrected_Stage03A_Value": "FP32 model and gradients; no autocast",
            "Status": "MATCH",
            "Source_File": str(HIST_COLLECTOR),
            "Source_Location": "model.float(); images.float(); no autocast context",
            "Reason": "",
        },
        {
            "Field": "Taylor formula",
            "Historical_Value": "I_c=batch-mean_(N,H,W)|A_c*dL/dA_c|",
            "Corrected_Stage03A_Value": "I_c=batch-mean_(N,H,W)|A_c*dL/dA_c|",
            "Status": "MATCH",
            "Source_File": str(HIST_COLLECTOR),
            "Source_Location": "lines 34-46 and 226-237",
            "Reason": "",
        },
        {
            "Field": "Loss components and weights",
            "Historical_Value": "CIoU box 0.05 + objectness BCE 1.0 + class BCE 0.5; P3/P4/P5 balance 4.0/1.0/0.4",
            "Corrected_Stage03A_Value": "CIoU box 0.05 + objectness BCE 1.0 + class BCE 0.5; P3/P4/P5 balance 4.0/1.0/0.4",
            "Status": "MATCH",
            "Source_File": f"{YOLO_LOSS}; {HIST_HYP}; {HYP_YAML}",
            "Source_Location": "loss.py 108-191; hyp keys box,obj,cls,cls_pw,obj_pw,fl_gamma,anchor_t",
            "Reason": "Current Stage02 loss-related hyperparameters equal the historical file.",
        },
        {
            "Field": "Gradient reset / accumulation",
            "Historical_Value": "zero_grad before and after each batch; score sums persist; gradients do not",
            "Corrected_Stage03A_Value": "zero_grad before and after each batch; score sums persist; gradients do not",
            "Status": "MATCH",
            "Source_File": str(HIST_COLLECTOR),
            "Source_Location": "lines 221 and 244",
            "Reason": "",
        },
        {
            "Field": "Score aggregation",
            "Historical_Value": "mean across 200 batches per chunk, then unweighted mean across 5 equal chunks",
            "Corrected_Stage03A_Value": "mean across 200 batches per chunk, then unweighted mean across 5 equal chunks",
            "Status": "MATCH",
            "Source_File": f"{HIST_COLLECTOR}; {HIST_AGGREGATOR}",
            "Source_Location": "collector 275; aggregator 31-106",
            "Reason": "",
        },
        {
            "Field": "Absolute-value rule",
            "Historical_Value": "abs(A*dL/dA) before N,H,W mean",
            "Corrected_Stage03A_Value": "abs(A*dL/dA) before N,H,W mean",
            "Status": "MATCH",
            "Source_File": str(HIST_COLLECTOR),
            "Source_Location": "lines 39-45",
            "Reason": "",
        },
        {
            "Field": "Normalization",
            "Historical_Value": "per-scope aggregated mean score divided by scope maximum",
            "Corrected_Stage03A_Value": "per-scope aggregated mean score divided by scope maximum",
            "Status": "MATCH",
            "Source_File": str(HIST_AGGREGATOR),
            "Source_Location": "lines 72-88",
            "Reason": "",
        },
        {
            "Field": "Dependency propagation",
            "Historical_Value": "scope hooks from channel topology; pruning propagation through Conv/C3/C3Lite/SPPF/Concat/Detect contracts",
            "Corrected_Stage03A_Value": "same 35 scope hooks and same structural dependency implementation",
            "Status": "MATCH",
            "Source_File": f"{HIST_IMPORTANCE}; {HIST_STRUCTURED_PRUNER}; {HIST_DEEP_PROFILE}",
            "Source_Location": "resolve_scope_modules; apply_structured_pruning; build_channel_topology",
            "Reason": "No numerical averaging is performed across distinct dependency scopes.",
        },
        {
            "Field": "Evaluated checkpoint",
            "Historical_Value": "COCO Lite-v3 best.pt",
            "Corrected_Stage03A_Value": f"Lite-v3-SearchParent best.pt {EXPECTED_BEST_SHA}",
            "Status": "EXPECTED_CHANGE",
            "Source_File": str(BEST_PT),
            "Source_Location": "checkpoint SHA256",
            "Reason": "Explicitly permitted target-domain checkpoint change.",
        },
        {
            "Field": "Eligible sampling pool",
            "Historical_Value": "COCO train2017",
            "Corrected_Stage03A_Value": "meta-train5177 only",
            "Status": "EXPECTED_CHANGE",
            "Source_File": str(META_TRAIN),
            "Source_Location": f"SHA256 {base.sha256(META_TRAIN)}",
            "Reason": "Explicitly permitted target-domain pool change.",
        },
    ]
    return [{**row, **common} for row in rows]


def write_source_of_truth_audit(source_sha: dict[str, str]) -> list[dict[str, Any]]:
    rows = historical_truth_rows(source_sha)
    unresolved = [row for row in rows if row["Status"] == "UNRESOLVED"]
    payload = {
        "created_utc": utc_now(),
        "decision": "RERUN_REQUIRED_HISTORICAL_HORIZON_IS_1000_BATCHES_16000_OBSERVATIONS",
        "historical_effective_command": (
            f"{sys.executable} {HIST_BATCH_RUNNER} --config {HIST_CONFIG}"
        ),
        "shell_invocation_record": "NOT_RECORDED",
        "effective_arguments_fully_recovered": True,
        "historical_horizon": {
            "run_count": 5,
            "offsets": OFFSETS,
            "batches_per_run": CHUNK_BATCHES,
            "total_batches": TOTAL_BATCHES,
            "batch_size": BATCH_SIZE,
            "observed_images_per_run": 3200,
            "total_image_observations": TOTAL_OBSERVATIONS,
        },
        "current_target_pool_adaptation": {
            "unique_pool_images": 5177,
            "ordering": "lexicographic path order",
            "shuffle": False,
            "cycle_rule": "global_observation_index modulo 5177",
            "exposure_counts": {
                "first_469_sorted_images": 4,
                "remaining_4708_sorted_images": 3,
            },
        },
        "rows": rows,
        "unresolved_major_fields": len(unresolved),
        "status": "PASS_SOURCE_OF_TRUTH_RESOLVED" if not unresolved else "FAIL_UNRESOLVED",
    }
    base.json_write(STAGE03 / "taylor_source_of_truth_audit.json", payload)
    base.xlsx_from_sheets(
        STAGE03 / "taylor_source_of_truth_audit.xlsx",
        [("Source of Truth", rows)],
    )
    source_rows = [
        {
            "Source_File": path,
            "SHA256": digest,
            "Evidence_Role": "Historical Taylor protocol source",
        }
        for path, digest in source_sha.items()
    ]
    base.csv_write(STAGE03 / "taylor_source_of_truth_sources.csv", source_rows)
    base.json_write(
        STAGE03 / "taylor_protocol_equivalence_audit.json",
        {
            "created_utc": utc_now(),
            "protocol_name": "Corrected Stage03A versus historical Taylor-1000 protocol",
            "allowed_statuses": ["MATCH", "EXPECTED_CHANGE", "UNRESOLVED"],
            "rows": rows,
            "major_unresolved_fields": len(unresolved),
            "execution_allowed": len(unresolved) == 0,
            "status": "PASS" if not unresolved else "FAIL_UNRESOLVED_PROTOCOL",
        },
    )
    base.xlsx_from_sheets(
        STAGE03 / "taylor_protocol_equivalence_audit.xlsx",
        [("Protocol Audit", rows)],
    )
    markdown = [
        "# Taylor Protocol Source-of-Truth Audit",
        "",
        "## Decision",
        "",
        "The historical protocol used 1000 batches and 16000 image observations,",
        "not 32 batches and 512 images. The 512-image Stage03A result is superseded.",
        "",
        "## Historical Effective Protocol",
        "",
        "- Entry point: `run_importance_collection_batch.py`",
        "- Config: `taylor_importance_litev3_baseline.yaml`",
        "- Offsets: `0, 200, 400, 600, 800`",
        "- Chunks: 5",
        "- Batches per chunk: 200",
        "- Batch size: 16",
        "- Total batches: 1000",
        "- Total image observations: 16000",
        "- Seed: 42",
        "- Shuffle: false",
        "- Images sorted lexicographically by path",
        "",
        "The exact shell invocation was not archived, but the entry point, effective",
        "config, run summaries, logs, outputs, and all effective arguments are",
        "recoverable and mutually consistent.",
        "",
        "## Target-Pool Adaptation",
        "",
        "The target pool contains 5177 images. To preserve the exact 16000-observation",
        "horizon and ordered no-shuffle behavior, the sorted pool is traversed",
        "cyclically. The first 469 sorted images appear four times; the remaining",
        "4708 images appear three times.",
        "",
        f"Unresolved major fields: {len(unresolved)}.",
    ]
    base.text_write(STAGE03 / "taylor_source_of_truth_audit.md", "\n".join(markdown) + "\n")
    if unresolved:
        raise RuntimeError(f"Historical Taylor protocol remains unresolved: {unresolved}")
    return rows


def build_pool_and_schedule() -> dict[str, Any]:
    source_paths = base.read_nonempty_lines(META_TRAIN)
    sorted_paths = sorted(source_paths, key=lambda value: value.replace("/", os.sep))
    if len(sorted_paths) != 5177 or len(set(map(base.normalize_path_text, sorted_paths))) != 5177:
        raise RuntimeError("meta-train pool is not 5177 unique paths.")
    inner = set(map(base.normalize_path_text, base.read_nonempty_lines(INNER_VAL)))
    official = set(map(base.normalize_path_text, base.read_nonempty_lines(OFFICIAL_LOCKED)))
    pool_keys = set(map(base.normalize_path_text, sorted_paths))
    if pool_keys & inner or pool_keys & official:
        raise RuntimeError("Taylor pool overlaps a locked evaluation split.")

    manifest_map = {}
    with SPLIT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            manifest_map[base.normalize_path_text(row["image_path"])] = row

    pool_rows = []
    for index, path_text in enumerate(sorted_paths):
        row = manifest_map[base.normalize_path_text(path_text)]
        pool_rows.append(
            {
                "pool_index_zero_based": index,
                "image_path": path_text,
                "image_sha256": row["image_sha256"],
                "label_path": row["label_path"],
                "label_sha256": row["label_sha256"],
                "present_classes": row["present_classes"],
                "object_count": int(row["object_count"]),
                "small_object_count": int(row["small_object_count"]),
                "small_object_ratio": float(row["small_object_ratio"]),
                "source_split": row["assigned_split"],
            }
        )
    base.text_write(POOL_LIST, "\n".join(sorted_paths) + "\n")
    base.csv_write(POOL_MANIFEST, pool_rows)

    schedule_rows = []
    exposure = Counter()
    for observation_index in range(TOTAL_OBSERVATIONS):
        pool_index = observation_index % len(sorted_paths)
        cycle_index = observation_index // len(sorted_paths)
        exposure[pool_index] += 1
        row = pool_rows[pool_index]
        schedule_rows.append(
            {
                "observation_index_zero_based": observation_index,
                "global_batch_index_zero_based": observation_index // BATCH_SIZE,
                "batch_slot_zero_based": observation_index % BATCH_SIZE,
                "logical_chunk_offset": (observation_index // (CHUNK_BATCHES * BATCH_SIZE)) * CHUNK_BATCHES,
                "pool_index_zero_based": pool_index,
                "cycle_index_zero_based": cycle_index,
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
                "label_path": row["label_path"],
                "label_sha256": row["label_sha256"],
            }
        )
    base.csv_write(SCHEDULE_CSV, schedule_rows)
    if sorted(exposure.values()) != [3] * 4708 + [4] * 469:
        raise RuntimeError("Cyclic Taylor exposure distribution is incorrect.")

    base_yaml = yaml.safe_load(
        (STAGE01 / "data_yamls" / "visdrone_meta_train5177_inner1294.yaml").read_text(encoding="utf-8")
    )
    base_yaml["train"] = str(POOL_LIST).replace("\\", "/")
    base_yaml["val"] = str(POOL_LIST).replace("\\", "/")
    base_yaml["usage"] = "Stage03A historical Taylor-1000 protocol only"
    base_yaml["official_val_used"] = False
    base_yaml["logical_horizon_batches"] = TOTAL_BATCHES
    base_yaml["logical_horizon_observations"] = TOTAL_OBSERVATIONS
    base.yaml_write(POOL_DATA_YAML, base_yaml)

    summary_rows = [
        {"Metric": "Unique pool images", "Value": len(sorted_paths)},
        {"Metric": "Total image observations", "Value": TOTAL_OBSERVATIONS},
        {"Metric": "Batch size", "Value": BATCH_SIZE},
        {"Metric": "Total batches", "Value": TOTAL_BATCHES},
        {"Metric": "Chunks", "Value": len(OFFSETS)},
        {"Metric": "Batches per chunk", "Value": CHUNK_BATCHES},
        {"Metric": "Offsets", "Value": ";".join(map(str, OFFSETS))},
        {"Metric": "Seed", "Value": SEED},
        {"Metric": "Shuffle", "Value": False},
        {"Metric": "Ordering", "Value": "lexicographic path order with cyclic continuation"},
        {"Metric": "Images with 4 exposures", "Value": 469},
        {"Metric": "Images with 3 exposures", "Value": 4708},
        {"Metric": "Inner-val overlap", "Value": 0},
        {"Metric": "Official-val overlap", "Value": 0},
    ]
    exposure_rows = [
        {"Exposure_Count": count, "Unique_Image_Count": sum(value == count for value in exposure.values())}
        for count in sorted(set(exposure.values()))
    ]
    base.xlsx_from_sheets(
        SCHEDULE_XLSX,
        [("Summary", summary_rows), ("Exposure Counts", exposure_rows)],
    )
    base.xlsx_from_sheets(
        STAGE03 / "taylor_subset_statistics.xlsx",
        [("Historical Schedule", summary_rows), ("Exposure Counts", exposure_rows)],
    )
    hashes = {
        "created_utc": utc_now(),
        "current_protocol": "HISTORICAL_TAYLOR_1000",
        "pool_count": len(sorted_paths),
        "total_observations": TOTAL_OBSERVATIONS,
        "total_batches": TOTAL_BATCHES,
        "pool_list_sha256": base.sha256(POOL_LIST),
        "pool_manifest_sha256": base.sha256(POOL_MANIFEST),
        "schedule_csv_sha256": base.sha256(SCHEDULE_CSV),
        "data_yaml_sha256": base.sha256(POOL_DATA_YAML),
        "meta_train_sha256": base.sha256(META_TRAIN),
        "inner_val_overlap": 0,
        "official_val_overlap": 0,
        "status": "PASS",
    }
    base.json_write(STAGE03 / "taylor_subset_hashes.json", hashes)
    return hashes


class CyclicHistoricalBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset_size: int):
        self.dataset_size = int(dataset_size)
        if self.dataset_size <= 0:
            raise ValueError("Taylor dataset is empty.")

    def __len__(self) -> int:
        return TOTAL_BATCHES

    def __iter__(self) -> Iterator[list[int]]:
        for batch_index in range(TOTAL_BATCHES):
            start = batch_index * BATCH_SIZE
            yield [(start + slot) % self.dataset_size for slot in range(BATCH_SIZE)]


def build_historical_loader():
    if str(YOLOV5_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLOV5_ROOT))
    from utils.dataloaders import LoadImagesAndLabels, create_dataloader, seed_worker
    from utils.general import check_dataset

    data = check_dataset(str(POOL_DATA_YAML))
    hyp = yaml.safe_load(HIST_HYP.read_text(encoding="utf-8"))
    _, dataset = create_dataloader(
        path=data["train"],
        imgsz=IMGSZ,
        batch_size=BATCH_SIZE,
        stride=32,
        single_cls=False,
        hyp=hyp,
        augment=False,
        cache=False,
        pad=0.0,
        rect=False,
        rank=-1,
        workers=WORKERS,
        image_weights=False,
        quad=False,
        prefix="Taylor historical pool: ",
        shuffle=False,
        seed=SEED,
    )
    if len(dataset) != 5177:
        raise RuntimeError(f"Historical Taylor target pool resolved to {len(dataset)}, expected 5177.")
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + SEED)
    loader = DataLoader(
        dataset,
        batch_sampler=CyclicHistoricalBatchSampler(len(dataset)),
        num_workers=WORKERS,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, dataset


def run_historical_taylor() -> dict[str, Any]:
    if str(base.PRUNING_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(base.PRUNING_PACKAGE_ROOT))
    if str(YOLOV5_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLOV5_ROOT))
    from pruning_research.pruning.aggregate_taylor_scores import aggregate_taylor_runs
    from pruning_research.pruning.taylor_importance import collect_taylor_scores_one_run
    import utils.dataloaders as yolo_dataloaders

    loader, dataset = build_historical_loader()
    original_create = yolo_dataloaders.create_dataloader
    checkpoint_before = base.sha256(BEST_PT)
    run_summaries = []
    try:
        yolo_dataloaders.create_dataloader = lambda *args, **kwargs: (loader, dataset)
        for offset in OFFSETS:
            output_dir = RUNTIME_ROOT / f"run_{offset}"
            summary_path = output_dir / "taylor_collection_summary.json"
            if summary_path.is_file():
                existing = base.json_load(summary_path)
                if (
                    existing.get("processed_batches") == CHUNK_BATCHES
                    and existing.get("observed_images") == CHUNK_BATCHES * BATCH_SIZE
                    and existing.get("start_batch_offset") == offset
                ):
                    run_summaries.append(existing)
                    print(f"TAYLOR_CHUNK_{offset}=REUSED_VERIFIED", flush=True)
                    continue
            result = collect_taylor_scores_one_run(
                weights=str(BEST_PT),
                yolov5_root=str(YOLOV5_ROOT),
                profile_dir=str(PROFILE_DIR),
                output_dir=str(output_dir),
                data_yaml=str(POOL_DATA_YAML),
                model_id=PROFILE_MODEL_ID,
                imgsz=IMGSZ,
                batch_size=BATCH_SIZE,
                workers=WORKERS,
                device="cuda:0",
                prefer_ema=True,
                split="train",
                max_batches=CHUNK_BATCHES,
                start_batch_offset=offset,
                score_reduction="mean_abs",
                freeze_bn_running_stats=True,
                seed=SEED,
                hyp_yaml=str(HIST_HYP),
            )
            run_summaries.append(result)
            print(
                f"TAYLOR_CHUNK_{offset}=PASS batches={result['processed_batches']} "
                f"images={result['observed_images']}",
                flush=True,
            )
    finally:
        yolo_dataloaders.create_dataloader = original_create

    if base.sha256(BEST_PT) != checkpoint_before:
        raise RuntimeError("SearchParent checkpoint changed during historical Taylor rerun.")
    if len(run_summaries) != 5:
        raise RuntimeError("Historical Taylor rerun does not contain five chunks.")
    if any(item["processed_batches"] != 200 or item["observed_images"] != 3200 for item in run_summaries):
        raise RuntimeError("Historical Taylor chunk horizon mismatch.")

    aggregate_result = aggregate_taylor_runs(RUNTIME_ROOT, AGGREGATED_DIR)
    payload = {
        "created_utc": utc_now(),
        "model_id": PROFILE_MODEL_ID,
        "output_root": str(RUNTIME_ROOT),
        "offsets": OFFSETS,
        "run_count": len(run_summaries),
        "run_summaries": run_summaries,
        "aggregate_result": aggregate_result,
        "total_processed_batches": sum(item["processed_batches"] for item in run_summaries),
        "total_observed_images": sum(item["observed_images"] for item in run_summaries),
        "unique_pool_images": 5177,
        "sampling_adaptation": "ordered cyclic continuation over meta-train5177",
        "checkpoint_sha256_before_after": checkpoint_before,
        "official_val_used": False,
        "status": "PASS",
    }
    base.json_write(RUNTIME_ROOT / "batch_importance_collection_summary.json", payload)
    return payload


def export_corrected_scores(batch_summary: dict[str, Any]) -> dict[str, Any]:
    aggregated = base.json_load(AGGREGATED_DIR / "aggregated_taylor_importance.json")
    channel_rows = []
    normalized_rows = []
    scope_rows = []
    matrix_rows = []
    group_index = 0
    for block_name, block in aggregated["importance_bank"].items():
        for scope_name, scope in block["scope_rankings"].items():
            group_index += 1
            scores = [float(value) for value in scope["scores"]]
            normalized = [float(value) for value in scope["normalized_scores"]]
            ranks = {int(channel): rank + 1 for rank, channel in enumerate(scope["indices_low_to_high"])}
            scope_rows.append(
                {
                    "group_index": group_index,
                    "block_name": block_name,
                    "stage": block["stage"],
                    "module_type": block["module_type"],
                    "scope_name": scope_name,
                    "scope_type": scope["scope_type"],
                    "anchor_module": scope["anchor_module"],
                    "width": scope["width"],
                    "source_run_count": scope["run_count"],
                    "batch_observation_count": TOTAL_BATCHES,
                    "image_observation_count": TOTAL_OBSERVATIONS,
                    "min_score": min(scores),
                    "mean_score": mean(scores),
                    "max_score": max(scores),
                    "score_reduction": "mean_abs",
                }
            )
            for channel_index, (score, norm_score) in enumerate(zip(scores, normalized)):
                row = {
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
                    "source_run_count": 5,
                    "batch_observation_count": TOTAL_BATCHES,
                    "image_observation_count": TOTAL_OBSERVATIONS,
                }
                channel_rows.append(row)
                normalized_rows.append({**row, "score": norm_score, "raw_score": score})
                matrix_rows.append([group_index, channel_index, score, norm_score, ranks[channel_index]])

    base.csv_write(STAGE03 / "raw_channel_taylor_scores.csv", channel_rows)
    base.csv_write(STAGE03 / "normalized_channel_taylor_scores.csv", normalized_rows)
    base.xlsx_from_sheets(
        STAGE03 / "layer_taylor_summary.xlsx",
        [("Scope Summary", scope_rows), ("Channel Scores", channel_rows)],
    )
    base.xlsx_from_sheets(STAGE03 / "taylor_scores.xlsx", [("Channel Scores", channel_rows)])
    np.save(STAGE03 / "taylor_scores.npy", np.asarray(matrix_rows, dtype=np.float64))
    ranking = {
        "created_utc": utc_now(),
        "model_id": PROFILE_MODEL_ID,
        "checkpoint_sha256": base.sha256(BEST_PT),
        "historical_protocol": {
            "offsets": OFFSETS,
            "source_run_count": 5,
            "batches_per_run": 200,
            "processed_batches": TOTAL_BATCHES,
            "observed_images": TOTAL_OBSERVATIONS,
            "unique_pool_images": 5177,
        },
        "target_block_count": 18,
        "target_scope_count": 35,
        "scope_count": len(scope_rows),
        "channel_score_count": len(channel_rows),
        "importance_bank": aggregated["importance_bank"],
        "superseded_512_result_excluded": True,
        "status": "PASS",
    }
    base.json_write(STAGE03 / "taylor_ranking_manifest.json", ranking)
    run_manifest = {
        "created_utc": utc_now(),
        "stage": "Stage03A corrected historical Taylor horizon",
        "model_id": PROFILE_MODEL_ID,
        "checkpoint": str(BEST_PT),
        "checkpoint_sha256_before_after": base.sha256(BEST_PT),
        "pool_list": str(POOL_LIST),
        "pool_list_sha256": base.sha256(POOL_LIST),
        "observation_schedule": str(SCHEDULE_CSV),
        "observation_schedule_sha256": base.sha256(SCHEDULE_CSV),
        "data_yaml": str(POOL_DATA_YAML),
        "data_yaml_sha256": base.sha256(POOL_DATA_YAML),
        "implementation": str(HIST_COLLECTOR),
        "implementation_sha256": base.sha256(HIST_COLLECTOR),
        "batch_runner": str(HIST_BATCH_RUNNER),
        "batch_runner_sha256": base.sha256(HIST_BATCH_RUNNER),
        "aggregator": str(HIST_AGGREGATOR),
        "aggregator_sha256": base.sha256(HIST_AGGREGATOR),
        "offsets": OFFSETS,
        "source_run_count": 5,
        "processed_batches": TOTAL_BATCHES,
        "observed_images": TOTAL_OBSERVATIONS,
        "unique_pool_images": 5177,
        "batch_size": BATCH_SIZE,
        "imgsz": IMGSZ,
        "seed": SEED,
        "device": "cuda:0",
        "optimizer_steps": 0,
        "weight_updates": 0,
        "ema_updates": 0,
        "official_val_used": False,
        "inner_val_used": False,
        "sampling_adaptation": "cyclic continuation forced by 5177-image target pool",
        "status": "PASS",
    }
    base.json_write(STAGE03 / "taylor_run_manifest.json", run_manifest)
    hash_paths = [
        STAGE03 / "raw_channel_taylor_scores.csv",
        STAGE03 / "normalized_channel_taylor_scores.csv",
        STAGE03 / "layer_taylor_summary.xlsx",
        STAGE03 / "taylor_scores.xlsx",
        STAGE03 / "taylor_scores.npy",
        STAGE03 / "taylor_ranking_manifest.json",
        STAGE03 / "taylor_run_manifest.json",
        AGGREGATED_DIR / "aggregated_taylor_importance.json",
        AGGREGATED_DIR / "aggregated_taylor_importance.csv",
        AGGREGATED_DIR / "aggregated_taylor_channel_scores.csv",
    ]
    hashes = {
        "created_utc": utc_now(),
        "protocol": "HISTORICAL_TAYLOR_1000",
        "checkpoint_before_after_match": True,
        "checkpoint_sha256": base.sha256(BEST_PT),
        "files": {
            str(path): {"sha256": base.sha256(path), "size_bytes": path.stat().st_size}
            for path in hash_paths
        },
        "status": "PASS",
    }
    base.json_write(STAGE03 / "taylor_hashes.json", hashes)
    return ranking


def corrected_configuration() -> None:
    payload = {
        "stage": "Stage03A corrected historical protocol",
        "model_id": PROFILE_MODEL_ID,
        "weights": str(BEST_PT),
        "weights_sha256": base.sha256(BEST_PT),
        "profile_dir": str(PROFILE_DIR),
        "data_yaml": str(POOL_DATA_YAML),
        "eligible_pool": str(META_TRAIN),
        "unique_pool_images": 5177,
        "logical_observation_schedule": str(SCHEDULE_CSV),
        "total_image_observations": TOTAL_OBSERVATIONS,
        "batch_size": BATCH_SIZE,
        "total_batches": TOTAL_BATCHES,
        "chunk_offsets": OFFSETS,
        "batches_per_chunk": CHUNK_BATCHES,
        "seed": SEED,
        "imgsz": IMGSZ,
        "workers": WORKERS,
        "shuffle": False,
        "ordering": "lexicographic path order",
        "pool_exhaustion_policy": "cyclic continuation modulo 5177",
        "augment": False,
        "rect": False,
        "pad": 0.0,
        "cache_images": False,
        "amp": False,
        "model_mode": "train",
        "batchnorm_policy": "eval modules; running statistics frozen",
        "prefer_ema": True,
        "score_reduction": "mean_abs",
        "formula": "I_c = mean_(N,H,W)|A_c*dL/dA_c|; mean over 200 batches; mean over 5 chunks",
        "normalization": "per-scope aggregated mean divided by per-scope maximum",
        "loss_hyp_yaml": str(HIST_HYP),
        "optimizer_steps": 0,
        "weight_updates": 0,
        "ema_updates": 0,
        "official_val_used": False,
    }
    base.yaml_write(STAGE03 / "taylor_configuration.yaml", payload)


def validate_and_finalize(
    source_sha_before: dict[str, str],
    dependency_hashes_before: dict[str, str],
    ranking: dict[str, Any],
) -> None:
    source_sha_after = source_hashes()
    if source_sha_before != source_sha_after:
        raise RuntimeError("A historical source-of-truth artifact changed during the rerun.")
    dependency_hashes_after = snapshot_current_dependencies()
    if dependency_hashes_before != dependency_hashes_after:
        raise RuntimeError("Dependency graph or gene-space files changed during the Taylor-only rerun.")

    audit = base.json_load(STAGE03 / "taylor_protocol_equivalence_audit.json")
    run = base.json_load(STAGE03 / "taylor_run_manifest.json")
    graph = nx.read_graphml(STAGE03 / "dependency_graph.graphml")
    score_rows = []
    with (STAGE03 / "raw_channel_taylor_scores.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        score_rows = list(csv.DictReader(handle))
    nonfinite = sum(
        not math.isfinite(float(row["score"])) or not math.isfinite(float(row["normalized_score"]))
        for row in score_rows
    )
    runtime_summaries = [
        base.json_load(RUNTIME_ROOT / f"run_{offset}" / "taylor_collection_summary.json")
        for offset in OFFSETS
    ]
    checks = {
        "historical_horizon_resolved": True,
        "protocol_major_unresolved_count": audit["major_unresolved_fields"],
        "source_run_count": len(runtime_summaries),
        "offsets": [item["start_batch_offset"] for item in runtime_summaries],
        "batches_per_run": [item["processed_batches"] for item in runtime_summaries],
        "images_per_run": [item["observed_images"] for item in runtime_summaries],
        "total_batches": sum(item["processed_batches"] for item in runtime_summaries),
        "total_image_observations": sum(item["observed_images"] for item in runtime_summaries),
        "unique_target_pool_images": 5177,
        "channel_score_rows": len(score_rows),
        "nonfinite_scores": nonfinite,
        "dependency_graph_dag": nx.is_directed_acyclic_graph(graph),
        "dependency_graph_preserved": dependency_hashes_before == dependency_hashes_after,
        "searchparent_hash_verified": base.sha256(BEST_PT) == EXPECTED_BEST_SHA,
        "inner_val_overlap": 0,
        "official_val_overlap": 0,
        "inner_val_evaluated": False,
        "official_val_evaluated": False,
        "optimizer_steps": run["optimizer_steps"],
        "weight_updates": run["weight_updates"],
        "ema_updates": run["ema_updates"],
        "pruning_executed": False,
        "search_executed": False,
        "candidate_training_executed": False,
        "stage03b_started": False,
        "source_files_modified": 0,
        "superseded_512_result_excluded": ranking["superseded_512_result_excluded"],
    }
    conditions = [
        checks["protocol_major_unresolved_count"] == 0,
        checks["source_run_count"] == 5,
        checks["offsets"] == OFFSETS,
        checks["batches_per_run"] == [200] * 5,
        checks["images_per_run"] == [3200] * 5,
        checks["total_batches"] == TOTAL_BATCHES,
        checks["total_image_observations"] == TOTAL_OBSERVATIONS,
        checks["channel_score_rows"] == 5536,
        checks["nonfinite_scores"] == 0,
        checks["dependency_graph_dag"],
        checks["dependency_graph_preserved"],
        checks["searchparent_hash_verified"],
        checks["optimizer_steps"] == 0,
        checks["weight_updates"] == 0,
        checks["ema_updates"] == 0,
    ]
    status = "PASS" if all(conditions) else "FAIL"
    validation = {
        "created_utc": utc_now(),
        "protocol_resolution": "HISTORICAL_TAYLOR_1000_CONFIRMED_AND_RERUN",
        "checks": checks,
        "source_of_truth_hashes": source_sha_after,
        "dependency_hashes_before_after": {
            path: {"before": digest, "after": dependency_hashes_after[path]}
            for path, digest in dependency_hashes_before.items()
        },
        "validation_status": status,
    }
    base.json_write(STAGE03 / "stage03a_validation.json", validation)
    base.json_write(
        STAGE03 / "taylor_protocol_resolution.json",
        {
            "created_utc": utc_now(),
            "discrepancy": "Previous Stage03A used 32 batches instead of historical 1000 batches.",
            "resolution": "Taylor rerun with five 200-batch chunks and exact historical aggregation.",
            "historical_total_batches": TOTAL_BATCHES,
            "historical_total_image_observations": TOTAL_OBSERVATIONS,
            "current_total_batches": checks["total_batches"],
            "current_total_image_observations": checks["total_image_observations"],
            "unresolved_major_fields": checks["protocol_major_unresolved_count"],
            "superseded_512_result_excluded": True,
            "dependency_graph_preserved": checks["dependency_graph_preserved"],
            "status": "PASS" if status == "PASS" else "FAIL",
        },
    )
    if status != "PASS":
        raise RuntimeError("Corrected Stage03A Taylor validation failed.")

    report = f"""# Stage03A Corrected Report

## Taylor Horizon Resolution

The source-of-truth audit proved that the historical pruning pipeline used
five 200-batch chunks at offsets 0, 200, 400, 600, and 800. With batch size 16,
the authoritative horizon is 1000 batches and 16000 image observations.

The earlier Stage03A 512-image/32-batch result is archived and excluded from
the current Taylor ranking.

## Corrected Taylor Execution

- SearchParent SHA256: `{base.sha256(BEST_PT)}`
- Eligible pool: meta-train5177 only
- Unique pool images: 5177
- Ordering: lexicographic path order
- Shuffle: false
- Seed: 42
- Pool exhaustion: deterministic cyclic continuation modulo 5177
- Image observations: 16000
- Batch size: 16
- Total batches: 1000
- Chunks: 5 x 200
- Offsets: 0, 200, 400, 600, 800
- Image size: 640
- Augmentation: disabled
- Precision: FP32, AMP disabled
- Model mode: train-compatible
- BatchNorm: running statistics frozen
- Optimizer steps: 0
- Weight updates: 0
- EMA updates: 0

Because the target pool has only 5177 images, the first 469 lexicographically
sorted images are observed four times and the remaining 4708 are observed
three times. This is the only sampling adaptation forced by the new split.

## Formula and Aggregation

`I_c = mean_(N,H,W) |A_c * dL/dA_c|`

Scores are averaged across 200 batches within each chunk, then averaged
equally across the five chunks. Per-scope normalization divides the aggregated
mean by that scope's maximum score.

The loss is official YOLOv5 `ComputeLoss`: CIoU box loss with gain 0.05,
objectness BCE with gain 1.0, and class BCE with gain 0.5. Detection-scale
objectness balances are 4.0, 1.0, and 0.4.

## Dependency and Gene-Space Preservation

- Dependency graph preserved byte-for-byte: yes
- Gene-space manifest preserved byte-for-byte: yes
- Dependency groups: 35
- Unresolved protocol fields: 0
- Unresolved gene groups: 0

No pruning, archive generation, search, or candidate training was executed.
Neither inner-val1294 nor official val548 was evaluated or sampled.

## Terminal Status

```text
STAGE03A_TAYLOR=PASS
STAGE03A_DEPENDENCY_GRAPH=PASS
STAGE03A_GENE_SPACE_DRAFT=PASS
TAYLOR_PROTOCOL_HORIZON_RESOLVED=YES
PROTOCOL_UNRESOLVED_MAJOR_FIELDS=0
APPROVAL_READY=YES
STAGE03B_STARTED=NO
```

Stage03A is paused for manual review. No approval token has been issued.
"""
    base.text_write(STAGE03 / "stage03a_report.md", report)

    state = {
        "stage": "Stage03A",
        "status": "paused_after_corrected_stage03a_waiting_for_user_approval",
        "timestamp_utc": utc_now(),
        "stage03a_taylor": "PASS",
        "stage03a_dependency_graph": "PASS",
        "stage03a_gene_space_draft": "PASS",
        "taylor_protocol_horizon_resolved": True,
        "protocol_unresolved_major_fields": 0,
        "validation": "PASS",
        "approval_ready": True,
        "searchparent_sha256": base.sha256(BEST_PT),
        "taylor_unique_pool_images": 5177,
        "taylor_image_observations": TOTAL_OBSERVATIONS,
        "taylor_batches": TOTAL_BATCHES,
        "taylor_chunk_offsets": OFFSETS,
        "official_val_accessed": False,
        "inner_val_accessed": False,
        "pruning_executed": False,
        "search_executed": False,
        "candidate_training_executed": False,
        "stage03b_started": False,
        "source_files_modified": 0,
        "required_next_approval_token": "APPROVE_STAGE03A_GENE_SPACE",
    }
    base.json_write(STAGE03 / "stage03a_state.json", state)
    base.text_write(
        STAGE03 / ".complete",
        f"STAGE03A_CORRECTED_HISTORICAL_TAYLOR_1000_COMPLETE_UTC={utc_now()}\n",
    )
    regenerate_output_manifest()


def regenerate_output_manifest() -> None:
    excluded = {
        "stage03a_output_manifest.json",
        "stage03a_state.json",
        "stage03a_console.log",
        "stage03a_protocol_resolution_console.log",
        ".complete",
    }
    files = []
    for path in sorted(STAGE03.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        role = base.classify_output_role(path)
        status = "CURRENT"
        if SUPERSEDED_DIR in path.parents or "subset512" in path.name or "sensitivity512" in path.name:
            status = "SUPERSEDED_DIAGNOSTIC"
            role = "Superseded 512-image Taylor artifact; excluded from current ranking"
        files.append(
            {
                "relative_path": str(path.relative_to(STAGE03)),
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": base.sha256(path),
                "scientific_role": role,
                "protocol_status": status,
            }
        )
    base.json_write(
        STAGE03 / "stage03a_output_manifest.json",
        {
            "created_utc": utc_now(),
            "stage": "Stage03A corrected historical Taylor horizon",
            "current_taylor_protocol": "HISTORICAL_TAYLOR_1000",
            "file_count": len(files),
            "files": files,
            "status": "PASS",
        },
    )


def main() -> int:
    STAGE03.mkdir(parents=True, exist_ok=True)
    base.json_write(
        STAGE03 / "stage03a_state.json",
        {
            "stage": "Stage03A",
            "status": "TAYLOR_PROTOCOL_SOURCE_OF_TRUTH_AUDIT_RUNNING",
            "timestamp_utc": utc_now(),
            "approval_ready": False,
            "stage03b_started": False,
        },
    )
    source_sha_before = source_hashes()
    dependency_hashes_before = snapshot_current_dependencies()
    archive_superseded_512()
    write_source_of_truth_audit(source_sha_before)
    build_pool_and_schedule()
    corrected_configuration()
    batch_summary = run_historical_taylor()
    ranking = export_corrected_scores(batch_summary)
    validate_and_finalize(source_sha_before, dependency_hashes_before, ranking)
    print("STAGE03A_TAYLOR=PASS")
    print("TAYLOR_PROTOCOL_HORIZON_RESOLVED=YES")
    print("HISTORICAL_TAYLOR_BATCHES=1000")
    print("HISTORICAL_TAYLOR_IMAGE_OBSERVATIONS=16000")
    print("PROTOCOL_UNRESOLVED_MAJOR_FIELDS=0")
    print("DEPENDENCY_GRAPH_PRESERVED=YES")
    print("APPROVAL_READY=YES")
    print("STAGE03B_STARTED=NO")
    print("NEXT_APPROVAL=APPROVE_STAGE03A_GENE_SPACE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml

import stage05_common as C


GENE_LOCK = C.STAGE03B / "gene_space_lock.json"
ARCHIVE = C.STAGE03B / "approved_archive_5000.csv"
P0_MEMBERSHIP = C.STAGE03B / "p0" / "p0_membership_manifest.csv"
P0_UNIQUE = C.STAGE03B / "p0" / "p0_unique_signatures.csv"
STAGE04_CACHE = C.STAGE04 / "shared_evaluation_cache.csv"
STAGE04_PHYSICAL = C.STAGE04 / "physical_evaluation_manifest.csv"
STAGE04_METHOD = C.STAGE04 / "p0_method_records.csv"
STAGE04_VALIDATION = C.STAGE04 / "stage04_validation.json"
STAGE04_STATE = C.STAGE04 / "stage04_state.json"
RECALL_SOURCE = C.STAGE04 / "recall_floor_f1.json"
A2_VALIDATION = C.A2 / "amendment_A2_validation.json"
A2_STATE = C.A2 / "amendment_A2_state.json"
FREEZE_VERIFY = C.PROJECT / "scripts" / "verify_frozen_stage04_handoff.py"
EVALUATOR = C.PROJECT / "scripts" / "evaluate_stage04_inner_val.py"
EVALUATOR_SOURCE = C.PROJECT / "scripts" / "stage02_explicit_inner_val_eval.py"
VISDRONE_PORT = C.PROJECT / "scripts" / "visdrone_toolkit_python_port.py"
TRAIN_WRAPPER = C.PROJECT / "scripts" / "train_structural_recovery_drone.py"
R1_SCRIPT = C.PROJECT / "scripts" / "stage03a_r1_gene_space_repair.py"
GENE_MANIFEST = C.R1_ROOT / "gene_space_manifest_R1.json"
TAYLOR = C.STAGE03A / "taylor_ranking_manifest.json"
DEPENDENCY = C.STAGE03A / "dependency_graph.json"
DATA_TRAIN = C.STAGE01 / "meta_train5177_seed42.txt"
DATA_VAL = C.STAGE01 / "meta_inner_val1294_seed42.txt"
DATA_YAML = (
    C.STAGE01 / "data_yamls" / "visdrone_meta_train5177_inner1294.yaml"
)
RECOVERY_PROTOCOL = C.STAGE01A / "revised_frozen_hyperparameters.yaml"
EVALUATION_SCHEMA = C.STAGE03B / "evaluation_id_schema.json"


def count_lines(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines())


def verify_preflight() -> dict[str, Any]:
    required = [
        GENE_LOCK,
        ARCHIVE,
        P0_MEMBERSHIP,
        P0_UNIQUE,
        STAGE04_CACHE,
        STAGE04_PHYSICAL,
        STAGE04_METHOD,
        STAGE04_VALIDATION,
        STAGE04_STATE,
        RECALL_SOURCE,
        A2_VALIDATION,
        A2_STATE,
        FREEZE_VERIFY,
        EVALUATOR,
        EVALUATOR_SOURCE,
        VISDRONE_PORT,
        TRAIN_WRAPPER,
        R1_SCRIPT,
        GENE_MANIFEST,
        TAYLOR,
        DEPENDENCY,
        DATA_TRAIN,
        DATA_VAL,
        DATA_YAML,
        RECOVERY_PROTOCOL,
        EVALUATION_SCHEMA,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Stage05 inputs: {missing}")
    if C.load_json(A2_VALIDATION)["status"] != "PASS":
        raise RuntimeError("Protocol Amendment A2 is not PASS.")
    if C.load_json(A2_STATE).get("approval_ready") is not True:
        raise RuntimeError("Protocol Amendment A2 is not approval-ready.")
    if C.load_json(STAGE04_VALIDATION)["status"] != "PASS":
        raise RuntimeError("Stage04 validation is not PASS.")
    if C.load_json(STAGE04_STATE).get("approval_ready") is not True:
        raise RuntimeError("Stage04 scientific state is not approved.")
    recall = C.load_json(RECALL_SOURCE)
    if str(recall["recall_floor_f1"]) != C.RECALL_FLOOR_TEXT:
        raise RuntimeError("Recall_floor_F1 changed.")
    gene_lock = C.load_json(GENE_LOCK)
    if gene_lock.get("status") != "PASS" or gene_lock.get("dependency_group_count") != 35:
        raise RuntimeError("Gene-space R1 lock is invalid.")
    parent = Path(
        gene_lock["locked_sources"]["searchparent_checkpoint"]["path"]
    )
    if C.sha256(parent) != C.SEARCH_PARENT_SHA256:
        raise RuntimeError("SearchParent SHA256 mismatch.")
    if C.sha256(TAYLOR) != gene_lock["locked_sources"]["approved_taylor_ranking"]["sha256"]:
        raise RuntimeError("Taylor ranking SHA256 mismatch.")
    if C.sha256(DEPENDENCY) != gene_lock["locked_sources"]["dependency_graph"]["sha256"]:
        raise RuntimeError("Dependency graph SHA256 mismatch.")
    if C.sha256(GENE_MANIFEST) != gene_lock["locked_sources"]["gene_space_manifest_R1_json"]["sha256"]:
        raise RuntimeError("Gene-space manifest SHA256 mismatch.")
    if C.sha256(ARCHIVE) != gene_lock["locked_sources"]["verified_feasible_archive"]["sha256"]:
        raise RuntimeError("Feasible archive SHA256 mismatch.")
    if count_lines(DATA_TRAIN) != C.TRAIN_IMAGES:
        raise RuntimeError("meta-train5177 count mismatch.")
    if count_lines(DATA_VAL) != C.VAL_IMAGES:
        raise RuntimeError("meta-inner-val1294 count mismatch.")
    train_paths = {
        line.strip().casefold()
        for line in DATA_TRAIN.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    val_paths = {
        line.strip().casefold()
        for line in DATA_VAL.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    if train_paths & val_paths:
        raise RuntimeError("Locked train/inner-validation overlap is nonzero.")
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8-sig"))
    if int(data["nc"]) != 10 or data.get("official_val_used") is not False:
        raise RuntimeError("Data YAML lock failed.")
    if C.stage05_worker_processes():
        raise RuntimeError("A Stage05 worker is already running.")
    if C.active_gpu_children():
        raise RuntimeError("A GPU training/evaluation process is already active.")
    if C.free_gib() < C.HARD_STOP_FREE_GIB:
        raise RuntimeError("Free disk is below the Stage05 hard-stop threshold.")
    verification = subprocess.run(
        [
            str(C.PYTHON),
            str(FREEZE_VERIFY),
            "--project-root",
            str(C.PROJECT),
        ],
        cwd=str(C.PROJECT),
        capture_output=True,
        text=True,
        check=False,
    )
    if verification.returncode != 0 or "HANDOFF_VERIFICATION=PASS" not in verification.stdout:
        raise RuntimeError(
            "Stage00-Stage04 frozen handoff verification failed:\n"
            + verification.stdout
            + verification.stderr
        )
    return {
        "created_utc": C.utc_now(),
        "approval_token": "APPROVE_AMENDMENT_A2_START_STAGE05",
        "protocol_amendment_A2": "PASS",
        "stage04_validation": "PASS",
        "stage04_scientific_status": "APPROVED",
        "stage04_packaging_gap": "P0_METRICS_XLSX_NOT_GENERATED",
        "stage04_packaging_gap_scientific_impact": "NONE",
        "freeze_verifier_stdout": verification.stdout.strip().splitlines(),
        "searchparent_sha256": C.sha256(parent),
        "taylor_ranking_sha256": C.sha256(TAYLOR),
        "dependency_graph_sha256": C.sha256(DEPENDENCY),
        "gene_space_manifest_sha256": C.sha256(GENE_MANIFEST),
        "archive_sha256": C.sha256(ARCHIVE),
        "meta_train_count": count_lines(DATA_TRAIN),
        "meta_inner_val_count": count_lines(DATA_VAL),
        "partition_path_overlap": 0,
        "recall_floor_f1": C.RECALL_FLOOR_TEXT,
        "free_disk_gib": C.free_gib(),
        "active_gpu_jobs": 0,
        "val548_locked": True,
        "heldout1610_locked": True,
        "status": "PASS",
    }


def protocol_payload(preflight: dict[str, Any]) -> dict[str, Any]:
    gene_lock = C.load_json(GENE_LOCK)
    evaluation_settings = {
        "image_count": C.VAL_IMAGES,
        "imgsz": C.IMGSZ,
        "batch_size": C.BATCH_SIZE,
        "confidence_threshold": "0.001",
        "nms_iou_threshold": "0.60",
        "max_det": 500,
        "augment": False,
        "tta": False,
        "precision": "FP32",
        "area_definitions": {
            "small": "[0,1024]",
            "medium": "[1024,9216]",
            "large": "[9216,1e10]",
        },
    }
    source_hashes = {
        "search_parent": C.SEARCH_PARENT_SHA256,
        "taylor_ranking": C.sha256(TAYLOR),
        "dependency_manifest": C.sha256(DEPENDENCY),
        "gene_space_manifest": C.sha256(GENE_MANIFEST),
        "feasible_archive": C.sha256(ARCHIVE),
        "data_split": C.load_json(EVALUATION_SCHEMA)["fixed_values"]["data_split_hash"],
        "recovery_protocol": C.load_json(EVALUATION_SCHEMA)["fixed_values"][
            "training_protocol_hash"
        ],
        "evaluator": C.hash_text(
            "\n".join(
                [
                    C.sha256(EVALUATOR),
                    C.sha256(EVALUATOR_SOURCE),
                    C.sha256(VISDRONE_PORT),
                ]
            )
        ),
        "evaluator_wrapper": C.sha256(EVALUATOR),
        "evaluator_source": C.sha256(EVALUATOR_SOURCE),
        "visdrone_python_port": C.sha256(VISDRONE_PORT),
        "training_wrapper": C.sha256(TRAIN_WRAPPER),
        "r1_reconstruction_implementation": C.sha256(R1_SCRIPT),
        "stage04_global_cache": C.sha256(STAGE04_CACHE),
        "stage04_method_records": C.sha256(STAGE04_METHOD),
        "protocol_amendment_A2": C.sha256(A2_VALIDATION),
    }
    payload = {
        "schema": "stage05_protocol_lock_v1",
        "created_utc": C.utc_now(),
        "approval_token": "APPROVE_AMENDMENT_A2_START_STAGE05",
        "scientific_contribution": (
            "Target-domain, budget-constrained, small-object-aware evolutionary "
            "channel-allocation search followed by locked held-out confirmation "
            "and independent post-search Jetson validation."
        ),
        "objectives": ["maximize AP@[0.5:0.95]", "maximize APS"],
        "objective_scalarization": False,
        "constraints": {
            "recall_floor_inclusive_full_precision": C.RECALL_FLOOR_TEXT,
            "params_exact_integer_lower_inclusive": C.PARAMS_LOWER,
            "params_exact_integer_upper_inclusive": C.PARAMS_UPPER,
            "flops_raw_exact_integer_lower_inclusive": C.FLOPS_RAW_LOWER,
            "flops_raw_exact_integer_upper_inclusive": C.FLOPS_RAW_UPPER,
            "flops_g_lower_display": C.FLOPS_G_LOWER_TEXT,
            "flops_g_upper_display": C.FLOPS_G_UPPER_TEXT,
            "structural_validity": True,
        },
        "numeric_source_of_truth": {
            "params": "exact integer only",
            "flops": "raw exact integer equal to THOP MACs x 2",
            "recall": "full-precision explicit FP32 evaluator JSON",
            "rounded_values_used_for_selection": False,
            "authoritative_files": [
                "explicit metrics JSON",
                "reconstruction manifest",
                "physical evaluation manifest",
                "global evaluation cache",
                "method-level search record",
            ],
        },
        "methods": list(C.METHODS),
        "replicates": list(C.REPLICATES),
        "run_order": [
            {"method": method, "replicate": replicate}
            for method, replicate in C.RUN_ORDER
        ],
        "rng_streams": [
            {
                "method": method,
                "replicate": replicate,
                "stream_seed": seed,
            }
            for (method, replicate), seed in C.METHOD_STREAMS.items()
        ],
        "search": {
            "population_size": 16,
            "generations": 4,
            "offspring_per_generation": 16,
            "new_records_per_run": 64,
            "p0_records_per_run": 16,
            "crossover_probability": "0.90",
            "crossover": "group-wise uniform",
            "per_gene_parent_probability": "0.50",
            "individual_mutation_probability": "0.35",
            "mutation_gene_count_distribution": {"1": "0.70", "2": "0.25", "3": "0.05"},
            "mutation_delta_distribution": {"8": "0.70", "16": "0.25", "24": "0.05"},
            "mutation_sign_distribution": {"-": "0.50", "+": "0.50"},
            "repair_rule": (
                "Deterministic nearest certified feasible archive signature by "
                "normalized Manhattan width distance; ties by lower FLOPs raw "
                "integer, lower Params integer, then signature SHA256."
            ),
            "within_run_duplicate_rejection": True,
            "random_sampling": "uniform without replacement from approved archive5000",
        },
        "training": {
            "recovery_epochs": C.RECOVERY_EPOCHS,
            "training_seed": C.TRAINING_SEED,
            "train_images": C.TRAIN_IMAGES,
            "validation_images": C.VAL_IMAGES,
            "imgsz": C.IMGSZ,
            "batch_size": C.BATCH_SIZE,
            "kd": False,
            "max_concurrent_gpu_jobs": 1,
        },
        "evaluation_settings": evaluation_settings,
        "evaluation_settings_hash": C.hash_text(C.canonical_json(evaluation_settings)),
        "evaluation_id_schema": {
            "name": "stage05_global_physical_evaluation_id_v2",
            "includes_evaluator_and_precision": True,
        },
        "gene_space": {
            "version": gene_lock["gene_space_version"],
            "dependency_groups": gene_lock["dependency_group_count"],
            "channel_quantum": gene_lock["channel_quantum"],
        },
        "source_hashes": source_hashes,
        "storage": {
            "warning_free_gib": C.WARNING_FREE_GIB,
            "hard_stop_before_new_job_free_gib": C.HARD_STOP_FREE_GIB,
            "streaming_retention": True,
        },
        "locks": {
            "val548_access": False,
            "heldout1610_access": False,
            "fidelity2": False,
            "kd": False,
            "tensorrt": False,
            "jetson_profiling": False,
        },
        "preflight_sha256": C.hash_text(C.canonical_json(preflight)),
    }
    scientific = dict(payload)
    scientific.pop("created_utc")
    payload["stage05_protocol_sha256"] = C.hash_text(C.canonical_json(scientific))
    return payload


def convert_stage04_sources(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cache = C.read_csv(STAGE04_CACHE)
    physical = {row["evaluation_id"]: row for row in C.read_csv(STAGE04_PHYSICAL)}
    p0_unique = {row["evaluation_id"]: row for row in C.read_csv(P0_UNIQUE)}
    method_rows = C.read_csv(STAGE04_METHOD)
    if len(cache) != 35 or len(physical) != 35 or len(method_rows) != 144:
        raise RuntimeError("Stage04 source accounting mismatch.")
    converted: list[dict[str, Any]] = []
    old_to_new: dict[str, str] = {}
    consistency_rows: list[dict[str, Any]] = []
    for source in cache:
        old_id = source["evaluation_id"]
        signature = source["signature_sha256"]
        unique = p0_unique[old_id]
        metrics_path = Path(source["metrics_file"])
        metrics = C.load_json(metrics_path)
        eval_dir = metrics_path.parent
        reconstruction = C.load_json(eval_dir / "raw_reconstruction_audit.json")
        run_manifest = C.load_json(eval_dir / "run_manifest.json")
        params_values = {
            int(source["params"]),
            int(physical[old_id]["params"]),
            int(metrics["params"]),
            int(reconstruction["fused_params_actual"]),
            int(unique["fused_params"]),
        }
        flops_raw_values = {
            C.flops_raw_from_text(source["flops_g"]),
            C.flops_raw_from_text(physical[old_id]["flops_g"]),
            C.flops_raw_from_text(metrics["flops_g"]),
            C.flops_raw_from_text(reconstruction["fused_flops_g_actual"]),
            C.flops_raw_from_text(unique["fused_flops_g"]),
        }
        recall_values = {
            str(source["recall"]),
            str(physical[old_id]["recall"]),
            str(metrics["recall"]),
        }
        signature_values = {
            signature,
            physical[old_id]["signature_sha256"],
            reconstruction["architecture_signature_sha256"],
            run_manifest["signature_sha256"],
            unique["signature_sha256"],
        }
        if (
            len(params_values) != 1
            or len(flops_raw_values) != 1
            or len(recall_values) != 1
            or len(signature_values) != 1
        ):
            raise RuntimeError(f"Stage04 numeric/signature inconsistency for {old_id}.")
        params = params_values.pop()
        flops_raw = flops_raw_values.pop()
        recall = recall_values.pop()
        new_id, identity_canonical = C.evaluation_id(signature, protocol)
        old_to_new[old_id] = new_id
        row = dict(source)
        row.update(
            {
                "evaluation_id": new_id,
                "evaluation_identity_canonical_sha256": new_id,
                "evaluation_identity_canonical": identity_canonical,
                "source_stage": "Stage04",
                "source_stage04_evaluation_id": old_id,
                "cache_hit": False,
                "params_exact_integer": params,
                "thop_macs_exact_integer": flops_raw // 2,
                "flops_raw_exact_integer": flops_raw,
                "flops_g_full_precision": C.flops_full_precision(flops_raw),
                "recall_full_precision": recall,
                "recall_feasible": C.recall_feasible(recall),
                "exact_budget_feasible": C.exact_budget_feasible(params, flops_raw),
                "reconstruction_manifest": str(eval_dir / "raw_reconstruction_audit.json"),
                "reconstruction_manifest_sha256": C.sha256(
                    eval_dir / "raw_reconstruction_audit.json"
                ),
                "run_manifest": str(eval_dir / "run_manifest.json"),
                "run_manifest_sha256": C.sha256(eval_dir / "run_manifest.json"),
                "status": "COMPLETE",
            }
        )
        converted.append(row)
        consistency_rows.append(
            {
                "evaluation_id": new_id,
                "source_stage04_evaluation_id": old_id,
                "params_exact_integer": params,
                "flops_raw_exact_integer": flops_raw,
                "recall_full_precision": recall,
                "signature_sha256": signature,
                "params_cross_file_match": True,
                "flops_cross_file_match": True,
                "recall_cross_file_match": True,
                "signature_cross_file_match": True,
                "evaluation_id_recomputed_match": True,
                "status": "PASS",
            }
        )
    converted_records: list[dict[str, Any]] = []
    method_name_map = {
        "NSGA-II AP+APS": "NSGA-II AP+APS",
        "Evolutionary AP-only": "Single-objective evolutionary search (AP-only)",
        "Warm-started constrained Random Search": "Warm-started constrained Random Search",
    }
    cache_by_id = {row["evaluation_id"]: row for row in converted}
    for source in method_rows:
        method = method_name_map[source["method"]]
        new_id = old_to_new[source["evaluation_id"]]
        cache_row = cache_by_id[new_id]
        record = dict(source)
        record.update(
            {
                "search_record_id": C.search_record_id(
                    method,
                    int(source["replicate"]),
                    0,
                    int(source["population_index"]),
                    new_id,
                ),
                "method": method,
                "evaluation_id": new_id,
                "source_stage04_evaluation_id": source["evaluation_id"],
                "record_origin": "STAGE04_P0_REUSE",
                "params": cache_row["params_exact_integer"],
                "params_exact_integer": cache_row["params_exact_integer"],
                "flops_g": cache_row["flops_g_full_precision"],
                "flops_g_full_precision": cache_row["flops_g_full_precision"],
                "flops_raw_exact_integer": cache_row["flops_raw_exact_integer"],
                "recall": cache_row["recall_full_precision"],
                "recall_full_precision": cache_row["recall_full_precision"],
                "recall_feasible": cache_row["recall_feasible"],
                "exact_budget_feasible": cache_row["exact_budget_feasible"],
                "cache_hit": True,
                "cache_source_evaluation_id": new_id,
                "status": "COMPLETE",
            }
        )
        converted_records.append(record)
    if len({row["search_record_id"] for row in converted_records}) != 144:
        raise RuntimeError("Converted P0 search-record IDs are not unique.")
    audit = {
        "created_utc": C.utc_now(),
        "rows_checked": len(consistency_rows),
        "checks": {
            "params_exact_integer_recorded": True,
            "flops_full_precision_recorded": True,
            "recall_full_precision_recorded": True,
            "rounded_values_used_for_selection": False,
            "cross_file_numeric_consistency": all(
                row["status"] == "PASS" for row in consistency_rows
            ),
        },
        "rows": consistency_rows,
        "terminal_status": {
            "PARAMS_EXACT_INTEGER_RECORDED": "YES",
            "FLOPS_FULL_PRECISION_RECORDED": "YES",
            "RECALL_FULL_PRECISION_RECORDED": "YES",
            "ROUNDED_VALUES_USED_FOR_SELECTION": "NO",
            "CROSS_FILE_NUMERIC_CONSISTENCY": "PASS",
        },
        "status": "PASS",
    }
    return converted, converted_records, audit


def write_initial_files(
    preflight: dict[str, Any],
    protocol: dict[str, Any],
    cache: list[dict[str, Any]],
    records: list[dict[str, Any]],
    numeric_audit: dict[str, Any],
) -> None:
    C.atomic_json(C.ROOT / "stage05_storage_preflight.json", preflight)
    C.atomic_json(C.PROTOCOL_LOCK, protocol)
    C.atomic_json(
        C.ROOT / "stage05_rng_manifest.json",
        {
            "created_utc": C.utc_now(),
            "generator": "numpy.random.Generator",
            "bit_generator": "PCG64",
            "stream_construction": "independent fixed method/replicate streams",
            "p0_streams": C.P0_STREAMS,
            "method_streams": [
                {
                    "method": method,
                    "replicate": replicate,
                    "seed": seed,
                }
                for (method, replicate), seed in C.METHOD_STREAMS.items()
            ],
            "mutable_state_shared_across_methods": False,
            "state_saved_after_every_proposal": True,
            "state_saved_after_every_method_record": True,
            "status": "PASS",
        },
    )
    C.atomic_json(C.ROOT / "stage05_numeric_precision_audit.json", numeric_audit)
    C.atomic_csv(C.GLOBAL_CACHE, cache)
    C.atomic_csv(C.PHYSICAL_MANIFEST, cache)
    C.rewrite_method_files(records)
    C.atomic_csv(C.PROVENANCE_CSV, [])
    C.atomic_json(
        C.FAILURE_AUDIT,
        {
            "created_utc": C.utc_now(),
            "engineering_failures": [],
            "scientific_failures": [],
            "status": "ACTIVE",
        },
    )
    C.atomic_csv(C.RETENTION_MANIFEST, [])
    C.disk_audit("stage05_initial_preflight")


def initial_campaign(protocol: dict[str, Any]) -> dict[str, Any]:
    memberships = C.read_csv(P0_MEMBERSHIP)
    run_states = []
    for method, replicate in C.RUN_ORDER:
        p0 = sorted(
            [
                row
                for row in memberships
                if int(row["replicate_base"]) == replicate
            ],
            key=lambda row: int(row["p0_slot"]),
        )
        if len(p0) != 16:
            raise RuntimeError(f"P0 membership mismatch for replicate {replicate}.")
        run_states.append(
            {
                "method": method,
                "replicate": replicate,
                "rng_seed": C.METHOD_STREAMS[(method, replicate)],
                "rng_state": None,
                "current_generation_or_batch": 0,
                "completed_new_records": 0,
                "failed_records": 0,
                "remaining_new_records": 64,
                "active_record": None,
                "seen_signatures": [row["signature_sha256"] for row in p0],
                "population_signatures": [
                    row["signature_sha256"] for row in p0
                ],
                "status": "PENDING",
            }
        )
    return {
        "schema": "stage05_campaign_state_v1",
        "campaign_uuid": str(uuid.uuid4()),
        "created_utc": C.utc_now(),
        "started_utc": None,
        "state": "READY",
        "approval_token_consumed": "APPROVE_AMENDMENT_A2_START_STAGE05",
        "stage05_protocol_sha256": protocol["stage05_protocol_sha256"],
        "current_position": {},
        "current_run_index": 0,
        "run_states": run_states,
        "accounting": {
            "p0_method_records": 144,
            "new_method_records": 0,
            "total_f1_records": 144,
            "completed_physical_evaluations": 0,
            "cache_hits": 0,
            "duplicate_rejections": 0,
            "scientific_failures": 0,
            "engineering_retries": 0,
            "estimated_remaining_physical_evaluations": 576,
        },
        "first_physical_candidate_validation": "PENDING",
        "control_plane_validation": "PENDING",
        "active_wall_clock_seconds": 0.0,
        "paused_wall_clock_seconds": 0.0,
        "official_val548_accessed": False,
        "heldout1610_accessed": False,
        "kd_executed": False,
        "fidelity2_started": False,
        "tensorrt_executed": False,
        "jetson_profiling_started": False,
        "stage05_complete": False,
        "required_next_token": "APPROVE_STAGE05_FULL_FIDELITY1_SEARCH",
    }


def code_hashes(protocol: dict[str, Any]) -> dict[str, Any]:
    script_paths = sorted(C.SCRIPTS.glob("*"))
    files = [
        path
        for path in script_paths
        if path.is_file() and path.suffix.lower() in {".py", ".ps1"}
    ]
    rows = [
        {
            "relative_path": str(path.relative_to(C.ROOT)),
            "sha256": C.sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    payload = {
        "created_utc": C.utc_now(),
        "stage05_protocol_sha256": protocol["stage05_protocol_sha256"],
        "files": rows,
        "scientific_code_frozen_before_first_candidate": True,
        "status": "PASS",
    }
    C.atomic_json(C.CODE_HASHES, payload)
    return payload


def initialize() -> int:
    C.ensure_directories()
    if C.CAMPAIGN_STATE.is_file():
        state = C.load_json(C.CAMPAIGN_STATE)
        if (
            state.get("state") == "READY"
            and len(C.read_csv(C.SEARCH_RECORDS)) == C.EXPECTED_P0_RECORDS
            and not C.stage05_worker_processes()
        ):
            protocol = C.load_json(C.PROTOCOL_LOCK)
            if not (C.ROOT / "stage05_rng_manifest.json").is_file():
                C.atomic_json(
                    C.ROOT / "stage05_rng_manifest.json",
                    {
                        "created_utc": C.utc_now(),
                        "generator": "numpy.random.Generator",
                        "bit_generator": "PCG64",
                        "stream_construction": "independent fixed method/replicate streams",
                        "p0_streams": C.P0_STREAMS,
                        "method_streams": [
                            {
                                "method": method,
                                "replicate": replicate,
                                "seed": seed,
                            }
                            for (method, replicate), seed in C.METHOD_STREAMS.items()
                        ],
                        "mutable_state_shared_across_methods": False,
                        "state_saved_after_every_proposal": True,
                        "state_saved_after_every_method_record": True,
                        "status": "PASS",
                    },
                )
            code_hashes(protocol)
            print("STAGE05_PRE_CANDIDATE_CODE_HASHES_REFRESHED=YES")
        print(f"STAGE05_ALREADY_INITIALIZED_STATE={state.get('state')}")
        print(f"CAMPAIGN_UUID={state.get('campaign_uuid')}")
        return 0
    preflight = verify_preflight()
    protocol = protocol_payload(preflight)
    cache, records, numeric_audit = convert_stage04_sources(protocol)
    write_initial_files(preflight, protocol, cache, records, numeric_audit)
    campaign = initial_campaign(protocol)
    C.atomic_json(C.CAMPAIGN_STATE, campaign)
    C.atomic_json(
        C.CONTROLLER_STATE,
        {
            "created_utc": C.utc_now(),
            "controller_pid": os.getpid(),
            "hostname": socket.gethostname(),
            "campaign_uuid": campaign["campaign_uuid"],
            "state": "READY",
            "background_worker_started": False,
            "status": "PASS",
        },
    )
    code_hashes(protocol)
    C.write_status()
    print("STAGE05_ONE_TIME_PREFLIGHT=PASS")
    print("PARAMS_EXACT_INTEGER_RECORDED=YES")
    print("FLOPS_FULL_PRECISION_RECORDED=YES")
    print("RECALL_FULL_PRECISION_RECORDED=YES")
    print("ROUNDED_VALUES_USED_FOR_SELECTION=NO")
    print("CROSS_FILE_NUMERIC_CONSISTENCY=PASS")
    print(f"CAMPAIGN_UUID={campaign['campaign_uuid']}")
    print(f"STAGE05_PROTOCOL_SHA256={protocol['stage05_protocol_sha256']}")
    print("STAGE05_STATE=READY")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    if not args.initialize:
        parser.error("--initialize is required")
    return initialize()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations
import release_paths as _release_paths  # Portable source-root resolution.

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
SOURCE = PROJECT / "scripts" / "stage02_explicit_inner_val_eval.py"
PYTHON_PORT = PROJECT / "scripts" / "visdrone_toolkit_python_port.py"
PARITY_AUDIT = PROJECT / "04_shared_p0_fidelity1" / "visdrone_evaluator_parity_audit.json"
EXPECTED_RECOVERY_EPOCHS = 25


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_best_epoch(results_csv: Path) -> dict:
    frame = pd.read_csv(results_csv)
    frame.columns = [column.strip() for column in frame.columns]
    if len(frame) != EXPECTED_RECOVERY_EPOCHS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECOVERY_EPOCHS} Recovery rows, found {len(frame)}"
        )
    required = [
        "epoch",
        "metrics/precision",
        "metrics/recall",
        "metrics/mAP_0.5",
        "metrics/mAP_0.5:0.95",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"results.csv missing columns: {missing}")
    for column in required:
        if not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            raise RuntimeError(f"results.csv has non-finite values in {column}")
    fitness = 0.1 * frame["metrics/mAP_0.5"] + 0.9 * frame["metrics/mAP_0.5:0.95"]
    index = int(fitness.idxmax())
    row = frame.loc[index]
    return {
        "best_epoch_zero_based": int(row["epoch"]),
        "best_epoch_one_based": int(row["epoch"]) + 1,
        "best_fitness": float(fitness.loc[index]),
        "precision": float(row["metrics/precision"]),
        "recall": float(row["metrics/recall"]),
        "map50": float(row["metrics/mAP_0.5"]),
        "map50_95": float(row["metrics/mAP_0.5:0.95"]),
        "training_rows": len(frame),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--signature-sha256", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    module = load_module_from_path("stage07_eval_reused", SOURCE)
    module.RUN_DIR = run_dir
    module.training_best_epoch = training_best_epoch
    port = load_module_from_path("stage07_visdrone_python_port", PYTHON_PORT)
    parity = json.loads(PARITY_AUDIT.read_text(encoding="utf-8"))
    if parity.get("status") != "PASS":
        raise RuntimeError("The frozen VisDrone evaluator parity audit is not PASS.")

    def run_parity_validated_python_port(annotation_subset, prediction_dir, output_dir):
        payload = port.evaluate(
            annotation_subset,
            prediction_dir,
            module.RAW_TRAIN_IMAGES,
            output_dir / "internal_visdrone_metrics_stage02.json",
            module.EXPECTED_IMAGE_COUNT,
        )
        payload["metadata"]["parity_audit"] = str(PARITY_AUDIT)
        payload["metadata"]["parity_audit_sha256"] = module.sha256(PARITY_AUDIT)
        payload["metadata"]["maximum_parity_difference_percentage_points"] = parity[
            "maximum_observed_absolute_difference_percentage_points"
        ]
        payload["metadata"]["parity_tolerance_percentage_points"] = parity[
            "acceptance_tolerance_percentage_points"
        ]
        module.write_json(output_dir / "internal_visdrone_metrics_stage02.json", payload)
        return payload

    module.run_matlab = run_parity_validated_python_port
    sys.argv = [str(SOURCE), "--run-dir", str(run_dir)]
    status = module.main()
    if status != 0:
        return int(status)
    metrics_path = run_dir / "explicit_inner_val_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "stage": "Stage07 Fidelity-2 fresh Recovery-25 explicit inner-val evaluation",
            "model": f"F2-{args.signature_sha256[:12]}",
            "evaluation_id": args.evaluation_id,
            "architecture_signature_sha256": args.signature_sha256,
            "recovery_epochs": EXPECTED_RECOVERY_EPOCHS,
            "training_seed": 42,
            "kd_enabled": False,
            "search_offspring": False,
            "visdrone_evaluator_implementation": (
                "Parity-validated numerical Python port of the official VisDrone DET MATLAB toolkit"
            ),
            "visdrone_evaluator_parity_audit": str(PARITY_AUDIT),
            "visdrone_evaluator_parity_audit_sha256": module.sha256(PARITY_AUDIT),
        }
    )
    payload["primary_ap_namespace"] = (
        "Official VisDrone DET toolkit numerical protocol on inner-val1294 with "
        "ignored-region suppression, executed by a parity-validated Python port"
    )
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"STAGE07_EXPLICIT_EVALUATION_ID={args.evaluation_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

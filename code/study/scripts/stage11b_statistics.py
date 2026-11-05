from __future__ import annotations
import release_paths as _release_paths  # Portable source-root resolution.

import importlib.util
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(_release_paths.expand('${STUDY_ROOT}'))
OUT = ROOT / "11b_heldout1610_confirmation"
STAGE11 = ROOT / "11_official_val_statistics"
SOURCE = ROOT / "scripts" / "stage11_statistics.py"
N_IMAGES = 1610
ITERATIONS = 10000
SEED = 62042


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temp.replace(path)


def load_source():
    spec = importlib.util.spec_from_file_location("stage11b_frozen_statistics_engine", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen Stage11 statistics engine: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVAL = OUT
    module.OUT = OUT
    module.N_IMAGES = N_IMAGES
    module.ITERATIONS = ITERATIONS
    module.BATCH = 64 if module.DEVICE.type == "cuda" else 2
    return module


def direction(primary: float, secondary: float) -> str:
    if primary > 0 and secondary > 0:
        return "CONSISTENT_POSITIVE"
    if primary < 0 and secondary < 0:
        return "CONSISTENT_NEGATIVE"
    if primary * secondary < 0:
        return "DIRECTION_REVERSED"
    return "NEAR_ZERO"


def main() -> int:
    metrics_payload = read_json(OUT / "per_checkpoint_metrics.json")
    if metrics_payload.get("status") != "PASS" or len(metrics_payload["records"]) != 15:
        raise RuntimeError("Stage11B evaluation is not complete")
    lock = read_json(OUT / "heldout_execution_lock.json")
    heldout_manifest = read_json(OUT / "heldout1610_manifest.json")
    metrics = metrics_payload["records"]
    metric_by_key = {(item["model"], int(item["seed"])): item for item in metrics}
    stats = load_source()
    cache_dir = OUT / "resampling_caches"
    distribution_dir = OUT / "bootstrap_distributions"
    cache_dir.mkdir(exist_ok=True)
    distribution_dir.mkdir(exist_ok=True)
    ones = np.ones((1, N_IMAGES), dtype=np.float64)
    counts = stats.bootstrap_counts(SEED)
    np.savez_compressed(OUT / "heldout_bootstrap_rng_draws.npz", counts=counts, rng_seed=np.asarray(SEED), iterations=np.asarray(ITERATIONS), image_count=np.asarray(N_IMAGES))

    caches: dict[tuple[str, int], dict[str, Any]] = {}
    distributions: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    parity_records: list[dict[str, Any]] = []
    for metric in metrics:
        key = (metric["model"], int(metric["seed"]))
        cache_path = cache_dir / f"{stats.model_slug(metric['model'])}_seed{metric['seed']}.pkl"
        cache = stats.build_cache(metric, heldout_manifest, cache_path)
        point = stats.evaluate_distribution(cache, ones)
        differences = {name: float(point[name][0] - float(metric[name])) for name in ("AP", "APS", "Recall")}
        if any(abs(value) > 2e-8 for value in differences.values()):
            raise RuntimeError(f"Heldout prediction parity failed for {key}: {differences}")
        caches[key] = cache
        distributions[key] = stats.evaluate_distribution(cache, counts)
        distribution_path = distribution_dir / f"{stats.model_slug(metric['model'])}_seed{metric['seed']}.npz"
        np.savez_compressed(distribution_path, **distributions[key])
        parity_records.append({"model": key[0], "seed": key[1], "cache_path": str(cache_path), "cache_sha256": stats.sha256(cache_path), "point_differences": differences, "status": "PASS"})
    write_json(OUT / "prediction_parity_audit.json", {"schema": "stage11b_prediction_parity_audit_v1", "records": parity_records, "tolerance": 2e-8, "status": "PASS", "second_capture_pass_executed": False})

    paired_records: list[dict[str, Any]] = []
    paired_summaries: list[dict[str, Any]] = []
    bootstrap_records: list[dict[str, Any]] = []
    for comparison in lock["comparisons"]:
        cid, model_a, model_b = comparison["comparison_id"], comparison["model_a"], comparison["model_b"]
        for metric_name in ("AP", "APS", "Recall"):
            differences: list[float] = []
            for seed in (42, 43, 44):
                value_a = float(metric_by_key[(model_a, seed)][metric_name])
                value_b = float(metric_by_key[(model_b, seed)][metric_name])
                delta = value_a - value_b
                differences.append(delta)
                paired_records.append({"comparison_id": cid, "model_a": model_a, "model_b": model_b, "metric": metric_name, "seed": seed, "model_a_value": value_a, "model_b_value": value_b, "paired_difference": delta})
            paired_summaries.append({
                "comparison_id": cid, "model_a": model_a, "model_b": model_b, "metric": metric_name,
                "seed42_difference": differences[0], "seed43_difference": differences[1], "seed44_difference": differences[2],
                "mean_paired_difference": statistics.mean(differences), "sample_sd": statistics.stdev(differences),
                "minimum": min(differences), "maximum": max(differences),
                "sign_consistency": all(value > 0 for value in differences) or all(value < 0 for value in differences) or all(value == 0 for value in differences),
            })
            distribution = np.mean(np.stack([
                distributions[(model_a, seed)][metric_name] - distributions[(model_b, seed)][metric_name]
                for seed in (42, 43, 44)
            ]), axis=0)
            low, high = stats.percentile_ci(distribution)
            dist_path = distribution_dir / f"{cid.lower()}_{metric_name.lower()}_mean_paired_delta.npz"
            np.savez_compressed(dist_path, distribution=distribution, rng_seed=np.asarray(SEED), iterations=np.asarray(ITERATIONS))
            bootstrap_records.append({
                "comparison_id": cid, "model_a": model_a, "model_b": model_b, "metric": metric_name,
                "mean_paired_point_estimate": statistics.mean(differences), "ci95_low": low, "ci95_high": high,
                "iterations": ITERATIONS, "rng_seed": SEED,
                "interpretation": "Image-sampling uncertainty conditional on the three fixed checkpoints",
                "p_value_generated": False,
            })
    stats.write_csv(OUT / "paired_seed_differences.csv", paired_records + paired_summaries)
    write_json(OUT / "paired_seed_differences.json", {"per_seed_records": paired_records, "summaries": paired_summaries, "formal_test": False, "status": "PASS"})
    stats.write_csv(OUT / "paired_bootstrap_results.csv", bootstrap_records)
    write_json(OUT / "paired_bootstrap_results.json", {"schema": "stage11b_paired_bootstrap_results_v1", "records": bootstrap_records, "status": "PASS", "randomization_executed": False, "p_values_generated": False, "holm_executed": False})

    stage11_summary = read_json(STAGE11 / "01_val548_evaluation" / "method_seed_summary.json")["records"]
    heldout_summary = read_json(OUT / "method_seed_summary.json")["records"]
    primary_by_model = {item["model"]: item for item in stage11_summary}
    heldout_by_model = {item["model"]: item for item in heldout_summary}
    cross_records: list[dict[str, Any]] = []
    for model in sorted(heldout_by_model):
        cross_records.append({
            "record_type": "METHOD", "model": model,
            "val548_mean_AP": primary_by_model[model]["mean_AP"], "heldout1610_mean_AP": heldout_by_model[model]["mean_AP"],
            "val548_mean_APS": primary_by_model[model]["mean_APS"], "heldout1610_mean_APS": heldout_by_model[model]["mean_APS"],
            "val548_mean_Recall": primary_by_model[model]["mean_Recall"], "heldout1610_mean_Recall": heldout_by_model[model]["mean_Recall"],
        })
    primary_metrics = read_json(STAGE11 / "01_val548_evaluation" / "per_checkpoint_metrics.json")["records"]
    primary_metric_by_key = {(item["model"], int(item["seed"])): item for item in primary_metrics}
    comparison_records: list[dict[str, Any]] = []
    for comparison in lock["comparisons"]:
        cid, model_a, model_b = comparison["comparison_id"], comparison["model_a"], comparison["model_b"]
        item: dict[str, Any] = {"record_type": "COMPARISON", "comparison_id": cid, "model_a": model_a, "model_b": model_b}
        for metric_name in ("AP", "APS", "Recall"):
            primary_delta = statistics.mean([float(primary_metric_by_key[(model_a, seed)][metric_name]) - float(primary_metric_by_key[(model_b, seed)][metric_name]) for seed in (42, 43, 44)])
            heldout_delta = statistics.mean([float(metric_by_key[(model_a, seed)][metric_name]) - float(metric_by_key[(model_b, seed)][metric_name]) for seed in (42, 43, 44)])
            item[f"val548_mean_delta_{metric_name}"] = primary_delta
            item[f"heldout1610_mean_delta_{metric_name}"] = heldout_delta
            item[f"direction_{metric_name}"] = direction(primary_delta, heldout_delta)
        comparison_records.append(item)
    stats.write_csv(OUT / "cross_split_consistency.csv", cross_records + comparison_records)
    write_json(OUT / "cross_split_consistency.json", {"method_records": cross_records, "comparison_records": comparison_records, "primary_authority": "val548", "secondary_role": "CONFIRMATION_ONLY", "status": "PASS"})
    write_json(OUT / "statistical_validation.json", {
        "schema": "stage11b_statistical_validation_v1", "created_utc": utc_now(), "status": "PASS",
        "checks": {"prediction_parity_verified": True, "paired_seeds_preserved": True, "paired_bootstrap_10000": True,
                   "bootstrap_rng_62042": True, "same_image_indices_within_pairs_and_across_seeds": True,
                   "randomization_executed": False, "heldout_pvalues_generated": False, "holm_executed": False},
        "device": str(stats.DEVICE), "numeric_dtype": str(stats.DTYPE),
    })
    print(json.dumps({"status": "STATISTICS_PASS", "bootstrap_iterations": ITERATIONS, "comparisons": 4, "formal_tests": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

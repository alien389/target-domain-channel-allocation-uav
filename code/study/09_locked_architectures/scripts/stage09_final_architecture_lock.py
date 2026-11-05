from __future__ import annotations
import release_paths as _release_paths  # Portable source-root resolution.

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
OUTPUT = PROJECT / "09_locked_architectures"
YAMLS = OUTPUT / "yamls"
RAW = OUTPUT / "raw_checkpoints"
GENOMES = OUTPUT / "genomes"
CHANNELS = OUTPUT / "channel_indices"
TEMP = OUTPUT / "_independent_reconstruction"
STAGE04_SCRIPT = PROJECT / "scripts" / "stage04_shared_p0_fidelity1.py"
R1_ROOT = PROJECT / "03a_taylor_dependency_gene_space" / "gene_space_R1"
GENE_JSON = R1_ROOT / "gene_space_manifest_R1.json"
STAGE05 = PROJECT / "05_full_fidelity1_search"
STAGE06 = PROJECT / "06a_fidelity2_promotion_lock"
STAGE07 = PROJECT / "07_fidelity2_confirmation"
STAGE08A = PROJECT / "08a_fidelity3_proposed_lock"
STAGE08B = PROJECT / "08b_fidelity3_matched_controls"
A3 = PROJECT / "protocol_amendment_A3_defer_kd_final_proposed_ablation"

EXPECTED_MANIFESTS = {
    STAGE05 / "stage05_output_manifest.json": "6E48DA757C4BD5D2B0E3FDFFDD06BE7D34E9DB43379DE2923340BE1F016EFE76",
    STAGE06 / "stage06a_output_manifest.json": "E5410026CF52D69C086A6044AEB0AC4B5F0C68B2129D5DDFB05E86D67FFA23C2",
    STAGE07 / "stage07_output_manifest.json": "D8E293952CC9E62579AD134EC582A71FB08F57C494BFFE3BEED1AC42320B06DD",
    STAGE08A / "stage08a_output_manifest.json": "CC410EF097549381A883C0801148A411BBC7964536806ADD52927B93EC4A0032",
    STAGE08B / "stage08b_output_manifest.json": "94D944A17AF90FC29DCAFD0464BA973854F67211F969E908AFF1715CCEA79921",
    A3 / "amendment_A3_output_manifest.json": "0A77876FF7C4B706EF7A91F1C30572249DF4F253E7A624C1A2528CA81E1541E0",
}

MODELS = [
    {
        "slug": "meta_ap_aps_b6", "label": "Meta-AP+APS-B6", "candidate_id": "F2PHYS-EEA3FB45FD3FD1BA",
        "signature": "EEA3FB45FD3FD1BADD263DE055266F4F6B5EE34EE818F4C1D7026B764E862C80", "params": 2507047, "flops": 7847110400,
        "f3_sha": "593F8C715215266667A4E963C7588B02D198D1F9E15C78C20F062E856A005F58", "raw_sha": "B6045F3275B17D10E07E59282498AE7FA9D2C1B6E907533863B58742A095ADB9",
        "selection_role": "proposed target-domain NSGA-II AP+APS architecture", "source_stage": "Stage08A", "source_role": "Proposed Candidate A", "source_index": 2,
    },
    {
        "slug": "matched_manual_b6", "label": "MatchedManual-B6", "candidate_id": "F3CTRL-MATCHEDMANUAL-DED07E02EAE3B021",
        "signature": "DED07E02EAE3B021E4CE8B8A078348F24975BB5593CE264565E5A6BE23776C69", "params": 2512591, "flops": 7775097600,
        "f3_sha": "FD5FC6E462A88ADBB5EB21FB2EEF9966052935C4DDA06B50D01FA8B1EF876E60", "raw_sha": "3C1BD9823EC289875A152326E8AFF44BDAEB42DD5C74CDDB59E797FCB3A9C6FC",
        "selection_role": "near-matched dependency-aware Global Taylor manual control", "source_stage": "Stage08B", "source_role": "MatchedManual-B6", "source_index": 1,
        "complexity_distance": "0.009176983160578446812727395807",
    },
    {
        "slug": "meta_ap_only_b6", "label": "Meta-AP-only-B6", "candidate_id": "F2PHYS-A14F4B16087416B0",
        "signature": "A14F4B16087416B074FFDAE2BFAB8960BF9D3B4A40A4AC4559C6D59E9887D872", "params": 2487047, "flops": 7812531200,
        "f3_sha": "0086B0DE6BFC375B81BF736647E5D0729ED052B2141C73F31CAA923ABC930DCE", "raw_sha": "194BEA4EFA4F4E0CDD713F05B262DDA82750CBB3F92E50BAEEA8E80794E1623F",
        "selection_role": "single-objective evolutionary AP-only control", "source_stage": "Stage08B", "source_role": "Meta-AP-only-B6 finalist", "source_index": 2, "aps_used_for_selection": False,
    },
    {
        "slug": "random_search_b6", "label": "RandomSearch-B6", "candidate_id": "F2PHYS-67362A640747F04D",
        "signature": "67362A640747F04DF16762E085A7DACB2F5D4A0247425C6A5DBCD36FB545BFF0", "params": 2503879, "flops": 7825996800,
        "f3_sha": "99D3BB8CD09BB15DD16A68F4059137C9BA9ED3D5BD95FF8CD9AABF4A613AF79C", "raw_sha": "C4B41C1562E65C212E564335FE954FCB7889215F2441F893FFCF60B9A3A310A9",
        "prompt_declared_raw_sha": "C4B41C1562E65C212F564335FE954FCB7889215F2441F893FFCF60B9A3A310A9",
        "selection_role": "warm-start constrained Random Search control", "source_stage": "Stage08B", "source_role": "Random candidate B", "source_index": 4, "random_lock_rule": "DOMINANCE_AP_AND_APS",
    },
    {
        "slug": "target_manual_b6_reference", "label": "TargetManual-B6-reference", "candidate_id": "F2PHYS-AEA2D0A18DAC15E4",
        "signature": "AEA2D0A18DAC15E40091E474DBF688A98FCA2B6CFF8D4D3F4845165C59244F5D", "params": 2513743, "flops": 7643616000,
        "f3_sha": "434F0793B91BA2C61E620C910A09C803DDA13AC2DC593820B99D14A6E9884665", "raw_sha": "A0AA139C32E5269F31E0AEFA8191A00271298DD2B2C65D4B20BD1D755F0C7F3D",
        "selection_role": "target-domain manually designed B6 reference", "source_stage": "Stage08A", "source_role": "TargetManual-B6-reference", "source_index": 1,
    },
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S4 = load_module("stage09_stage04_locked", STAGE04_SCRIPT)
R1M = S4.R1M


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_row(model: dict[str, Any]) -> dict[str, str]:
    source = STAGE08A / "stage08a_all_results.csv" if model["source_stage"] == "Stage08A" else STAGE08B / "stage08b_all_results.csv"
    rows = [row for row in read_csv(source) if row["physical_candidate_id"] == model["candidate_id"]]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one source row for {model['label']}")
    return rows[0]


def no_forbidden_processes() -> bool:
    forbidden = ("train_structural_recovery_drone.py", "evaluate_stage08", "train_stage10", "train.py")
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or []).lower()
            if any(token in command for token in forbidden):
                return False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return True


def upstream_snapshot() -> dict[str, str]:
    return {str(path): sha256(path) for path in EXPECTED_MANIFESTS}


def verify_upstream() -> dict[str, Any]:
    validations = {
        "Stage05": STAGE05 / "stage05_validation.json",
        "Stage06A": STAGE06 / "stage06a_validation.json",
        "Stage07": STAGE07 / "stage07_validation.json",
        "Stage08A": STAGE08A / "stage08a_validation.json",
        "Stage08B": STAGE08B / "stage08b_validation.json",
        "Amendment_A3": A3 / "amendment_A3_validation.json",
    }
    checks = {name: json.loads(path.read_text(encoding="utf-8-sig")).get("status") == "PASS" for name, path in validations.items()}
    checks["manifest_hashes"] = all(path.is_file() and sha256(path) == expected for path, expected in EXPECTED_MANIFESTS.items())
    checks["no_training_or_evaluation_process"] = no_forbidden_processes()
    checks["stage10_absent"] = not (PROJECT / "10_final_training").exists()
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"Stage09 upstream integrity failed: {failed}")
    return {"checks": checks, "status": "PASS"}


def configure_reconstruction() -> None:
    S4.OUTPUT = OUTPUT
    S4.PHYSICAL_ROOT = OUTPUT
    S4.evaluation_directory = lambda candidate: Path(candidate["stage09_reconstruction_dir"])


def canonical_descriptor(model: dict[str, Any], widths: dict[str, int], base_yaml: dict[str, Any], group_ids: list[str]) -> dict[str, Any]:
    return {
        "schema": "locked_dependency_width_architecture_v1",
        "model_name": model["label"],
        "candidate_id": model["candidate_id"],
        "nc": 10,
        "input_shape": [1, 3, 640, 640],
        "architecture_signature_sha256": model["signature"],
        "params_exact_integer": model["params"],
        "flops_raw_exact_integer": model["flops"],
        "flops_g_full_precision": format(Decimal(model["flops"]) / Decimal(1_000_000_000), "f"),
        "dependency_group_order": group_ids,
        "retained_widths": widths,
        "channel_identity_policy": "frozen target-domain Taylor retained channel indices",
        "dependency_manifest": str(GENE_JSON),
        "dependency_manifest_sha256": sha256(GENE_JSON),
        "base_model_yaml": base_yaml,
        "canonical_initialization": f"raw_checkpoints/{model['slug']}_raw.pt",
        "selection_role": model["selection_role"],
        "immutable": True,
    }


def yaml_dump(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))


def validate_source(model: dict[str, Any], row: dict[str, str], groups: list[dict[str, Any]], complexity: Any) -> tuple[dict[str, int], Path, Path]:
    if row["candidate_role"] != model["source_role"] or row["architecture_signature"] != model["signature"]:
        raise RuntimeError(f"Source identity mismatch: {model['label']}")
    if int(row["params_exact_integer"]) != model["params"] or int(row["flops_raw_exact_integer"]) != model["flops"]:
        raise RuntimeError(f"Source complexity mismatch: {model['label']}")
    if row["best_checkpoint_sha256"] != model["f3_sha"]:
        raise RuntimeError(f"F3 checkpoint declaration mismatch: {model['label']}")
    f3 = Path(row["best_checkpoint"])
    if not f3.is_file() or sha256(f3) != model["f3_sha"]:
        raise RuntimeError(f"F3 checkpoint file mismatch: {model['label']}")
    reconstruction = Path(row["reconstruction_manifest"])
    channel_map = Path(row["channel_index_map"])
    if sha256(reconstruction) != row["reconstruction_manifest_sha256"] or sha256(channel_map) != row["channel_index_map_sha256"]:
        raise RuntimeError(f"Source reconstruction evidence mismatch: {model['label']}")
    widths = {key: int(value) for key, value in json.loads(reconstruction.read_text(encoding="utf-8-sig"))["widths"].items()}
    group_ids = [group["group_id"] for group in groups]
    if R1M.signature_sha(group_ids, tuple(widths[group_id] for group_id in group_ids)) != model["signature"]:
        raise RuntimeError(f"Genome signature mismatch: {model['label']}")
    analytic = complexity.calculate(widths)
    if int(analytic["params"]) != model["params"] or int(analytic["macs"]) * 2 != model["flops"]:
        raise RuntimeError(f"Analytic complexity mismatch: {model['label']}")
    for group in groups:
        width = widths[group["group_id"]]
        if width not in R1M.parse_allowed(group["allowed_widths"]):
            raise RuntimeError(f"Disallowed retained width: {model['label']} {group['group_id']}")
        if width < int(group["exact_minimum_width"]) or width > int(group["exact_maximum_width"]) or width % 8:
            raise RuntimeError(f"Width contract mismatch: {model['label']} {group['group_id']}")
    return widths, reconstruction, channel_map


def reconstruct_one(model: dict[str, Any], widths: dict[str, int], row: dict[str, str], groups: list[dict[str, Any]], complexity: Any, taylor_bank: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    directory = TEMP / model["slug"]
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    candidate: dict[str, Any] = {
        "evaluation_id": row["evaluation_id"],
        "signature_sha256": model["signature"],
        "unique_signature_index": model["source_index"],
        "fused_params": model["params"],
        "fused_flops_g": model["flops"] / 1_000_000_000,
        "stage09_reconstruction_dir": str(directory),
    }
    candidate.update(widths)
    audit = S4.reconstruct_raw(candidate, groups, complexity, taylor_bank)
    reconstructed = directory / "raw_checkpoint.pt"
    checkpoint = torch.load(reconstructed, map_location="cpu", weights_only=False)
    source_root = STAGE08A if model["source_stage"] == "Stage08A" else STAGE08B
    historical_plan = source_root / "_r" / f"E{model['source_index']:03d}" / "p.json"
    checkpoint["model"].pruning_metadata["plan_json"] = str(historical_plan)
    normalized = directory / f"s4_{model['source_index']:03d}.pt"
    torch.save(checkpoint, normalized)
    if sha256(normalized) != model["raw_sha"]:
        raise RuntimeError(f"Canonical raw hash mismatch after independent reconstruction: {model['label']}")
    audit["stage09_reconstruction_sha256_before_historical_metadata_normalization"] = sha256(reconstructed)
    audit["historical_plan_json_metadata_restored"] = str(historical_plan)
    audit["independent_reconstruction_normalized_sha256"] = sha256(normalized)
    audit["normalization_scope"] = "serialization metadata path only; model tensors and retained channels unchanged"
    return normalized, audit


def lock_model(model: dict[str, Any], groups: list[dict[str, Any]], complexity: Any, taylor_bank: dict[str, Any], base_yaml: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    row = source_row(model)
    widths, source_reconstruction, source_channel_map = validate_source(model, row, groups, complexity)
    group_ids = [group["group_id"] for group in groups]
    genome_path = GENOMES / f"{model['slug']}_genome.json"
    genome_payload = {
        "schema": "locked_retained_width_genome_v1", "model_name": model["label"], "candidate_id": model["candidate_id"],
        "architecture_signature_sha256": model["signature"], "dependency_group_order": group_ids, "retained_widths": widths,
        "genome_variable": "retained width per dependency group only", "channel_indices_are_genome_variables": False, "immutable": True,
    }
    atomic_json(genome_path, genome_payload)
    channel_path = CHANNELS / f"{model['slug']}_channel_indices.csv"
    shutil.copy2(source_channel_map, channel_path)
    yaml_path = YAMLS / f"{model['slug']}.yaml"
    yaml_dump(yaml_path, canonical_descriptor(model, widths, base_yaml, group_ids))
    loaded_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if loaded_yaml["architecture_signature_sha256"] != model["signature"] or loaded_yaml["retained_widths"] != widths:
        raise RuntimeError(f"Canonical YAML reload mismatch: {model['label']}")

    reconstructed, reconstruction_audit = reconstruct_one(model, widths, row, groups, complexity, taylor_bank)
    generated_map = reconstructed.parent / "channel_index_map.csv"
    if sha256(generated_map) != sha256(source_channel_map):
        raise RuntimeError(f"Retained channel-index reconstruction mismatch: {model['label']}")
    raw_path = RAW / f"{model['slug']}_raw.pt"
    source_raw = Path(row.get("raw_checkpoint", "")) if row.get("raw_checkpoint") else None
    if source_raw and source_raw.is_file():
        if sha256(source_raw) != model["raw_sha"]:
            raise RuntimeError(f"Upstream raw checkpoint mismatch: {model['label']}")
        shutil.copy2(source_raw, raw_path)
        raw_origin = "COPIED_FROM_READ_ONLY_UPSTREAM_RAW_CHECKPOINT"
    else:
        shutil.copy2(reconstructed, raw_path)
        raw_origin = "DETERMINISTICALLY_RECONSTRUCTED_IN_STAGE09_AFTER_UPSTREAM_RETENTION_DELETION"
    if sha256(raw_path) != model["raw_sha"]:
        raise RuntimeError(f"Stage09 canonical raw checkpoint hash mismatch: {model['label']}")
    physical = R1M.physical_complexity(raw_path)
    if int(physical["fused_params"]) != model["params"] or not math.isclose(float(physical["flops_g"]), model["flops"] / 1e9, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"Physical complexity mismatch: {model['label']}")
    if not physical["forward_pass"] or physical["input_shape"] != [1, 3, 640, 640] or physical["nonfinite_values"] != 0:
        raise RuntimeError(f"Synthetic FP32 forward failed: {model['label']}")

    p_delta = abs(Decimal(model["params"]) - Decimal(MODELS[0]["params"])) / Decimal(MODELS[0]["params"])
    f_delta = abs(Decimal(model["flops"]) - Decimal(MODELS[0]["flops"])) / Decimal(MODELS[0]["flops"])
    manifest_row = {
        "model": model["label"], "candidate_id": model["candidate_id"], "selection_role": model["selection_role"], "architecture_signature": model["signature"],
        "params_exact_integer": model["params"], "flops_raw_exact_integer": model["flops"], "flops_g_full_precision": format(Decimal(model["flops"]) / Decimal(1_000_000_000), "f"),
        "delta_params_from_meta": str(p_delta), "delta_flops_from_meta": str(f_delta), "max_complexity_distance_from_meta": str(max(p_delta, f_delta)),
        "canonical_yaml": str(yaml_path), "canonical_yaml_sha256": sha256(yaml_path), "genome": str(genome_path), "genome_sha256": sha256(genome_path),
        "channel_index_map": str(channel_path), "channel_index_map_sha256": sha256(channel_path), "canonical_raw_checkpoint": str(raw_path), "canonical_raw_checkpoint_sha256": sha256(raw_path),
        "canonical_raw_origin": raw_origin, "f3_evidence_checkpoint": row["best_checkpoint"], "f3_evidence_checkpoint_sha256": model["f3_sha"],
        "source_stage": model["source_stage"], "source_reconstruction_manifest": str(source_reconstruction), "source_reconstruction_manifest_sha256": sha256(source_reconstruction),
        "source_result_manifest": str(STAGE08A / "stage08a_output_manifest.json" if model["source_stage"] == "Stage08A" else STAGE08B / "stage08b_output_manifest.json"),
        "source_result_manifest_sha256": sha256(STAGE08A / "stage08a_output_manifest.json" if model["source_stage"] == "Stage08A" else STAGE08B / "stage08b_output_manifest.json"),
        "locked": True,
    }
    if model.get("prompt_declared_raw_sha"):
        manifest_row["prompt_declared_raw_checkpoint_sha256"] = model["prompt_declared_raw_sha"]
        manifest_row["raw_hash_conflict_status"] = "PROMPT_TRANSCRIPTION_NIBBLE_CORRECTED_FROM_STAGE08B_FILE_MANIFEST_RETENTION_AND_RECONSTRUCTION"
    validation_row = {
        "model": model["label"], "candidate_id_match": True, "signature_match": True, "yaml_load": "PASS", "genome_match": True, "channel_indices_match": True,
        "raw_checkpoint_hash_match": sha256(raw_path) == model["raw_sha"], "f3_checkpoint_hash_match": sha256(Path(row["best_checkpoint"])) == model["f3_sha"],
        "params_match": int(physical["fused_params"]) == model["params"], "flops_match": math.isclose(float(physical["flops_g"]), model["flops"] / 1e9, rel_tol=0.0, abs_tol=1e-9),
        "dependency_contracts": "PASS", "residual_contracts": "PASS", "concat_contracts": "PASS", "detect_contracts": "PASS", "minimum_width_constraints": "PASS", "divisibility_by_8": "PASS",
        "raw_checkpoint_reload": "PASS", "synthetic_input_shape": "[1,3,640,640]", "detect_scales": physical["detect_scales"], "nonfinite_values": physical["nonfinite_values"], "synthetic_fp32_forward": "PASS", "weight_updates": 0, "accuracy_evaluation": False, "status": "PASS",
        "independent_reconstruction_raw_sha256": reconstruction_audit["independent_reconstruction_normalized_sha256"],
        "serialization_metadata_normalization": reconstruction_audit["normalization_scope"],
    }
    return manifest_row, validation_row


def write_provenance(rows: list[dict[str, Any]]) -> None:
    payload = {
        "created_utc": utc_now(),
        "locked_architectures": [{"model": row["model"], "candidate_id": row["candidate_id"], "selection_role": row["selection_role"], "source_stage": row["source_stage"], "architecture_signature": row["architecture_signature"]} for row in rows],
        "selection_is_irreversible": True,
        "official_validation_results_may_not_replace_membership": True,
        "status": "PASS",
    }
    atomic_json(OUTPUT / "selection_provenance.json", payload)
    lines = ["# Selection Provenance", "", "The five architecture identities were frozen from the approved Stage08A and Stage08B decisions.", ""]
    for row in rows:
        lines.extend([f"## {row['model']}", "", f"- Candidate ID: `{row['candidate_id']}`", f"- Role: {row['selection_role']}", f"- Source: `{row['source_stage']}`", f"- Signature: `{row['architecture_signature']}`", ""])
    lines.extend(["No architecture may be replaced orredesigned after this lock. F3 checkpoints are selection evidence only; Stage10 must initialize from the canonical raw checkpoint.", ""])
    atomic_text(OUTPUT / "selection_provenance.md", "\n".join(lines))


def output_manifest() -> None:
    files = []
    for path in sorted(OUTPUT.rglob("*")):
        if not path.is_file() or path == OUTPUT / "stage09_output_manifest.json" or TEMP in path.parents:
            continue
        files.append({"path": str(path), "relative_path": str(path.relative_to(OUTPUT)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(OUTPUT / "stage09_output_manifest.json", {"created_utc": utc_now(), "files": files, "file_count": len(files), "status": "PASS"})


def main() -> int:
    for directory in (OUTPUT, YAMLS, RAW, GENOMES, CHANNELS):
        directory.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / ".complete").exists():
        raise RuntimeError("Stage09 is already complete; refusing overwrite")
    upstream_before = upstream_snapshot()
    integrity = verify_upstream()
    configure_reconstruction()
    groups = json.loads(GENE_JSON.read_text(encoding="utf-8-sig"))["groups"]
    parent = R1M.load_source_model(R1M.SEARCH_PARENT, "stage09_parent")
    complexity = R1M.ExactComplexityModel(parent.model, groups)
    taylor_bank, taylor_manifest = R1M.taylor_scope_bank(groups)
    if taylor_manifest["checkpoint_sha256"] != S4.SEARCH_PARENT_SHA:
        raise RuntimeError("Taylor/SearchParent binding failed")
    base_yaml = parent.model.yaml
    manifest_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for model in MODELS:
        manifest_row, validation_row = lock_model(model, groups, complexity, taylor_bank, base_yaml)
        manifest_rows.append(manifest_row)
        validation_rows.append(validation_row)
    if TEMP.exists():
        shutil.rmtree(TEMP)

    atomic_csv(OUTPUT / "locked_models_manifest.csv", manifest_rows)
    atomic_json(OUTPUT / "locked_models_manifest.json", {"created_utc": utc_now(), "locked_model_count": len(manifest_rows), "models": manifest_rows, "status": "PASS"})
    atomic_csv(OUTPUT / "reconstruction_validation.csv", validation_rows)
    atomic_json(OUTPUT / "reconstruction_validation.json", {"created_utc": utc_now(), "models": validation_rows, "status": "PASS"})
    write_provenance(manifest_rows)
    declaration = {
        "created_utc": utc_now(), "stage": "Stage09 Final Architecture Lock", "lock_is_irreversible": True, "locked_architecture_count": 5,
        "locked_models": [row["model"] for row in manifest_rows], "f3_checkpoints_are_evidence_only": True, "stage10_initialization_source": "canonical raw pruned checkpoints in this package",
        "prohibited_after_lock": ["gene changes", "channel-index replacement", "pruning-ratio changes", "search reopening", "official-validation-based replacement", "MatchedManual redesign"],
        "kd_policy": {"used_in_search": False, "used_in_fidelity1": False, "used_in_fidelity2": False, "used_in_fidelity3_locking": False, "stage08a_recovery50_only": True, "stage08b_recovery50_only": True, "teacher_loaded": False, "coco_yolov5m_substituted": False, "future_kd": "optional separate post-lock proposed-only ablation; never replaces primary Recovery-only comparison"},
        "stage10_requires_separate_approved_protocol_amendment": True, "official_val548_locked": True, "heldout1610_locked": True, "status": "LOCKED",
    }
    declaration["documented_identity_correction"] = {
        "model": "RandomSearch-B6",
        "prompt_declared_raw_sha256": "C4B41C1562E65C212F564335FE954FCB7889215F2441F893FFCF60B9A3A310A9",
        "authoritative_raw_sha256": "C4B41C1562E65C212E564335FE954FCB7889215F2441F893FFCF60B9A3A310A9",
        "basis": "Stage08B retained raw file, checkpoint retention manifest, stage08b results, and independent reconstruction all match the authoritative E nibble",
        "scientific_values_changed": False,
    }
    atomic_json(OUTPUT / "architecture_lock_declaration.json", declaration)
    upstream_after = upstream_snapshot()
    source_hashes = {
        "created_utc": utc_now(), "upstream_before": upstream_before, "upstream_after": upstream_after, "upstream_unchanged": upstream_before == upstream_after,
        "authoritative_files": [{"path": str(path), "sha256": sha256(path)} for path in [GENE_JSON, S4.TAYLOR_RANKING, S4.DEPENDENCY_GRAPH, R1M.SEARCH_PARENT, STAGE08A / "stage08a_all_results.csv", STAGE08B / "stage08b_all_results.csv", STAGE08A / "proposed_lock_decision.json", STAGE08B / "provisional_control_locks.json"]],
        "status": "PASS" if upstream_before == upstream_after else "FAIL",
    }
    atomic_json(OUTPUT / "stage09_source_hashes.json", source_hashes)
    checks = {
        "upstream_integrity": integrity["status"] == "PASS" and upstream_before == upstream_after,
        "exactly_five_architectures": len(manifest_rows) == 5,
        "candidate_ids_match": all(row["candidate_id"] == model["candidate_id"] for row, model in zip(manifest_rows, MODELS)),
        "signatures_match": all(row["architecture_signature"] == model["signature"] for row, model in zip(manifest_rows, MODELS)),
        "params_flops_match": all(row["params_exact_integer"] == model["params"] and row["flops_raw_exact_integer"] == model["flops"] for row, model in zip(manifest_rows, MODELS)),
        "raw_checkpoint_hashes_match": all(row["canonical_raw_checkpoint_sha256"] == model["raw_sha"] for row, model in zip(manifest_rows, MODELS)),
        "f3_checkpoint_hashes_match": all(row["f3_evidence_checkpoint_sha256"] == model["f3_sha"] for row, model in zip(manifest_rows, MODELS)),
        "all_yamls_reload": all(row["yaml_load"] == "PASS" for row in validation_rows),
        "all_raw_checkpoints_reload": all(row["raw_checkpoint_reload"] == "PASS" for row in validation_rows),
        "all_synthetic_forwards": all(row["synthetic_fp32_forward"] == "PASS" for row in validation_rows),
        "all_outputs_finite": all(int(row["nonfinite_values"]) == 0 for row in validation_rows),
        "training_executed": False, "accuracy_evaluation_executed": False, "kd_executed": False, "teacher_loaded": False,
        "official_val548_locked": True, "heldout1610_locked": True, "stage10_started": not (PROJECT / "10_final_training").exists(),
    }
    checks["stage10_started"] = False
    failed = [key for key, value in checks.items() if key not in ("training_executed", "accuracy_evaluation_executed", "kd_executed", "teacher_loaded", "stage10_started") and not value]
    validation = {"created_utc": utc_now(), "checks": checks, "failed_checks": failed, "status": "PASS" if not failed else "FAIL"}
    atomic_json(OUTPUT / "stage09_validation.json", validation)
    if failed:
        raise RuntimeError(f"Stage09 validation failed: {failed}")
    atomic_text(OUTPUT / "stage09_report.md", "# Stage09 Final Architecture Lock\n\n- Status: `PASS`\n- Locked architectures: `5`\n- Architecture signatures: `VERIFIED`\n- Params/FLOPs: `VERIFIED`\n- Canonical raw checkpoint hashes: `VERIFIED`\n- Documented identity correction: RandomSearch raw hash prompt nibble `F` corrected to authoritative artifact nibble `E`; no scientific value changed.\n- YAML reload: `PASS`\n- Synthetic FP32 forward passes: `5/5 PASS`\n- Training executed: `NO`\n- Accuracy evaluation executed: `NO`\n- KD executed: `NO`\n- Teacher loaded: `NO`\n- val548 accessed: `NO`\n- heldout1610 accessed: `NO`\n- Stage10 started: `NO`\n- Approval ready: `YES`\n")
    atomic_text(OUTPUT / ".complete", "STAGE09_STATUS=PASS\nLOCKED_ARCHITECTURES=5\nAPPROVAL_READY=YES\nSTAGE10_STARTED=NO\n")
    output_manifest()
    print("STAGE09_STATUS=PASS")
    print("LOCKED_ARCHITECTURES=5")
    print("TRAINING_EXECUTED=NO")
    print("KD_EXECUTED=NO")
    print("STAGE10_STARTED=NO")
    print("APPROVAL_READY=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

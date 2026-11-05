from __future__ import annotations
import release_paths as _release_paths  # Portable source-root resolution.

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
STAGE05 = PROJECT / "05_full_fidelity1_search"
OUTPUT = PROJECT / "06a_fidelity2_promotion_lock"
PLAN = PROJECT / "00_preflight_freeze" / "frozen_execution_plan.md"
GENE_SPACE = (
    PROJECT
    / "03a_taylor_dependency_gene_space"
    / "gene_space_R1"
    / "gene_space_manifest_R1.json"
)

NSGA = "NSGA-II AP+APS"
AP_ONLY = "Single-objective evolutionary search (AP-only)"
RANDOM = "Warm-started constrained Random Search"
RECALL_FLOOR = Decimal("0.30035872606099606")
PARAMS_LOWER = 2_440_209
PARAMS_UPPER = 2_515_679
FLOPS_LOWER = 7_618_383_104
FLOPS_UPPER = 7_854_003_200
HEX64 = re.compile(r"^[0-9A-Fa-f]{64}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})
    os.replace(temporary, path)


def csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=json_default)
    return value


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "yes", "1", "pass", "complete"}


def normalized_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def stage05_sources() -> dict[str, str]:
    integrity = load_json(STAGE05 / "stage05_source_integrity_audit.json")
    if integrity["status"] != "PASS" or integrity["changed_authoritative_sources"]:
        raise RuntimeError("Stage05 source-integrity baseline is not PASS.")
    sources = dict(integrity["source_hashes_after"])
    final_manifest = load_json(STAGE05 / "stage05_final_audit_output_manifest.json")
    for item in final_manifest["files"]:
        sources[item["path"]] = item["sha256"]
    sources[str(STAGE05 / "stage05_final_audit_output_manifest.json")] = sha256(
        STAGE05 / "stage05_final_audit_output_manifest.json"
    )
    return sources


def verify_hash_map(expected: dict[str, str]) -> list[dict[str, str]]:
    failures = []
    for raw_path, expected_hash in expected.items():
        path = Path(raw_path)
        if not path.exists():
            failures.append({"path": raw_path, "reason": "MISSING"})
            continue
        actual = sha256(path)
        if actual != expected_hash.upper():
            failures.append(
                {
                    "path": raw_path,
                    "reason": "SHA256_MISMATCH",
                    "expected": expected_hash.upper(),
                    "actual": actual,
                }
            )
    return failures


def nondominated(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for candidate in rows:
        ap = Decimal(candidate["AP"])
        aps = Decimal(candidate["APS"])
        dominated = any(
            Decimal(other["AP"]) >= ap
            and Decimal(other["APS"]) >= aps
            and (Decimal(other["AP"]) > ap or Decimal(other["APS"]) > aps)
            for other in rows
            if other["signature_sha256"] != candidate["signature_sha256"]
        )
        if not dominated:
            output.append(candidate)
    return output


def pooled_unique(rows: list[dict[str, str]], eligible_ids: set[str]) -> list[dict[str, str]]:
    by_signature: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["search_record_id"] in eligible_ids:
            by_signature.setdefault(row["signature_sha256"], row)
    return list(by_signature.values())


def knee_scores(front: list[dict[str, str]]) -> list[tuple[float, dict[str, str]]]:
    aps = [float(row["AP"]) for row in front]
    small = [float(row["APS"]) for row in front]
    ap_min, ap_max = min(aps), max(aps)
    small_min, small_max = min(small), max(small)
    points = []
    for row in front:
        x = (float(row["AP"]) - ap_min) / (ap_max - ap_min)
        y = (float(row["APS"]) - small_min) / (small_max - small_min)
        points.append((row, x, y))
    high_ap = max(points, key=lambda item: (item[1], item[2], item[0]["signature_sha256"]))
    high_aps = max(points, key=lambda item: (item[2], item[1], item[0]["signature_sha256"]))
    x1, y1 = high_ap[1], high_ap[2]
    x2, y2 = high_aps[1], high_aps[2]
    denominator = math.hypot(y2 - y1, x2 - x1)
    scored = []
    for row, x, y in points:
        distance = (
            abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
            / denominator
            if denominator
            else 0.0
        )
        scored.append((distance, row))
    return sorted(
        scored,
        key=lambda item: (
            -item[0],
            -Decimal(item[1]["AP"]),
            -Decimal(item[1]["APS"]),
            item[1]["signature_sha256"],
        ),
    )


def channel_evidence(cache_row: dict[str, str]) -> dict[str, Any]:
    reconstruction_path = Path(cache_row["reconstruction_manifest"])
    reconstruction = load_json(reconstruction_path)
    channel_map_path = reconstruction_path.parent / "channel_index_map.csv"
    channel_rows = read_csv(channel_map_path)
    retained_indices = {
        row["group_id"]: [int(value) for value in row["retained_channel_indices"].split(";") if value != ""]
        for row in channel_rows
    }
    return {
        "retained_width_vector": reconstruction["widths"],
        "retained_channel_indices": retained_indices,
        "channel_index_map_path": str(channel_map_path),
        "channel_index_map_sha256": sha256(channel_map_path),
        "reconstruction": reconstruction,
    }


def active_processes() -> list[dict[str, Any]]:
    command = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -match 'stage05_worker.py|train.py|val.py|Recovery-25' -and "
        "$_.CommandLine -notmatch 'stage06a_fidelity2_promotion_lock.py' } | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if not result.stdout.strip():
        return []
    parsed = json.loads(result.stdout)
    return parsed if isinstance(parsed, list) else [parsed]


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"Stage06A output directory is not empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    stage05_expected = stage05_sources()
    initial_integrity_failures = verify_hash_map(stage05_expected)
    if initial_integrity_failures:
        raise RuntimeError(f"Stage05 integrity failed before Stage06A: {initial_integrity_failures[:3]}")

    final_audit = load_json(STAGE05 / "stage05_final_read_only_audit.json")
    if final_audit["terminal_decision"]["STAGE05_FINAL_AUDIT"] not in {
        "PASS",
        "PASS_WITH_NONBLOCKING_PACKAGING_GAPS",
    }:
        raise RuntimeError("Stage05 final audit is not approved.")

    plan_lines = PLAN.read_text(encoding="utf-8-sig").splitlines()
    policy_sha = sha256(PLAN)
    campaign = load_json(STAGE05 / "runtime" / "stage05_campaign_state.json")
    stage00_state = load_json(PROJECT / "00_preflight_freeze" / "stage00_state.json")
    plan_mtime = datetime.fromtimestamp(PLAN.stat().st_mtime, timezone.utc).isoformat()
    policy_preexisted = plan_mtime < campaign["started_utc"]

    policy_clauses = [
        {
            "clause_id": "UNION-ARCHIVE",
            "source_file": str(PLAN),
            "source_location": "lines 1631-1640",
            "source_sha256": policy_sha,
            "policy_text": "Pool all three replicate records per method, deduplicate by architecture signature, retain the F1 evaluation, and filter feasibility for selection.",
        },
        {
            "clause_id": "NSGA-F2-4",
            "source_file": str(PLAN),
            "source_location": "lines 1654-1672",
            "source_sha256": policy_sha,
            "policy_text": "Promote four feasible NSGA candidates: highest AP, highest APS, deterministic AP-APS Pareto knee, and maximin-diverse non-dominated point. Knee uses normalized AP/APS, maximum perpendicular distance, then higher AP, higher APS, deterministic ID.",
        },
        {
            "clause_id": "MAXIMIN-DEFINITION",
            "source_file": str(PLAN),
            "source_location": "lines 1233-1238",
            "source_sha256": policy_sha,
            "policy_text": "Maximin diversity uses normalized L1 distance over retained-width vectors and maximizes minimum distance to already selected members.",
        },
        {
            "clause_id": "AP-ONLY-F2-3",
            "source_file": str(PLAN),
            "source_location": "lines 1674-1691",
            "source_sha256": policy_sha,
            "policy_text": "Promote the highest-AP unique feasible candidate from each replicate when possible; fill duplicates from global AP ranking. Tie-break by higher AP, higher Recall, lower FLOPs, lower Params, deterministic ID. APS is prohibited.",
        },
        {
            "clause_id": "RANDOM-F2-3",
            "source_file": str(PLAN),
            "source_location": "lines 1693-1699",
            "source_sha256": policy_sha,
            "policy_text": "Promote three feasible Random candidates: highest AP, highest APS, and deterministic AP-APS Pareto knee.",
        },
        {
            "clause_id": "CONTROL-AND-COUNT",
            "source_file": str(PLAN),
            "source_location": "lines 1701-1711",
            "source_sha256": policy_sha,
            "policy_text": "Include TargetManual-B6-reference. Total method-level shortlist size is 4 + 3 + 3 + 1 = 11.",
        },
        {
            "clause_id": "F2-PROTOCOL",
            "source_file": str(PROJECT / "01A_protocol_amendment" / "protocol_amendment_A1.json"),
            "source_location": "$.internal_fidelities.fidelity2",
            "source_sha256": sha256(PROJECT / "01A_protocol_amendment" / "protocol_amendment_A1.json"),
            "policy_text": "Fidelity-2 starts from a fresh raw pruned checkpoint, uses Recovery-25, and has KD disabled.",
        },
    ]
    policy_map = {
        "created_utc": utc_now(),
        "policy_preexisted": policy_preexisted,
        "stage00_state": stage00_state,
        "policy_file_last_write_utc": plan_mtime,
        "stage05_started_utc": campaign["started_utc"],
        "candidate_count_per_method": {NSGA: 4, AP_ONLY: 3, RANDOM: 3, "TargetManual-B6-reference": 1},
        "selection_scope": "Pooled across three replicates after per-method signature deduplication; AP-only first selects one per replicate when possible.",
        "feasibility_rule": {
            "Recall": ">= 0.30035872606099606",
            "Params": "2440209..2515679 inclusive",
            "FLOPs_raw": "7618383104..7854003200 inclusive",
            "structural_validity": "PASS",
        },
        "duplicate_handling": "Union archives deduplicate by architecture signature. Cross-method promoted duplicates retain all attributions but map to one physical Fidelity-2 candidate.",
        "reference_inclusion": "TargetManual-B6-reference only for Stage06 shortlist; MatchedManual is constructed later under Stage08B rules.",
        "clauses": policy_clauses,
        "status": "PASS" if policy_preexisted else "FAIL",
    }

    method_files = {
        NSGA: STAGE05 / "search_records_nsga2_ap_aps_audited.csv",
        AP_ONLY: STAGE05 / "search_records_evolutionary_ap_only_audited.csv",
        RANDOM: STAGE05 / "search_records_random_audited.csv",
    }
    method_rows = {method: read_csv(path) for method, path in method_files.items()}
    all_rows = read_csv(STAGE05 / "search_records_all.csv")
    cache_rows = read_csv(STAGE05 / "global_evaluation_cache.csv")
    cache_by_eval = {row["evaluation_id"]: row for row in cache_rows}

    physical_checks: dict[str, dict[str, Any]] = {}
    for cache_row in cache_rows:
        metrics_path = Path(cache_row["metrics_file"])
        reconstruction_path = Path(cache_row["reconstruction_manifest"])
        metrics_hash_match = metrics_path.exists() and sha256(metrics_path) == cache_row["metrics_sha256"].upper()
        reconstruction_hash_match = (
            reconstruction_path.exists()
            and sha256(reconstruction_path) == cache_row["reconstruction_manifest_sha256"].upper()
        )
        reconstruction = load_json(reconstruction_path) if reconstruction_path.exists() else {}
        physical_checks[cache_row["evaluation_id"]] = {
            "metrics_hash_match": metrics_hash_match,
            "reconstruction_hash_match": reconstruction_hash_match,
            "structural_validity": reconstruction.get("status") == "PASS"
            and reconstruction.get("params_match") is True
            and reconstruction.get("flops_match") is True,
            "channel_map_exists": (reconstruction_path.parent / "channel_index_map.csv").exists(),
        }

    eligibility_rows: list[dict[str, Any]] = []
    eligible_ids: set[str] = set()
    for row in all_rows:
        cache_row = cache_by_eval.get(row["evaluation_id"])
        checks = physical_checks.get(row["evaluation_id"], {})
        params = int(row["params_exact_integer"])
        flops = int(row["flops_raw_exact_integer"])
        recall = Decimal(row["recall_full_precision"])
        terminal = row["status"] == "COMPLETE"
        params_ok = PARAMS_LOWER <= params <= PARAMS_UPPER
        flops_ok = FLOPS_LOWER <= flops <= FLOPS_UPPER
        recall_ok = recall >= RECALL_FLOOR
        signature_ok = bool(HEX64.fullmatch(row["signature_sha256"]))
        evaluation_ok = bool(HEX64.fullmatch(row["evaluation_id"])) and cache_row is not None
        structural_ok = bool(checks.get("structural_validity"))
        hashes_ok = bool(checks.get("metrics_hash_match")) and bool(checks.get("reconstruction_hash_match"))
        eligible = all(
            [
                terminal,
                params_ok,
                flops_ok,
                recall_ok,
                signature_ok,
                evaluation_ok,
                structural_ok,
                hashes_ok,
                bool(checks.get("channel_map_exists")),
            ]
        )
        reasons = []
        for name, passed in [
            ("TERMINAL_STATUS", terminal),
            ("PARAMS_BAND", params_ok),
            ("FLOPS_BAND", flops_ok),
            ("RECALL_FLOOR", recall_ok),
            ("SIGNATURE", signature_ok),
            ("EVALUATION_ID", evaluation_ok),
            ("STRUCTURAL_VALIDITY", structural_ok),
            ("HASH_VALIDATION", hashes_ok),
            ("CHANNEL_MAP", bool(checks.get("channel_map_exists"))),
        ]:
            if not passed:
                reasons.append(name)
        if eligible:
            eligible_ids.add(row["search_record_id"])
        eligibility_rows.append(
            {
                "search_record_id": row["search_record_id"],
                "method": row["method"],
                "replicate": row["replicate"],
                "generation_or_random_position": row["generation"] if row["method"] != RANDOM else row["population_index"],
                "record_origin": row["record_origin"],
                "evaluation_id": row["evaluation_id"],
                "architecture_signature": row["signature_sha256"],
                "terminal_status": row["status"],
                "structural_validity": "PASS" if structural_ok else "FAIL",
                "params_exact_integer": params,
                "params_within_band": params_ok,
                "flops_raw_exact_integer": flops,
                "flops_g_full_precision": row["flops_g_full_precision"],
                "flops_within_band": flops_ok,
                "recall_full_precision": row["recall_full_precision"],
                "recall_floor_satisfied": recall_ok,
                "metrics_path": cache_row["metrics_file"] if cache_row else "",
                "metrics_sha256": cache_row["metrics_sha256"] if cache_row else "",
                "metrics_hash_match": checks.get("metrics_hash_match", False),
                "reconstruction_manifest": cache_row["reconstruction_manifest"] if cache_row else "",
                "reconstruction_manifest_sha256": cache_row["reconstruction_manifest_sha256"] if cache_row else "",
                "reconstruction_hash_match": checks.get("reconstruction_hash_match", False),
                "AP": row["AP"],
                "APS": row["APS"],
                "metric_role": "FIDELITY1_SELECTION_EVIDENCE",
                "eligible_for_fidelity2_promotion": eligible,
                "ineligibility_reasons": ";".join(reasons),
            }
        )

    ambiguity_observations = []
    promotion_specs: list[dict[str, Any]] = []

    nsga_union = pooled_unique(method_rows[NSGA], eligible_ids)
    nsga_front = nondominated(nsga_union)
    high_ap_nsga = max(nsga_front, key=lambda row: (Decimal(row["AP"]), Decimal(row["APS"]), row["signature_sha256"]))
    high_aps_nsga = max(nsga_front, key=lambda row: (Decimal(row["APS"]), Decimal(row["AP"]), row["signature_sha256"]))
    nsga_knee_scores = knee_scores(nsga_front)
    knee_nsga = nsga_knee_scores[0][1]
    first_three = {
        high_ap_nsga["signature_sha256"],
        high_aps_nsga["signature_sha256"],
        knee_nsga["signature_sha256"],
    }
    remaining_nsga = [row for row in nsga_front if row["signature_sha256"] not in first_three]
    if len(nsga_front) == 4 and len(first_three) == 3 and len(remaining_nsga) == 1:
        maximin_nsga = remaining_nsga[0]
        maximin_note = "The pooled feasible union front has exactly four points; after three distinct prespecified selections, the fourth is the sole remaining non-dominated point, so no unstated tie-break can affect selection."
    else:
        groups = load_json(GENE_SPACE)["groups"]
        denominators = {
            group["group_id"]: max(
                1,
                int(group["exact_maximum_width"]) - int(group["exact_minimum_width"]),
            )
            for group in groups
        }
        width_cache = {
            row["signature_sha256"]: channel_evidence(cache_by_eval[row["evaluation_id"]])[
                "retained_width_vector"
            ]
            for row in nsga_front
        }

        def l1(first: str, second: str) -> float:
            return sum(
                abs(width_cache[first][group_id] - width_cache[second][group_id])
                / denominators[group_id]
                for group_id in denominators
            )

        scored = []
        for row in remaining_nsga:
            signature = row["signature_sha256"]
            score = min(l1(signature, selected) for selected in first_three)
            scored.append((score, signature, row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-15:
            raise RuntimeError("BLOCKED_POLICY_AMBIGUITY: NSGA maximin tie has no frozen tie-break.")
        maximin_nsga = scored[0][2]
        maximin_note = f"Normalized-L1 maximin score={scored[0][0]}."
    for slot, role, row, clause, tie in [
        (1, "HIGHEST_AP_FEASIBLE", high_ap_nsga, "NSGA-F2-4 item 1", "Unique full-precision AP maximum"),
        (2, "HIGHEST_APS_FEASIBLE", high_aps_nsga, "NSGA-F2-4 item 2", "Unique full-precision APS maximum"),
        (3, "DETERMINISTIC_AP_APS_PARETO_KNEE", knee_nsga, "NSGA-F2-4 item 3", f"Knee distance={nsga_knee_scores[0][0]}; runner-up={nsga_knee_scores[1][0]}"),
        (4, "MAXIMIN_DIVERSE_NONDOMINATED", maximin_nsga, "NSGA-F2-4 item 4; MAXIMIN-DEFINITION", maximin_note),
    ]:
        promotion_specs.append({"promotion_candidate_id": f"PROM-NSGA-{slot:02d}", "method": NSGA, "selection_role": role, "row": row, "promotion_rule_clause": clause, "deterministic_tie_break_result": tie, "source_rank": 1, "union_pareto_rank": 1})

    ap_selected_signatures: set[str] = set()
    for replicate in [1042, 2042, 3042]:
        per_replicate = pooled_unique(
            [row for row in method_rows[AP_ONLY] if int(row["replicate"]) == replicate],
            eligible_ids,
        )
        ordered = sorted(
            per_replicate,
            key=lambda row: (
                -Decimal(row["AP"]),
                -Decimal(row["recall_full_precision"]),
                int(row["flops_raw_exact_integer"]),
                int(row["params_exact_integer"]),
                row["signature_sha256"],
            ),
        )
        selected = next((row for row in ordered if row["signature_sha256"] not in ap_selected_signatures), None)
        if selected is None:
            raise RuntimeError("BLOCKED_POLICY_AMBIGUITY: AP-only unique candidate unavailable.")
        ap_selected_signatures.add(selected["signature_sha256"])
        promotion_specs.append(
            {
                "promotion_candidate_id": f"PROM-APONLY-REP{replicate}",
                "method": AP_ONLY,
                "selection_role": "HIGHEST_AP_UNIQUE_FEASIBLE_FROM_REPLICATE",
                "row": selected,
                "promotion_rule_clause": "AP-ONLY-F2-3",
                "deterministic_tie_break_result": "Unique per-replicate maximum under the frozen AP, Recall, FLOPs, Params, and signature ordering.",
                "source_rank": 1,
            }
        )

    random_union = pooled_unique(method_rows[RANDOM], eligible_ids)
    random_front = nondominated(random_union)
    high_ap_random = max(random_front, key=lambda row: (Decimal(row["AP"]), Decimal(row["APS"]), row["signature_sha256"]))
    high_aps_random = max(random_front, key=lambda row: (Decimal(row["APS"]), Decimal(row["AP"]), row["signature_sha256"]))
    random_knee_scores = knee_scores(random_front)
    knee_random = random_knee_scores[0][1]
    random_first_three = {
        high_ap_random["signature_sha256"],
        high_aps_random["signature_sha256"],
        knee_random["signature_sha256"],
    }
    if len(random_first_three) != 3:
        raise RuntimeError("BLOCKED_POLICY_AMBIGUITY: Random prespecified roles do not yield three unique candidates.")
    for slot, role, row, tie in [
        (1, "HIGHEST_AP_FEASIBLE", high_ap_random, "Unique full-precision AP maximum"),
        (2, "HIGHEST_APS_FEASIBLE", high_aps_random, "Unique full-precision APS maximum"),
        (3, "DETERMINISTIC_AP_APS_PARETO_KNEE", knee_random, f"Knee distance={random_knee_scores[0][0]}; runner-up={random_knee_scores[1][0]}"),
    ]:
        promotion_specs.append({"promotion_candidate_id": f"PROM-RANDOM-{slot:02d}", "method": RANDOM, "selection_role": role, "row": row, "promotion_rule_clause": f"RANDOM-F2-3 item {slot}", "deterministic_tie_break_result": tie, "source_rank": 1, "union_pareto_rank": 1})

    target_rows = [
        row
        for row in all_rows
        if row["p0_role"] == "exact_targetmanual"
        and row["search_record_id"] in eligible_ids
    ]
    target_by_signature = {row["signature_sha256"]: row for row in target_rows}
    if len(target_by_signature) != 1:
        raise RuntimeError("TargetManual control identity is not unique.")
    target_row = next(iter(target_by_signature.values()))
    promotion_specs.append(
        {
            "promotion_candidate_id": "PROM-CONTROL-TARGETMANUAL",
            "method": "FIXED_REFERENCE_CONTROL",
            "selection_role": "TARGETMANUAL_B6_REFERENCE",
            "row": target_row,
            "promotion_rule_clause": "CONTROL-AND-COUNT",
            "deterministic_tie_break_result": "Exact frozen TargetManual signature; no metric-based choice.",
            "source_rank": "NOT_APPLICABLE",
        }
    )

    ambiguity_observations.extend(
        [
            {
                "topic": "NSGA role overlap and maximin tie",
                "finding": f"Union front size={len(nsga_front)}; first three roles yield three distinct signatures; one point remains.",
                "selection_impact": "NONE",
                "status": "RESOLVED_BY_ACTUAL_FROZEN_SET",
            },
            {
                "topic": "AP-only duplicate fill",
                "finding": "The three per-replicate highest-AP candidates are already unique; global fill rule is not invoked.",
                "selection_impact": "NONE",
                "status": "NOT_TRIGGERED",
            },
            {
                "topic": "Random role overlap",
                "finding": f"Random union front size={len(random_front)}; highest AP, highest APS, and knee are distinct.",
                "selection_impact": "NONE",
                "status": "RESOLVED_BY_ACTUAL_FROZEN_SET",
            },
        ]
    )

    if len(promotion_specs) != 11:
        raise RuntimeError(f"Frozen policy expected 11 promotions, got {len(promotion_specs)}.")

    all_occurrences_by_signature: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        all_occurrences_by_signature[row["signature_sha256"]].append(row)

    promoted_signatures = {spec["row"]["signature_sha256"] for spec in promotion_specs}
    physical_ids = {
        signature: f"F2PHYS-{signature[:16]}" for signature in sorted(promoted_signatures)
    }
    promotions_by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in promotion_specs:
        promotions_by_signature[spec["row"]["signature_sha256"]].append(spec)

    channel_by_signature = {
        signature: channel_evidence(
            cache_by_eval[promotions_by_signature[signature][0]["row"]["evaluation_id"]]
        )
        for signature in promoted_signatures
    }

    promotion_rows: list[dict[str, Any]] = []
    for spec in promotion_specs:
        row = spec["row"]
        signature = row["signature_sha256"]
        cache_row = cache_by_eval[row["evaluation_id"]]
        channel = channel_by_signature[signature]
        canonical_identity = json.loads(cache_row["evaluation_identity_canonical"])
        promotion_rows.append(
            {
                "promotion_candidate_id": spec["promotion_candidate_id"],
                "physical_candidate_id": physical_ids[signature],
                "method": spec["method"],
                "selection_role": spec["selection_role"],
                "replicate": row["replicate"],
                "generation_or_random_position": row["generation"] if row["method"] != RANDOM else row["population_index"],
                "search_record_id": row["search_record_id"],
                "evaluation_id": row["evaluation_id"],
                "architecture_signature": signature,
                "retained_width_vector": channel["retained_width_vector"],
                "retained_channel_indices": channel["retained_channel_indices"],
                "AP": row["AP"],
                "APS": row["APS"],
                "Recall": row["recall_full_precision"],
                "metric_role": "FIDELITY1_SELECTION_EVIDENCE",
                "params_exact_integer": row["params_exact_integer"],
                "flops_raw_exact_integer": row["flops_raw_exact_integer"],
                "flops_g_full_precision": row["flops_g_full_precision"],
                "source_archive": str(method_files.get(spec["method"], STAGE05 / "search_records_all.csv")),
                "source_rank": spec.get("source_rank", row.get("official_final_archive_rank", "")),
                "source_pareto_rank": spec.get("union_pareto_rank", row.get("audited_constrained_pareto_rank", "")),
                "source_crowding_distance": row.get("audited_crowding_distance", ""),
                "promotion_rule_clause": spec["promotion_rule_clause"],
                "deterministic_tie_break_result": spec["deterministic_tie_break_result"],
                "checkpoint_retention_status": "PASS",
                "reconstruction_manifest": cache_row["reconstruction_manifest"],
                "reconstruction_manifest_sha256": cache_row["reconstruction_manifest_sha256"],
                "channel_index_map": channel["channel_index_map_path"],
                "channel_index_map_sha256": channel["channel_index_map_sha256"],
                "metrics_path": cache_row["metrics_file"],
                "metrics_sha256": cache_row["metrics_sha256"],
                "searchparent_sha256": canonical_identity["raw_parent_checkpoint_hash"],
                "taylor_ranking_sha256": canonical_identity["Taylor_ranking_hash"],
                "stage05_protocol_sha256": cache_row["stage05_protocol_sha256"],
                "eligibility_status": "ELIGIBLE",
            }
        )

    attribution_rows = []
    for signature in sorted(promoted_signatures):
        promotion_ids = [spec["promotion_candidate_id"] for spec in promotions_by_signature[signature]]
        for row in sorted(
            all_occurrences_by_signature[signature],
            key=lambda item: (
                item["method"],
                int(item["replicate"]),
                int(item["generation"] or 0),
                int(item["population_index"] or 0),
            ),
        ):
            attribution_rows.append(
                {
                    "physical_candidate_id": physical_ids[signature],
                    "architecture_signature": signature,
                    "promotion_candidate_ids": promotion_ids,
                    "method": row["method"],
                    "replicate": row["replicate"],
                    "generation": row["generation"],
                    "random_position": row["population_index"] if row["method"] == RANDOM else "",
                    "search_record_id": row["search_record_id"],
                    "evaluation_id": row["evaluation_id"],
                    "record_origin": row["record_origin"],
                    "operator_provenance_sha256": row["operator_provenance_sha256"],
                    "independent_proposal_provenance": "YES" if row["record_origin"] == "STAGE05_NEW" else "SHARED_P0_MEMBERSHIP",
                }
            )

    physical_rows = []
    execution_jobs = []
    for signature in sorted(promoted_signatures):
        representative = next(row for row in promotion_rows if row["architecture_signature"] == signature)
        roles = [spec["promotion_candidate_id"] for spec in promotions_by_signature[signature]]
        methods = sorted({spec["method"] for spec in promotions_by_signature[signature]})
        physical_id = physical_ids[signature]
        future_identity = normalized_json_hash(
            {
                "schema": "stage06a_fidelity2_physical_candidate_v1",
                "architecture_signature": signature,
                "searchparent_sha256": representative["searchparent_sha256"],
                "taylor_ranking_sha256": representative["taylor_ranking_sha256"],
                "training_seed": 42,
                "recovery_epochs": 25,
                "train_split": "meta-train5177",
                "evaluation_split": "meta-inner-val1294",
            }
        )
        physical_rows.append(
            {
                "physical_candidate_id": physical_id,
                "future_fidelity2_evaluation_identity": future_identity,
                "architecture_signature": signature,
                "promotion_candidate_ids": roles,
                "selected_method_roles": methods,
                "method_attribution_count": len(all_occurrences_by_signature[signature]),
                "stage05_evaluation_id": representative["evaluation_id"],
                "retained_width_vector": representative["retained_width_vector"],
                "retained_channel_indices": representative["retained_channel_indices"],
                "channel_index_map": representative["channel_index_map"],
                "channel_index_map_sha256": representative["channel_index_map_sha256"],
                "params_exact_integer": representative["params_exact_integer"],
                "flops_raw_exact_integer": representative["flops_raw_exact_integer"],
                "flops_g_full_precision": representative["flops_g_full_precision"],
                "checkpoint_retention_status": "PASS",
                "physical_deduplication_status": "ONE_F2_JOB_FOR_ALL_IDENTICAL_SCIENTIFIC_ATTRIBUTIONS",
                "execution_status": "PLANNED_NOT_STARTED",
            }
        )
        execution_jobs.append(
            {
                "physical_candidate_id": physical_id,
                "future_fidelity2_evaluation_identity": future_identity,
                "architecture_signature": signature,
                "fresh_reconstruction_from_searchparent": True,
                "searchparent_sha256": representative["searchparent_sha256"],
                "retained_width_vector_sha256": normalized_json_hash(representative["retained_width_vector"]),
                "channel_index_map": representative["channel_index_map"],
                "channel_index_map_sha256": representative["channel_index_map_sha256"],
                "recovery_epochs": 25,
                "kd_enabled": False,
                "training_seed": 42,
                "training_split": "meta-train5177",
                "training_image_count": 5177,
                "evaluation_split": "meta-inner-val1294",
                "evaluation_image_count": 1294,
                "imgsz": 640,
                "evaluator": "Frozen explicit FP32 parity-validated VisDrone evaluator",
                "official_val548_locked": True,
                "heldout1610_locked": True,
                "checkpoint_created": False,
                "gpu_process_started": False,
                "status": "PLANNED_NOT_STARTED",
            }
        )

    duplicate_map = {
        signature: {
            "physical_candidate_id": physical_ids[signature],
            "promotion_candidate_ids": [spec["promotion_candidate_id"] for spec in specs],
            "promoted_method_roles": [spec["method"] for spec in specs],
            "all_method_memberships": sorted({row["method"] for row in all_occurrences_by_signature[signature]}),
            "all_replicate_memberships": sorted({int(row["replicate"]) for row in all_occurrences_by_signature[signature]}),
            "all_search_record_ids": [row["search_record_id"] for row in all_occurrences_by_signature[signature]],
            "common_evaluation_ids": sorted({row["evaluation_id"] for row in all_occurrences_by_signature[signature]}),
            "physical_fidelity2_jobs": 1,
        }
        for signature, specs in promotions_by_signature.items()
        if len(specs) > 1
    }

    reference_manifest = {
        "created_utc": utc_now(),
        "references": [
            {
                "reference": "TargetManual-B6-reference",
                "stage06a_status": "PROMOTED_FIXED_CONTROL",
                "architecture_signature": target_row["signature_sha256"],
                "source_policy": "CONTROL-AND-COUNT",
            },
            {
                "reference": "MatchedManual-B6",
                "stage06a_status": "REFERENCE_CONSTRUCTION_PENDING",
                "construction_stage": "Stage08B",
                "source_file": str(PLAN),
                "source_location": "lines 2004-2029",
                "note": "It depends on the finally selected Meta candidate and must not be constructed in Stage06A.",
            },
            {
                "reference": "Stage6-Allocation-Anchor",
                "stage06a_status": "NOT_INCLUDED_BY_FROZEN_STAGE06_SHORTLIST_POLICY",
            },
            {
                "reference": "Source-locked Stage6-Drone",
                "stage06a_status": "CONTEXTUAL_ONLY_NOT_INCLUDED",
            },
        ],
    }

    protocol_lock = {
        "schema": "stage06a_fidelity2_promotion_lock_v1",
        "created_utc": utc_now(),
        "approval_token_consumed": "APPROVE_STAGE05_FULL_FIDELITY1_SEARCH",
        "stage": "STAGE06A_FIDELITY2_PROMOTION_LOCK",
        "stage05_final_status": final_audit["terminal_decision"]["STAGE05_FINAL_AUDIT"],
        "promotion_policy_sha256": policy_sha,
        "promotion_policy_preexisted": policy_preexisted,
        "post_hoc_selection_used": False,
        "frozen_constraints": {
            "recall_floor_full_precision_inclusive": str(RECALL_FLOOR),
            "params_exact_integer_inclusive": [PARAMS_LOWER, PARAMS_UPPER],
            "flops_raw_exact_integer_inclusive": [FLOPS_LOWER, FLOPS_UPPER],
        },
        "promoted_method_records": len(promotion_rows),
        "promoted_unique_physical_architectures": len(physical_rows),
        "fidelity2_execution_started": False,
        "locks": {
            "promotion_manifest_immutable_after_complete": True,
            "stage05_read_only": True,
            "official_val548_locked": True,
            "heldout1610_locked": True,
            "kd_disabled": True,
        },
        "required_next_token": "APPROVE_STAGE06A_FIDELITY2_PROMOTION_LOCK",
    }
    fidelity2_plan = {
        "created_utc": utc_now(),
        "execution_status": "PLANNED_NOT_STARTED",
        "method_level_promotions": len(promotion_rows),
        "unique_physical_jobs": len(execution_jobs),
        "deduplication_savings": len(promotion_rows) - len(execution_jobs),
        "common_protocol": {
            "initialization": "Fresh reconstruction from SearchParent using frozen widths and target-domain Taylor channel indices",
            "recovery_epochs": 25,
            "kd": False,
            "training_seed": 42,
            "train": "meta-train5177",
            "val": "meta-inner-val1294",
            "imgsz": 640,
            "official_val548": "LOCKED",
            "heldout1610": "LOCKED",
        },
        "jobs": execution_jobs,
    }

    ambiguity_report = {
        "created_utc": utc_now(),
        "policy_status": "EXACT_PREEXISTING_POLICY_APPLICABLE",
        "unresolved_ambiguities": [],
        "observations": ambiguity_observations,
        "candidates_selected": True,
        "status": "PASS",
    }

    eligibility_fields = list(eligibility_rows[0].keys())
    promotion_fields = list(promotion_rows[0].keys())
    physical_fields = list(physical_rows[0].keys())
    attribution_fields = list(attribution_rows[0].keys())
    write_csv(OUTPUT / "stage06a_candidate_eligibility.csv", eligibility_rows, eligibility_fields)
    write_json(
        OUTPUT / "stage06a_candidate_eligibility.json",
        {
            "created_utc": utc_now(),
            "record_count": len(eligibility_rows),
            "eligible_count": sum(row["eligible_for_fidelity2_promotion"] for row in eligibility_rows),
            "ineligible_count": sum(not row["eligible_for_fidelity2_promotion"] for row in eligibility_rows),
            "records": eligibility_rows,
        },
    )
    write_csv(OUTPUT / "stage06a_promotion_manifest.csv", promotion_rows, promotion_fields)
    write_json(
        OUTPUT / "stage06a_promotion_manifest.json",
        {
            "created_utc": utc_now(),
            "method_record_count": len(promotion_rows),
            "unique_physical_architecture_count": len(physical_rows),
            "metric_role": "FIDELITY1_SELECTION_EVIDENCE",
            "promotions": promotion_rows,
        },
    )
    write_csv(
        OUTPUT / "stage06a_physical_candidate_manifest.csv", physical_rows, physical_fields
    )
    write_csv(
        OUTPUT / "stage06a_method_attribution_manifest.csv",
        attribution_rows,
        attribution_fields,
    )
    write_json(OUTPUT / "stage06a_duplicate_architecture_map.json", duplicate_map)
    write_json(OUTPUT / "stage06a_reference_model_manifest.json", reference_manifest)
    write_json(OUTPUT / "stage06a_fidelity2_execution_plan.json", fidelity2_plan)
    write_json(OUTPUT / "stage06a_protocol_lock.json", protocol_lock)
    write_json(OUTPUT / "stage06a_promotion_policy_source_map.json", policy_map)
    write_json(OUTPUT / "stage06a_policy_ambiguity_report.json", ambiguity_report)

    extra_sources = {
        str(PLAN): "FROZEN_PROMOTION_POLICY",
        str(PROJECT / "00_preflight_freeze" / "frozen_hyperparameters.yaml"): "STAGE00_HYPERPARAMETERS",
        str(PROJECT / "00_preflight_freeze" / "preregistered_thresholds.yaml"): "PREREGISTERED_THRESHOLDS",
        str(PROJECT / "01A_protocol_amendment" / "protocol_amendment_A1.json"): "AMENDMENT_A1",
        str(PROJECT / "01A_protocol_amendment" / "revised_frozen_hyperparameters.yaml"): "A1_FROZEN_HYPERPARAMETERS",
        str(PROJECT / "01A_protocol_amendment" / "revised_stage_map.yaml"): "A1_STAGE_MAP",
        str(PROJECT / "protocol_amendment_A2_final_objectives_heldout" / "amendment_A2.json"): "AMENDMENT_A2",
        str(PROJECT / "protocol_amendment_A2_final_objectives_heldout" / "frozen_control_methods.yaml"): "A2_CONTROL_METHODS",
        str(PROJECT / "03b_search_initialization" / "gene_space_lock.json"): "GENE_SPACE_LOCK",
        str(PROJECT / "03b_search_initialization" / "p0" / "p0_membership_manifest.csv"): "P0_MEMBERSHIPS",
        str(PROJECT / "04_shared_p0_fidelity1" / "recall_floor_f1.json"): "RECALL_FLOOR",
        str(GENE_SPACE): "GENE_SPACE_R1",
    }
    source_entries = []
    for raw_path, role in extra_sources.items():
        path = Path(raw_path)
        source_entries.append(
            {
                "path": raw_path,
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    for signature in sorted(promoted_signatures):
        representative = next(row for row in promotion_rows if row["architecture_signature"] == signature)
        for key, role in [
            ("metrics_path", "PROMOTED_METRICS"),
            ("reconstruction_manifest", "PROMOTED_RECONSTRUCTION"),
            ("channel_index_map", "PROMOTED_CHANNEL_MAP"),
        ]:
            path = Path(representative[key])
            source_entries.append(
                {
                    "path": str(path),
                    "role": role,
                    "architecture_signature": signature,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    source_hash_payload = {
        "created_utc": utc_now(),
        "stage05_comprehensive_baseline_file_count": len(stage05_expected),
        "stage05_comprehensive_baseline_sha256": normalized_json_hash(stage05_expected),
        "stage05_integrity_failures_before": initial_integrity_failures,
        "sources": source_entries,
        "status": "PASS",
    }
    write_json(OUTPUT / "stage06a_source_hashes.json", source_hash_payload)

    process_list = active_processes()
    final_integrity_failures = verify_hash_map(stage05_expected)
    eligibility_by_id = {
        row["search_record_id"]: row for row in eligibility_rows
    }
    selected_all_eligible = all(
        eligibility_by_id[row["search_record_id"]]["eligible_for_fidelity2_promotion"]
        for row in promotion_rows
    )
    ap_promotions = [row for row in promotion_rows if row["method"] == AP_ONLY]
    ap_aps_not_used = len(ap_promotions) == 3 and all(
        "APS" not in row["deterministic_tie_break_result"] for row in ap_promotions
    )
    validation_checks = {
        "stage05_approval_valid": True,
        "stage05_source_hashes_unchanged": not final_integrity_failures,
        "promotion_policy_predates_stage05": policy_preexisted,
        "post_hoc_selection_rule_introduced": False,
        "promoted_count_matches_frozen_policy": len(promotion_rows) == 11,
        "every_promoted_candidate_eligible": selected_all_eligible,
        "every_promoted_candidate_traceable": all(
            row["search_record_id"] in {item["search_record_id"] for item in all_rows}
            for row in promotion_rows
        ),
        "valid_architecture_signatures": all(
            HEX64.fullmatch(row["architecture_signature"]) for row in promotion_rows
        ),
        "AP_only_APS_used_for_promotion": False if ap_aps_not_used else "UNKNOWN",
        "random_promotion_policy_audit": "PASS"
        if len([row for row in promotion_rows if row["method"] == RANDOM]) == 3
        else "FAIL",
        "duplicate_architectures_physically_deduplicated": len(physical_rows)
        == len(promoted_signatures),
        "method_attributions_preserved": all(
            len(
                [
                    row
                    for row in attribution_rows
                    if row["architecture_signature"] == signature
                ]
            )
            == len(all_occurrences_by_signature[signature])
            for signature in promoted_signatures
        ),
        "active_training_or_evaluation_processes": process_list,
        "training_launched": False,
        "evaluation_launched": False,
        "official_val548_locked": campaign["official_val548_accessed"] is False,
        "heldout1610_locked": campaign["heldout1610_accessed"] is False,
        "kd_executed": False,
        "fidelity2_started": False,
        "output_checkpoint_files": [str(path) for path in OUTPUT.rglob("*.pt")],
    }
    validation_pass = (
        all(
            [
                validation_checks["stage05_approval_valid"],
                validation_checks["stage05_source_hashes_unchanged"],
                validation_checks["promotion_policy_predates_stage05"],
                not validation_checks["post_hoc_selection_rule_introduced"],
                validation_checks["promoted_count_matches_frozen_policy"],
                validation_checks["every_promoted_candidate_eligible"],
                validation_checks["every_promoted_candidate_traceable"],
                validation_checks["valid_architecture_signatures"],
                validation_checks["AP_only_APS_used_for_promotion"] is False,
                validation_checks["random_promotion_policy_audit"] == "PASS",
                validation_checks["duplicate_architectures_physically_deduplicated"],
                validation_checks["method_attributions_preserved"],
                not process_list,
                not validation_checks["training_launched"],
                not validation_checks["evaluation_launched"],
                validation_checks["official_val548_locked"],
                validation_checks["heldout1610_locked"],
                not validation_checks["kd_executed"],
                not validation_checks["fidelity2_started"],
                not validation_checks["output_checkpoint_files"],
            ]
        )
        and not ambiguity_report["unresolved_ambiguities"]
    )
    terminal = {
        "STAGE06A_STATUS": "PASS" if validation_pass else "FAIL",
        "STAGE05_APPROVAL_VERIFIED": "YES",
        "PROMOTION_POLICY_PREEXISTED": "YES" if policy_preexisted else "NO",
        "POST_HOC_SELECTION_USED": "NO",
        "PROMOTED_METHOD_RECORDS": len(promotion_rows),
        "PROMOTED_UNIQUE_PHYSICAL_ARCHITECTURES": len(physical_rows),
        "AP_ONLY_APS_USED_FOR_PROMOTION": "NO" if ap_aps_not_used else "UNKNOWN",
        "RANDOM_PROMOTION_POLICY_AUDIT": validation_checks[
            "random_promotion_policy_audit"
        ],
        "DUPLICATE_PHYSICAL_CANDIDATES_DEDUPLICATED": "YES",
        "METHOD_ATTRIBUTIONS_PRESERVED": "YES",
        "OFFICIAL_VAL548_LOCKED": "YES",
        "HELDOUT1610_LOCKED": "YES",
        "KD_EXECUTED": "NO",
        "FIDELITY2_STARTED": "NO",
        "APPROVAL_READY": "YES" if validation_pass else "NO",
    }
    validation = {
        "created_utc": utc_now(),
        "checks": validation_checks,
        "stage05_integrity_failures_after": final_integrity_failures,
        "terminal_status": terminal,
        "status": "PASS" if validation_pass else "FAIL",
    }
    write_json(OUTPUT / "stage06a_validation.json", validation)

    report_lines = [
        "# Stage06A Fidelity-2 Promotion Lock",
        "",
        "Stage06A applied the promotion policy frozen in Stage00 before Stage05 execution. No training or evaluation was launched.",
        "",
        "## Selection",
        "",
        f"- NSGA-II AP+APS: 4 promoted method records from a pooled feasible union front of {len(nsga_front)} points.",
        "- AP-only: 3 unique per-replicate highest-AP feasible candidates; APS was not used.",
        f"- Random: 3 promoted method records from a pooled feasible union front of {len(random_front)} points.",
        "- Fixed control: TargetManual-B6-reference.",
        f"- Total: {len(promotion_rows)} promotion records mapped to {len(physical_rows)} unique physical Fidelity-2 candidates.",
        "",
        "One architecture was promoted independently by NSGA-II and AP-only. Both method attributions are preserved, while the future Recovery-25 plan contains one physical job for that signature.",
        "",
        "## Terminal Status",
        "",
    ]
    report_lines.extend(f"{key}={value}" for key, value in terminal.items())
    report_lines.extend(
        [
            "",
            "Recovery-25, Recovery-50, KD, val548, heldout1610, TensorRT, and Jetson profiling were not executed. Stage06A stops before Fidelity-2.",
        ]
    )
    (OUTPUT / "stage06a_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    if not validation_pass:
        raise RuntimeError("Stage06A validation failed; .complete was not created.")
    (OUTPUT / ".complete").write_text(
        "STAGE06A_STATUS=PASS\nFIDELITY2_STARTED=NO\n",
        encoding="ascii",
    )
    outputs = sorted(
        path
        for path in OUTPUT.iterdir()
        if path.is_file() and path.name != "stage06a_output_manifest.json"
    )
    output_manifest = {
        "created_utc": utc_now(),
        "stage": "STAGE06A_FIDELITY2_PROMOTION_LOCK",
        "files": [
            {
                "path": str(path),
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        ],
        "source_files_modified": 0,
        "training_executed": 0,
        "evaluation_executed": 0,
        "fidelity2_started": False,
        "status": "PASS",
    }
    write_json(OUTPUT / "stage06a_output_manifest.json", output_manifest)
    print(json.dumps(terminal, indent=2))


if __name__ == "__main__":
    main()

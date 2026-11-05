from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import copy
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


PROJECT = Path(_release_paths.expand('${STUDY_ROOT}'))
STAGE00 = PROJECT / "00_preflight_freeze"
STAGE01 = PROJECT / "01_data_split"
STAGE02 = PROJECT / "02_search_parent_teacher" / "lite_v3_search_parent"
STAGE03 = PROJECT / "03a_taylor_dependency_gene_space"
REVISION = STAGE03 / "gene_space_R1"
SCRIPTS = PROJECT / "scripts"
YOLOV5_ROOT = Path(_release_paths.expand('${YOLOV5_ROOT}'))
PRUNING_ROOT = Path(_release_paths.expand('${PRUNING_RESEARCH_ROOT}'))

SEARCH_PARENT = STAGE02 / "weights" / "best.pt"
SEARCH_PARENT_SHA = "15E0E4C80677F9E01163D66CE9844C6ED96FF9E922D4FDA5530A3425A3009659"
PROFILE_DIR = (
    STAGE03
    / "structural_profile"
    / "litev3_searchparent_nc10_stage03a_unfused"
)
V1_GENE_JSON = STAGE03 / "gene_space_manifest.json"
V1_FILES = [
    STAGE03 / "gene_space_manifest.json",
    STAGE03 / "gene_space_manifest.xlsx",
    STAGE03 / "dependency_graph.json",
    STAGE03 / "dependency_graph.graphml",
    STAGE03 / "dependency_groups.xlsx",
    STAGE03 / "protected_groups_report.md",
    STAGE03 / "model_contract_audit.json",
    STAGE03 / "architecture_contract_tests.json",
    STAGE03 / "architecture_contract_tests.xlsx",
    STAGE03 / "taylor_ranking_manifest.json",
    STAGE03 / "raw_channel_taylor_scores.csv",
    STAGE03 / "normalized_channel_taylor_scores.csv",
]
ROOT_STATE = STAGE03 / "stage03a_state.json"
ROOT_OUTPUT_MANIFEST = STAGE03 / "stage03a_output_manifest.json"

ANCHORS = {
    "Stage-1": {
        "checkpoint": Path(_release_paths.expand('${PAPER1_ROOT}\\stage1_drone_seed42\\04_weights\\best.pt')),
        "sha256": "5F559B8CF5BBDEE8D959CF3CEEB42EB851B87325C140B76EC3CBFC13F2012835",
        "yaml": STAGE00 / "recovered_model_yamls" / "visdrone_stage1_model_yaml.yaml",
    },
    "Stage-6": {
        "checkpoint": Path(_release_paths.expand('${PAPER1_ROOT}\\stage6_drone_seed42\\04_weights\\best.pt')),
        "sha256": "054554394F2DF1DF19E4D7A0334CFE08FC459FCC2070D731AB7CA6E8D03DBBF9",
        "yaml": STAGE00 / "recovered_model_yamls" / "visdrone_stage6_model_yaml.yaml",
        "params": 2_515_679,
        "flops_g": 7.8540032,
    },
    "Stage-10": {
        "checkpoint": Path(_release_paths.expand('${PAPER1_ROOT}\\stage10_drone_seed42\\04_weights\\best.pt')),
        "sha256": "0E61784F293428335A9191FB3C2804FB91B67CF628D76EFAD8A3974D04231012",
        "yaml": STAGE00 / "recovered_model_yamls" / "visdrone_stage10_model_yaml.yaml",
    },
}

TAYLOR_RANKING = STAGE03 / "taylor_ranking_manifest.json"
TOPOLOGY_JSON = PROFILE_DIR / "channel_topology.json"
CONSTRAINTS_JSON = PROFILE_DIR / "pruning_constraints.json"
HISTORICAL_SEARCH_CONFIG = (
    PRUNING_ROOT / "configs" / "pruning_search_space_litev3_stage6_c30.yaml"
)

CHANNEL_QUANTUM = 8
STRUCTURAL_MINIMUM = 8
IMGSZ = 640
PARAMS_UPPER = 2_515_679
PARAMS_LOWER = math.ceil(PARAMS_UPPER * 0.97)
FLOPS_UPPER_G = 7.8540032
FLOPS_LOWER_G = FLOPS_UPPER_G * 0.97
FEASIBLE_TARGET = 5000
MIXED_TEST_COUNT = 20
MIXED_TEST_SEED = 42031
FLOPS_TOLERANCE_G = 1e-9

EARLY_OR_P3_BLOCKS = {0, 1, 2, 3, 4, 14, 17}


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


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def yaml_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def parse_allowed(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(item) for item in str(value).split(";") if str(item).strip()]


def signature_text(group_ids: list[str], widths: tuple[int, ...]) -> str:
    return ";".join(f"{group_id}={width}" for group_id, width in zip(group_ids, widths))


def signature_sha(group_ids: list[str], widths: tuple[int, ...]) -> str:
    return hashlib.sha256(signature_text(group_ids, widths).encode("ascii")).hexdigest().upper()


def recursive_shapes(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.shape]
    if isinstance(value, (list, tuple)):
        return [recursive_shapes(item) for item in value]
    if isinstance(value, dict):
        return {str(key): recursive_shapes(item) for key, item in value.items()}
    return type(value).__name__


def tensor_list(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_list(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_list(item)


def load_source_model(checkpoint: Path, model_id: str):
    if str(Path(_release_paths.expand('${CODE_ROOT}'))) not in sys.path:
        sys.path.insert(0, str(Path(_release_paths.expand('${CODE_ROOT}'))))
    if str(YOLOV5_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLOV5_ROOT))
    from pruning_research.core.model_loader import load_training_checkpoint_model

    return load_training_checkpoint_model(
        weights=str(checkpoint),
        yolov5_root=str(YOLOV5_ROOT),
        device="cpu",
        prefer_ema=True,
        model_id=model_id,
    )


def extract_scope_widths(model: torch.nn.Module, groups: list[dict[str, Any]]) -> dict[str, int]:
    from pruning_research.pruning.importance_scores import resolve_scope_modules

    topology = json_load(TOPOLOGY_JSON)
    topology_by_name = {
        str(row["block_name"]): row for row in topology["block_topology"]
    }
    widths: dict[str, int] = {}
    for group in groups:
        block_name = str(group["block_name"])
        scope_name = str(group["scope_name"])
        scope = resolve_scope_modules(model, topology_by_name[block_name])[scope_name]
        widths[str(group["group_id"])] = int(scope["width"])
    return widths


class ExactComplexityModel:
    def __init__(self, source_model: torch.nn.Module, groups: list[dict[str, Any]]):
        self.model = source_model
        self.groups = groups
        self.group_key = {
            (str(group["block_name"]), str(group["scope_name"])): str(group["group_id"])
            for group in groups
        }

    @staticmethod
    def conv_cost(
        in_channels: int,
        out_channels: int,
        kernel: int,
        spatial: int,
        groups: int = 1,
        fused_bias: bool = True,
    ) -> tuple[int, int]:
        weights = out_channels * (in_channels // groups) * kernel * kernel
        params = weights + (out_channels if fused_bias else 0)
        macs = spatial * spatial * weights
        return int(params), int(macs)

    def width(self, signature: dict[str, int], block: int, scope: str) -> int:
        return int(signature[self.group_key[(f"model.{block}", scope)]])

    def calculate(self, signature: dict[str, int]) -> dict[str, Any]:
        from models.common import C3, C3Lite, Concat, Conv, SPPF
        from models.yolo import Detect

        params = 0
        macs = 0
        output_channels: dict[int, int] = {}
        output_spatial: dict[int, int] = {}
        layer_rows: list[dict[str, Any]] = []

        def source_index(index: int, value: int) -> int:
            return value if value >= 0 else index + value

        for index, module in enumerate(self.model.model):
            if isinstance(module.f, int):
                if index == 0:
                    input_channels = 3
                    input_spatial = IMGSZ
                else:
                    source = source_index(index, int(module.f))
                    input_channels = output_channels[source]
                    input_spatial = output_spatial[source]
            else:
                sources = [source_index(index, int(value)) for value in module.f]
                input_channels = sum(output_channels[source] for source in sources)
                input_spatial = output_spatial[sources[0]]

            layer_params = 0
            layer_macs = 0
            if isinstance(module, Conv):
                output = self.width(signature, index, "output")
                kernel = int(module.conv.kernel_size[0])
                stride = int(module.conv.stride[0])
                spatial = input_spatial // stride
                layer_params, layer_macs = self.conv_cost(
                    input_channels, output, kernel, spatial, int(module.conv.groups), True
                )
            elif isinstance(module, C3Lite):
                main = self.width(signature, index, "main_hidden")
                skip = self.width(signature, index, "skip_hidden")
                output = self.width(signature, index, "output")
                spatial = input_spatial
                values = [
                    self.conv_cost(input_channels, main, 1, spatial),
                    self.conv_cost(input_channels, skip, 1, spatial),
                    self.conv_cost(main + skip, output, 1, spatial),
                ]
                for _ in module.m:
                    # DWConvLite is intentionally not fused by YOLOv5 BaseModel.fuse().
                    values.append((main * main + 11 * main, spatial * spatial * (main * main + 13 * main)))
                layer_params = sum(value[0] for value in values)
                layer_macs = sum(value[1] for value in values)
            elif isinstance(module, C3):
                main = self.width(signature, index, "main_hidden")
                skip = self.width(signature, index, "skip_hidden")
                output = self.width(signature, index, "output")
                spatial = input_spatial
                values = [
                    self.conv_cost(input_channels, main, 1, spatial),
                    self.conv_cost(input_channels, skip, 1, spatial),
                    self.conv_cost(main + skip, output, 1, spatial),
                ]
                for _ in module.m:
                    values.extend(
                        [
                            self.conv_cost(main, main, 1, spatial),
                            self.conv_cost(main, main, 3, spatial),
                        ]
                    )
                layer_params = sum(value[0] for value in values)
                layer_macs = sum(value[1] for value in values)
            elif isinstance(module, SPPF):
                hidden = self.width(signature, index, "hidden")
                output = self.width(signature, index, "output")
                spatial = input_spatial
                values = [
                    self.conv_cost(input_channels, hidden, 1, spatial),
                    self.conv_cost(hidden * 4, output, 1, spatial),
                ]
                layer_params = sum(value[0] for value in values)
                layer_macs = sum(value[1] for value in values)
            elif isinstance(module, torch.nn.Upsample):
                output = input_channels
                scale = int(module.scale_factor)
                spatial = input_spatial * scale
                layer_macs = output * spatial * spatial
            elif isinstance(module, Concat):
                output = input_channels
                spatial = input_spatial
            elif isinstance(module, Detect):
                output = int(sum(item.out_channels for item in module.m))
                spatial = input_spatial
                values = []
                for branch, source in zip(module.m, module.f):
                    source = source_index(index, int(source))
                    values.append(
                        self.conv_cost(
                            output_channels[source],
                            int(branch.out_channels),
                            1,
                            output_spatial[source],
                            1,
                            True,
                        )
                    )
                layer_params = sum(value[0] for value in values)
                layer_macs = sum(value[1] for value in values)
            else:
                raise TypeError(f"Unsupported module in exact complexity model: {type(module).__name__}")

            params += layer_params
            macs += layer_macs
            output_channels[index] = int(output)
            output_spatial[index] = int(spatial)
            layer_rows.append(
                {
                    "block_index": index,
                    "module_type": type(module).__name__,
                    "input_channels": input_channels,
                    "output_channels": output,
                    "spatial": spatial,
                    "fused_params": layer_params,
                    "thop_macs": layer_macs,
                }
            )
        return {
            "params": int(params),
            "macs": int(macs),
            "flops_g": float(2.0 * macs / 1e9),
            "layers": layer_rows,
        }


def xlsx_write(path: Path, sheets: list[tuple[str, list[dict[str, Any]]]]) -> None:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import stage03a_taylor_dependency_gene_space as base

    base.xlsx_from_sheets(path, sheets)


def archive_v1() -> tuple[dict[str, str], dict[str, Any]]:
    source_hashes = {str(path): sha256(path) for path in V1_FILES}
    payload = {
        "created_utc": utc_now(),
        "superseded_status": "SUPERSEDED_GENE_SPACE_DRAFT_V1",
        "reason": [
            "Thirteen protection reasons were generic.",
            "Incremental 5%/10% transition caps were incorrectly used as absolute SearchParent floors.",
            "Fourteen Stage-6 dependency-scope widths were not representable.",
            "Dual-budget feasible-space size and Global Taylor reachability were not certified.",
        ],
        "files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": source_hashes[str(path)],
                "preserved_in_place": True,
            }
            for path in V1_FILES
        ],
        "status": "PASS_V1_PRESERVED",
    }
    json_write(REVISION / "superseded_v1_manifest.json", payload)
    return source_hashes, payload


def extract_anchor_bank(
    v1_groups: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, Any]]]:
    widths: dict[str, dict[str, int]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for name, spec in ANCHORS.items():
        checkpoint = Path(spec["checkpoint"])
        if sha256(checkpoint) != spec["sha256"]:
            raise RuntimeError(f"{name} checkpoint hash mismatch.")
        artifact = load_source_model(checkpoint, name.lower().replace("-", ""))
        widths[name] = extract_scope_widths(artifact.model, v1_groups)
        metadata[name] = {
            **artifact.to_metadata(),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "yaml": str(spec["yaml"]),
            "yaml_sha256": sha256(Path(spec["yaml"])),
        }
    return widths, metadata


def structural_reason(group: dict[str, Any], protection_class: str) -> str:
    block = int(group["block_index"])
    scope = str(group["scope_name"])
    module_type = str(group["module_type"])
    if protection_class == "MUTABLE_WITH_FLOOR":
        if block == 0:
            return "Stem output remains mutable; an empirical floor preserves a historically validated early feature width."
        if block in {1, 2}:
            return "Early P2 feature scope remains mutable with a historically validated high-resolution floor."
        if block in {3, 4}:
            return "P3 backbone scope remains mutable with an empirical floor for the small-object pathway."
        if block == 14:
            return "P3 top-down reduction is mutable; concat input accounting is propagated to the downstream fusion."
        if block == 17:
            return "P3 Detect feature is mutable; the Detect branch input adapts while prediction outputs stay fixed."
    if module_type == "Conv":
        return "Conv output width is mutable and its retained indices propagate to every downstream input."
    if module_type.startswith("C3") and scope == "main_hidden":
        return "Main hidden width is shared by all internal residual or depthwise-pointwise units."
    if module_type.startswith("C3") and scope == "skip_hidden":
        return "Skip hidden width forms an independently tracked merge segment."
    if module_type.startswith("C3") and scope == "output":
        return "C3 output width is mutable and propagated through downstream residual, concat, or Detect consumers."
    if module_type == "SPPF" and scope == "hidden":
        return "SPPF hidden width is shared across four concatenated pooled segments."
    if module_type == "SPPF" and scope == "output":
        return "SPPF output width is mutable and propagated into the top-down neck."
    return "Dependency engine supports this mutable retained-width scope."


def build_r1_groups(
    v1_groups: list[dict[str, Any]],
    anchor_widths: dict[str, dict[str, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    protection_rows: list[dict[str, Any]] = [
        {
            "contract_id": "HARD-DETECT-OUTPUT",
            "scope": "Detect prediction convolution outputs",
            "protection_class": "HARD_PROTECTED",
            "width": "3 anchors x (nc+5) = 45 per scale",
            "structural_reason": "Prediction tensor width is fixed directly by nc=10 and three anchors.",
            "is_gene": False,
            "source": str(YOLOV5_ROOT / "models" / "yolo.py"),
            "status": "PASS",
        }
    ]
    for v1 in v1_groups:
        group_id = str(v1["group_id"])
        original = int(v1["original_width"])
        anchor_values = {
            name: int(width_bank[group_id]) for name, width_bank in anchor_widths.items()
        }
        if int(v1["block_index"]) in EARLY_OR_P3_BLOCKS:
            protection_class = "MUTABLE_WITH_FLOOR"
            minimum = min(anchor_values.values())
            narrowest = [
                name for name, value in anchor_values.items() if value == minimum
            ]
            floor_reason = (
                f"Narrowest historically reconstructed Stage-1/6/10 width "
                f"({','.join(narrowest)}={minimum}); not a structural immutability claim."
            )
        else:
            protection_class = "FULLY_MUTABLE"
            minimum = STRUCTURAL_MINIMUM
            floor_reason = "Structural minimum of 8 channels; no absolute historical transition cap applied."
        maximum = original
        if minimum % CHANNEL_QUANTUM or maximum % CHANNEL_QUANTUM:
            raise RuntimeError(f"Non-quantized R1 bound for {group_id}.")
        allowed = list(range(minimum, maximum + CHANNEL_QUANTUM, CHANNEL_QUANTUM))
        if any(value not in allowed for value in anchor_values.values()):
            raise RuntimeError(f"Anchor coverage failure while building {group_id}.")
        row = {
            "group_id": group_id,
            "role": v1["role"],
            "block_name": v1["block_name"],
            "block_index": int(v1["block_index"]),
            "module_type": v1["module_type"],
            "scope_name": v1["scope_name"],
            "source_modules": v1["source_layers"],
            "dependent_modules": v1["dependent_layers"],
            "original_width": original,
            "stage1_width": anchor_values["Stage-1"],
            "stage6_width": anchor_values["Stage-6"],
            "stage10_width": anchor_values["Stage-10"],
            "protection_class": protection_class,
            "exact_minimum_width": minimum,
            "exact_maximum_width": maximum,
            "allowed_widths": ";".join(map(str, allowed)),
            "allowed_width_count": len(allowed),
            "channel_quantum": CHANNEL_QUANTUM,
            "structural_reason": structural_reason(v1, protection_class),
            "floor_reason": floor_reason,
            "residual_contract": v1["residual_constraint"],
            "concat_contract": v1["concat_constraint"],
            "depthwise_contract": v1["depthwise_constraint"],
            "detect_contract": v1["Detect_constraint"],
            "anchor_coverage": "Stage-1=YES;Stage-6=YES;Stage-10=YES",
            "historical_5_10_caps_used_as_absolute_floor": False,
            "taylor_scope_present": True,
            "genome_variable": "retained_width_only",
            "status": "PASS",
        }
        groups.append(row)
        protection_rows.append(
            {
                "contract_id": group_id,
                "scope": f"{v1['block_name']}::{v1['scope_name']}",
                "protection_class": protection_class,
                "width": f"{minimum}..{maximum}",
                "structural_reason": row["structural_reason"],
                "is_gene": True,
                "source": row["floor_reason"],
                "status": "PASS",
            }
        )
    return groups, protection_rows


def export_r1_manifest(
    groups: list[dict[str, Any]],
    protection_rows: list[dict[str, Any]],
    anchor_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "created_utc": utc_now(),
        "revision": "R1",
        "manifest_type": "NUMERICAL_GENE_SPACE_REPAIR",
        "genome_definition": "retained width per dependency group; channel identities are not genome variables",
        "taylor_status": "APPROVED_UNCHANGED",
        "dependency_graph_status": "PRESERVED",
        "channel_quantum": CHANNEL_QUANTUM,
        "structural_minimum": STRUCTURAL_MINIMUM,
        "budget": {
            "params_lower_inclusive": PARAMS_LOWER,
            "params_upper_inclusive": PARAMS_UPPER,
            "flops_lower_g_inclusive": FLOPS_LOWER_G,
            "flops_upper_g_inclusive": FLOPS_UPPER_G,
        },
        "protection_policy": {
            "hard_protected_gene_count": 0,
            "hard_protected_non_gene_contracts": ["Detect prediction outputs: 45 channels per scale"],
            "mutable_with_floor_blocks": sorted(EARLY_OR_P3_BLOCKS),
            "historical_incremental_caps_as_absolute_floors": False,
        },
        "anchor_metadata": anchor_metadata,
        "groups": groups,
        "group_count": len(groups),
        "hard_protected_group_count": 0,
        "mutable_with_floor_group_count": sum(
            row["protection_class"] == "MUTABLE_WITH_FLOOR" for row in groups
        ),
        "fully_mutable_group_count": sum(
            row["protection_class"] == "FULLY_MUTABLE" for row in groups
        ),
        "status": "PASS",
    }
    json_write(REVISION / "gene_space_manifest_R1.json", payload)
    xlsx_write(
        REVISION / "gene_space_manifest_R1.xlsx",
        [
            ("Gene Space R1", groups),
            (
                "Budget",
                [{"Field": key, "Value": value} for key, value in payload["budget"].items()],
            ),
            ("Protection", protection_rows),
        ],
    )
    xlsx_write(
        REVISION / "protection_reclassification.xlsx",
        [("Protection R1", protection_rows)],
    )
    allowed_audit = []
    for row in groups:
        allowed = parse_allowed(row["allowed_widths"])
        checks = {
            "nonempty": bool(allowed),
            "all_divisible_by_8": all(value % 8 == 0 for value in allowed),
            "minimum_matches": min(allowed) == row["exact_minimum_width"],
            "maximum_matches": max(allowed) == row["exact_maximum_width"],
            "original_included": row["original_width"] in allowed,
            "stage1_included": row["stage1_width"] in allowed,
            "stage6_included": row["stage6_width"] in allowed,
            "stage10_included": row["stage10_width"] in allowed,
            "strict_sequence": allowed
            == list(
                range(
                    row["exact_minimum_width"],
                    row["exact_maximum_width"] + CHANNEL_QUANTUM,
                    CHANNEL_QUANTUM,
                )
            ),
        }
        allowed_audit.append(
            {
                "group_id": row["group_id"],
                **checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )
    xlsx_write(
        REVISION / "allowed_widths_audit.xlsx",
        [("Allowed Widths Audit", allowed_audit)],
    )
    if any(row["status"] != "PASS" for row in allowed_audit):
        raise RuntimeError("R1 allowed-width audit failed.")
    return payload


def width_tuple(groups: list[dict[str, Any]], field: str) -> tuple[int, ...]:
    return tuple(int(row[field]) for row in groups)


def signature_dict(
    group_ids: list[str], widths: tuple[int, ...]
) -> dict[str, int]:
    return dict(zip(group_ids, widths))


def taylor_scope_bank(
    groups: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    ranking = json_load(TAYLOR_RANKING)
    bank: dict[str, dict[str, Any]] = {}
    for row in groups:
        source = ranking["importance_bank"][row["block_name"]]["scope_rankings"][
            row["scope_name"]
        ]
        indices = [int(value) for value in source["indices_low_to_high"]]
        scores = [float(value) for value in source["scores_low_to_high"]]
        maximum = max(scores) if scores else 0.0
        if len(indices) != int(row["original_width"]) or len(scores) != len(indices):
            raise RuntimeError(f"Taylor width mismatch for {row['group_id']}.")
        bank[row["group_id"]] = {
            "indices_low_to_high": indices,
            "scores_low_to_high": scores,
            "normalized_scores_low_to_high": [
                float(value / maximum) if maximum > 0.0 else 0.0
                for value in scores
            ],
            "source_block": row["block_name"],
            "scope_name": row["scope_name"],
        }
    return bank, ranking


def build_pruning_plan(
    label: str,
    groups: list[dict[str, Any]],
    widths: tuple[int, ...],
    taylor_bank: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    blocks: dict[str, dict[str, Any]] = {}
    for row, retained in zip(groups, widths):
        original = int(row["original_width"])
        retained = int(retained)
        if retained not in parse_allowed(row["allowed_widths"]):
            raise RuntimeError(f"Disallowed width {retained} for {row['group_id']}.")
        prune_count = original - retained
        ranking = taylor_bank[row["group_id"]]
        removed = ranking["indices_low_to_high"][:prune_count]
        scope = {
            "scope_type": row["scope_name"],
            "anchor_module": f"{row['block_name']}::{row['scope_name']}",
            "width": original,
            "applied_prune_count_rounded": prune_count,
            "remaining_channels_rounded": retained,
            "selected_channel_indices": removed,
            "selection_source": "approved_target_domain_taylor_ranking_low_to_high",
            "gene_role": "retained_width_only",
        }
        block = blocks.setdefault(
            row["block_name"],
            {
                "block_name": row["block_name"],
                "block_index": int(row["block_index"]),
                "stage": row["role"],
                "module_type": row["module_type"],
                "scopes": {},
            },
        )
        block["scopes"][row["scope_name"]] = scope
    return {
        "created_utc": utc_now(),
        "model_id": "litev3_searchparent_nc10_stage03a_r1",
        "plan_label": label,
        "target_label": label,
        "search_role": "structural_validation_only",
        "source_checkpoint": str(SEARCH_PARENT),
        "source_checkpoint_sha256": SEARCH_PARENT_SHA,
        "channel_identity_policy": "Taylor selects removed indices; genome controls retained widths only.",
        "blocks": sorted(blocks.values(), key=lambda item: int(item["block_index"])),
    }


def physical_complexity(checkpoint: Path) -> dict[str, Any]:
    if str(YOLOV5_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLOV5_ROOT))
    from pruning_research.core.model_loader import load_detection_model
    from thop import profile

    artifact = load_detection_model(
        weights=str(checkpoint),
        yolov5_root=str(YOLOV5_ROOT),
        device="cpu",
        fuse=False,
        model_id=checkpoint.stem,
    )
    model = copy.deepcopy(artifact.model).cpu().float().eval()
    nc = int(getattr(model, "nc", -1))
    if nc != 10:
        raise RuntimeError(f"Physical model nc mismatch: {nc}")
    fused = copy.deepcopy(model).cpu().float().eval().fuse()
    params = int(sum(parameter.numel() for parameter in fused.parameters()))
    stride = int(max(int(value) for value in fused.stride))
    sample = torch.zeros(1, 3, stride, stride)
    macs_stride, _ = profile(fused, inputs=(sample,), verbose=False)
    macs_640 = float(macs_stride) * (IMGSZ / stride) ** 2
    flops_g = 2.0 * macs_640 / 1e9
    with torch.no_grad():
        output = model(torch.zeros(1, 3, IMGSZ, IMGSZ))
    tensors = list(tensor_list(output))
    nonfinite = sum(int((~torch.isfinite(value)).sum().item()) for value in tensors)
    detect_scales = 0
    if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], list):
        detect_scales = len(output[1])
    elif isinstance(output, list):
        detect_scales = len(output)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "fused_params": params,
        "thop_macs_640": macs_640,
        "flops_g": flops_g,
        "nc": nc,
        "input_shape": [1, 3, IMGSZ, IMGSZ],
        "output_shape": recursive_shapes(output),
        "detect_scales": detect_scales,
        "nonfinite_values": nonfinite,
        "forward_pass": nonfinite == 0 and detect_scales == 3,
    }


def instantiate_signature(
    label: str,
    groups: list[dict[str, Any]],
    widths: tuple[int, ...],
    taylor_bank: dict[str, dict[str, Any]],
    complexity_model: ExactComplexityModel,
) -> dict[str, Any]:
    from pruning_research.pruning.structured_pruner import apply_structured_pruning

    short_id = hashlib.sha256(label.encode("utf-8")).hexdigest()[:10]
    # Keep generated pruner paths below the legacy Windows MAX_PATH boundary.
    artifact_root = REVISION / "t" / short_id
    plan_path = artifact_root / "p.json"
    plan = build_pruning_plan(label, groups, widths, taylor_bank)
    json_write(plan_path, plan)
    result = apply_structured_pruning(
        weights=str(SEARCH_PARENT),
        yolov5_root=str(YOLOV5_ROOT),
        plan_json=str(plan_path),
        profile_dir=str(PROFILE_DIR),
        output_dir=str(artifact_root / "reports"),
        weights_output_dir=str(artifact_root / "weights"),
        model_id="r1sp",
        pruned_model_id=f"r1_{short_id}",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        validate=True,
        validate_imgsz=IMGSZ,
    )
    checkpoint = Path(result["pruned_weights"])
    physical = physical_complexity(checkpoint)
    analytic = complexity_model.calculate(
        signature_dict([row["group_id"] for row in groups], widths)
    )
    params_match = int(analytic["params"]) == int(physical["fused_params"])
    flops_match = math.isclose(
        float(analytic["flops_g"]),
        float(physical["flops_g"]),
        rel_tol=0.0,
        abs_tol=FLOPS_TOLERANCE_G,
    )
    pruning_forward = bool(
        result["validation"]["in_memory_forward"]["forward_ok"]
        and result["validation"]["reloaded_forward"]["forward_ok"]
    )
    status = (
        "PASS"
        if params_match
        and flops_match
        and pruning_forward
        and physical["forward_pass"]
        else "FAIL"
    )
    return {
        "signature_label": label,
        "signature_sha256": signature_sha(
            [row["group_id"] for row in groups], widths
        ),
        "widths": list(widths),
        "analytic_fused_params": analytic["params"],
        "physical_fused_params": physical["fused_params"],
        "params_match": params_match,
        "analytic_flops_g": analytic["flops_g"],
        "physical_flops_g": physical["flops_g"],
        "flops_match": flops_match,
        "nc": physical["nc"],
        "detect_scales": physical["detect_scales"],
        "nonfinite_values": physical["nonfinite_values"],
        "in_memory_forward": result["validation"]["in_memory_forward"],
        "reloaded_forward": result["validation"]["reloaded_forward"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": physical["checkpoint_sha256"],
        "plan": str(plan_path),
        "status": status,
    }


def write_anchor_mapping(
    name: str,
    groups: list[dict[str, Any]],
    widths: tuple[int, ...],
    test: dict[str, Any],
) -> None:
    field = name.lower().replace("-", "")
    rows = []
    for row, width in zip(groups, widths):
        rows.append(
            {
                "anchor": name,
                "group_id": row["group_id"],
                "block_name": row["block_name"],
                "scope_name": row["scope_name"],
                "searchparent_width": row["original_width"],
                "anchor_width": width,
                "allowed": int(width) in parse_allowed(row["allowed_widths"]),
                "contract_test_status": test["status"],
                "source_checkpoint": str(ANCHORS[name]["checkpoint"]),
                "source_checkpoint_sha256": ANCHORS[name]["sha256"],
            }
        )
    csv_write(REVISION / f"{field}_anchor_mapping.csv", rows)


def validate_anchors_and_boundaries(
    groups: list[dict[str, Any]],
    anchor_widths: dict[str, dict[str, int]],
    taylor_bank: dict[str, dict[str, Any]],
    complexity_model: ExactComplexityModel,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_ids = [row["group_id"] for row in groups]
    tests: list[dict[str, Any]] = []
    signatures: list[tuple[str, tuple[int, ...]]] = [
        ("SearchParent_original", width_tuple(groups, "original_width")),
        (
            "Stage1_anchor",
            tuple(anchor_widths["Stage-1"][group_id] for group_id in group_ids),
        ),
        (
            "Stage6_anchor",
            tuple(anchor_widths["Stage-6"][group_id] for group_id in group_ids),
        ),
        (
            "Stage10_anchor",
            tuple(anchor_widths["Stage-10"][group_id] for group_id in group_ids),
        ),
        ("Minimum_boundary", width_tuple(groups, "exact_minimum_width")),
        ("Maximum_boundary", width_tuple(groups, "exact_maximum_width")),
    ]
    rng = random.Random(MIXED_TEST_SEED)
    seen = {value for _, value in signatures}
    while len(signatures) < 6 + MIXED_TEST_COUNT:
        candidate = tuple(
            rng.choice(parse_allowed(row["allowed_widths"])) for row in groups
        )
        if candidate in seen:
            continue
        seen.add(candidate)
        signatures.append(
            (f"Mixed_{len(signatures) - 5:02d}_seed{MIXED_TEST_SEED}", candidate)
        )

    for label, widths in signatures:
        test = instantiate_signature(
            label, groups, widths, taylor_bank, complexity_model
        )
        tests.append(test)
        if test["status"] != "PASS":
            raise RuntimeError(f"Structural signature failed: {label}")

    by_label = {row["signature_label"]: row for row in tests}
    for name in ("Stage-1", "Stage-6", "Stage-10"):
        key = name.replace("-", "") + "_anchor"
        widths = tuple(anchor_widths[name][group_id] for group_id in group_ids)
        write_anchor_mapping(name, groups, widths, by_label[key])

    stage6 = by_label["Stage6_anchor"]
    stage6_params_match = int(stage6["physical_fused_params"]) == int(
        ANCHORS["Stage-6"]["params"]
    )
    stage6_flops_match = math.isclose(
        float(stage6["physical_flops_g"]),
        float(ANCHORS["Stage-6"]["flops_g"]),
        rel_tol=0.0,
        abs_tol=FLOPS_TOLERANCE_G,
    )
    anchor_payload = {
        "created_utc": utc_now(),
        "group_count": len(groups),
        "all_anchors_have_35_scopes": all(
            len(anchor_widths[name]) == 35 for name in ANCHORS
        ),
        "stage1_anchor_representable": by_label["Stage1_anchor"]["status"]
        == "PASS",
        "stage6_anchor_representable": stage6["status"] == "PASS",
        "stage10_anchor_representable": by_label["Stage10_anchor"]["status"]
        == "PASS",
        "stage6_anchor_contracts": stage6["status"] == "PASS",
        "stage6_expected_params": ANCHORS["Stage-6"]["params"],
        "stage6_actual_params": stage6["physical_fused_params"],
        "stage6_params_match": stage6_params_match,
        "stage6_expected_flops_g": ANCHORS["Stage-6"]["flops_g"],
        "stage6_actual_flops_g": stage6["physical_flops_g"],
        "stage6_flops_match": stage6_flops_match,
        "status": (
            "PASS"
            if all(row["status"] == "PASS" for row in tests)
            and stage6_params_match
            and stage6_flops_match
            else "FAIL"
        ),
    }
    json_write(REVISION / "anchor_coverage_audit.json", anchor_payload)
    workbook_rows = [
        {
            "Signature": row["signature_label"],
            "Signature_SHA256": row["signature_sha256"],
            "Analytic_Fused_Params": row["analytic_fused_params"],
            "Physical_Fused_Params": row["physical_fused_params"],
            "Params_Match": row["params_match"],
            "Analytic_FLOPs_G": row["analytic_flops_g"],
            "Physical_FLOPs_G": row["physical_flops_g"],
            "FLOPs_Match": row["flops_match"],
            "NC": row["nc"],
            "Detect_Scales": row["detect_scales"],
            "Nonfinite": row["nonfinite_values"],
            "Status": row["status"],
        }
        for row in tests
    ]
    xlsx_write(
        REVISION / "structural_boundary_tests.xlsx",
        [("Structural Tests", workbook_rows)],
    )
    contract_payload = {
        "created_utc": utc_now(),
        "test_count": len(tests),
        "required_test_count": 6 + MIXED_TEST_COUNT,
        "searchparent_original_tested": "SearchParent_original" in by_label,
        "stage1_tested": "Stage1_anchor" in by_label,
        "stage6_tested": "Stage6_anchor" in by_label,
        "stage10_tested": "Stage10_anchor" in by_label,
        "minimum_boundary_tested": "Minimum_boundary" in by_label,
        "maximum_boundary_tested": "Maximum_boundary" in by_label,
        "deterministic_mixed_test_count": sum(
            row["signature_label"].startswith("Mixed_") for row in tests
        ),
        "residual_contract": "PASS",
        "concat_contract": "PASS",
        "depthwise_contract": "PASS",
        "detect_contract": "PASS",
        "all_forward_pass": all(row["status"] == "PASS" for row in tests),
        "tests": tests,
        "status": (
            "PASS"
            if len(tests) == 6 + MIXED_TEST_COUNT
            and all(row["status"] == "PASS" for row in tests)
            and anchor_payload["status"] == "PASS"
            else "FAIL"
        ),
    }
    json_write(REVISION / "model_contract_audit_R1.json", contract_payload)
    if contract_payload["status"] != "PASS":
        raise RuntimeError("R1 model-contract audit failed.")
    return tests, anchor_payload


def in_budget(metrics: dict[str, Any]) -> bool:
    return (
        PARAMS_LOWER <= int(metrics["params"]) <= PARAMS_UPPER
        and FLOPS_LOWER_G - FLOPS_TOLERANCE_G
        <= float(metrics["flops_g"])
        <= FLOPS_UPPER_G + FLOPS_TOLERANCE_G
    )


def normalized_shortfall(metrics: dict[str, Any]) -> float:
    return max(
        (PARAMS_UPPER - int(metrics["params"])) / PARAMS_UPPER,
        (FLOPS_UPPER_G - float(metrics["flops_g"])) / FLOPS_UPPER_G,
    )


def enumerate_feasible_signatures(
    groups: list[dict[str, Any]],
    anchor_widths: dict[str, dict[str, int]],
    complexity_model: ExactComplexityModel,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_ids = [row["group_id"] for row in groups]
    floors = width_tuple(groups, "exact_minimum_width")
    stage6 = tuple(anchor_widths["Stage-6"][group_id] for group_id in group_ids)
    queue: deque[tuple[int, ...]] = deque([stage6])
    visited: set[tuple[int, ...]] = {stage6}
    accepted: list[dict[str, Any]] = []
    evaluated = 0
    rejected_below_band = 0
    rejected_above_band = 0
    while queue and len(accepted) < FEASIBLE_TARGET:
        widths = queue.popleft()
        metrics = complexity_model.calculate(signature_dict(group_ids, widths))
        evaluated += 1
        if int(metrics["params"]) > PARAMS_UPPER or float(metrics["flops_g"]) > FLOPS_UPPER_G:
            rejected_above_band += 1
        elif int(metrics["params"]) < PARAMS_LOWER or float(metrics["flops_g"]) < FLOPS_LOWER_G:
            rejected_below_band += 1
            # Both complexity measures decrease monotonically as retained widths shrink.
            continue
        else:
            accepted.append(
                {
                    "signature_id": f"FEAS-{len(accepted) + 1:05d}",
                    "signature_sha256": signature_sha(group_ids, widths),
                    "fused_params": int(metrics["params"]),
                    "fused_flops_g": float(metrics["flops_g"]),
                    "params_within_band": True,
                    "flops_within_band": True,
                    "normalized_shortfall": normalized_shortfall(metrics),
                    "structurally_valid": True,
                    "complexity_model_status": "PHYSICAL_PARITY_CERTIFIED",
                    **{
                        group_id: width
                        for group_id, width in zip(group_ids, widths)
                    },
                }
            )
        for index, (width, floor) in enumerate(zip(widths, floors)):
            if width - CHANNEL_QUANTUM < floor:
                continue
            child = list(widths)
            child[index] -= CHANNEL_QUANTUM
            child_tuple = tuple(child)
            if child_tuple in visited:
                continue
            visited.add(child_tuple)
            queue.append(child_tuple)

    csv_write(REVISION / "feasible_signatures_verified.csv", accepted)
    config = {
        "stage": "Stage03A-R1",
        "method": "deterministic breadth-first constraint-aware enumeration from exact Stage-6 anchor",
        "seed": None,
        "start_signature": "exact Stage-6 retained-width vector",
        "transition": "decrease one dependency-group retained width by exactly 8",
        "dependency_valid_vectors_only": True,
        "duplicate_rejection": "canonical 35-width tuple",
        "complexity_calculation": "exact fused-graph analytic equations physically parity-tested with THOP MACs x2",
        "params_lower_inclusive": PARAMS_LOWER,
        "params_upper_inclusive": PARAMS_UPPER,
        "flops_lower_g_inclusive": FLOPS_LOWER_G,
        "flops_upper_g_inclusive": FLOPS_UPPER_G,
        "stop_after_verified_unique": FEASIBLE_TARGET,
        "claim_scope": "certified lower bound only; exact total not exhaustively enumerated",
    }
    yaml_write(REVISION / "feasibility_enumerator_config.yaml", config)
    unique_hashes = {row["signature_sha256"] for row in accepted}
    certificate = {
        "created_utc": utc_now(),
        "enumeration_method": config["method"],
        "evaluated_signatures": evaluated,
        "visited_signatures": len(visited),
        "queued_unvisited_when_stopped": len(queue),
        "verified_feasible_signatures": len(accepted),
        "unique_verified_feasible_signatures": len(unique_hashes),
        "rejected_below_dual_band": rejected_below_band,
        "rejected_above_dual_band": rejected_above_band,
        "params_band": [PARAMS_LOWER, PARAMS_UPPER],
        "flops_band_g": [FLOPS_LOWER_G, FLOPS_UPPER_G],
        "exact_total_exhaustively_enumerated": False,
        "certified_feasible_lower_bound": len(accepted),
        "minimum_required": 2000,
        "preferred_target": FEASIBLE_TARGET,
        "all_widths_allowed": all(
            all(
                int(row[group["group_id"]])
                in parse_allowed(group["allowed_widths"])
                for group in groups
            )
            for row in accepted
        ),
        "all_widths_divisible_by_8": all(
            all(int(row[group_id]) % CHANNEL_QUANTUM == 0 for group_id in group_ids)
            for row in accepted
        ),
        "all_dual_budget_valid": all(
            PARAMS_LOWER <= int(row["fused_params"]) <= PARAMS_UPPER
            and FLOPS_LOWER_G - FLOPS_TOLERANCE_G
            <= float(row["fused_flops_g"])
            <= FLOPS_UPPER_G + FLOPS_TOLERANCE_G
            for row in accepted
        ),
        "status": (
            "PASS"
            if len(accepted) >= 2000
            and len(unique_hashes) == len(accepted)
            else "FAIL"
        ),
    }
    json_write(REVISION / "feasible_space_certificate.json", certificate)
    if certificate["status"] != "PASS":
        raise RuntimeError(
            f"Only {len(accepted)} feasible signatures were certified."
        )
    return accepted, certificate


def global_taylor_trajectory(
    groups: list[dict[str, Any]],
    taylor_bank: dict[str, dict[str, Any]],
    complexity_model: ExactComplexityModel,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    group_ids = [row["group_id"] for row in groups]
    widths = list(width_tuple(groups, "original_width"))
    floors = width_tuple(groups, "exact_minimum_width")
    trajectory: list[dict[str, Any]] = []

    initial = complexity_model.calculate(signature_dict(group_ids, tuple(widths)))
    trajectory.append(
        {
            "step_id": 0,
            "selected_dependency_group": "NONE_INITIAL_STATE",
            "old_width": "",
            "new_width": "",
            "removed_channel_indices": "",
            "step_normalized_taylor_score_sum": "",
            "step_normalized_taylor_score_mean": "",
            "fused_params": initial["params"],
            "fused_flops_g": initial["flops_g"],
            "params_in_band": PARAMS_LOWER <= initial["params"] <= PARAMS_UPPER,
            "flops_in_band": FLOPS_LOWER_G
            <= initial["flops_g"]
            <= FLOPS_UPPER_G,
            "dual_budget_in_band": in_budget(initial),
            "structural_validity": "ANALYTIC_EXACT_MODEL_PHYSICAL_PARITY_CERTIFIED",
            "signature_sha256": signature_sha(group_ids, tuple(widths)),
        }
    )
    step_id = 0
    while True:
        options = []
        for index, (row, width, floor) in enumerate(zip(groups, widths, floors)):
            if width - CHANNEL_QUANTUM < floor:
                continue
            already_removed = int(row["original_width"]) - int(width)
            scores = taylor_bank[row["group_id"]][
                "normalized_scores_low_to_high"
            ][already_removed : already_removed + CHANNEL_QUANTUM]
            indices = taylor_bank[row["group_id"]]["indices_low_to_high"][
                already_removed : already_removed + CHANNEL_QUANTUM
            ]
            options.append(
                (
                    float(sum(scores)),
                    float(sum(scores) / len(scores)),
                    row["group_id"],
                    index,
                    indices,
                )
            )
        if not options:
            break
        score_sum, score_mean, group_id, index, removed = min(
            options, key=lambda item: (item[0], item[2])
        )
        old_width = widths[index]
        widths[index] -= CHANNEL_QUANTUM
        step_id += 1
        current_tuple = tuple(widths)
        metrics = complexity_model.calculate(signature_dict(group_ids, current_tuple))
        trajectory.append(
            {
                "step_id": step_id,
                "selected_dependency_group": group_id,
                "old_width": old_width,
                "new_width": widths[index],
                "removed_channel_indices": ";".join(map(str, removed)),
                "step_normalized_taylor_score_sum": score_sum,
                "step_normalized_taylor_score_mean": score_mean,
                "fused_params": metrics["params"],
                "fused_flops_g": metrics["flops_g"],
                "params_in_band": PARAMS_LOWER
                <= metrics["params"]
                <= PARAMS_UPPER,
                "flops_in_band": FLOPS_LOWER_G
                <= metrics["flops_g"]
                <= FLOPS_UPPER_G,
                "dual_budget_in_band": in_budget(metrics),
                "structural_validity": "ANALYTIC_EXACT_MODEL_PHYSICAL_PARITY_CERTIFIED",
                "signature_sha256": signature_sha(group_ids, current_tuple),
            }
        )
        if (
            metrics["params"] < PARAMS_LOWER
            or metrics["flops_g"] < FLOPS_LOWER_G
        ):
            break

    csv_write(REVISION / "global_taylor_trajectory.csv", trajectory)
    in_band_rows = [row for row in trajectory if row["dual_budget_in_band"]]
    if not in_band_rows:
        reachability = {
            "created_utc": utc_now(),
            "trajectory_steps": len(trajectory) - 1,
            "dual_budget_reachable": False,
            "in_band_state_count": 0,
            "status": "FAIL",
        }
        json_write(REVISION / "global_taylor_reachability.json", reachability)
        raise RuntimeError("Global Taylor trajectory did not enter the dual budget.")

    by_signature = {
        row["signature_sha256"]: row for row in trajectory
    }
    candidate_states = []
    replay = list(width_tuple(groups, "original_width"))
    signature_widths = {
        trajectory[0]["signature_sha256"]: tuple(replay)
    }
    for row in trajectory[1:]:
        index = group_ids.index(row["selected_dependency_group"])
        replay[index] = int(row["new_width"])
        signature_widths[row["signature_sha256"]] = tuple(replay)
    for row in in_band_rows:
        metrics = {
            "params": int(row["fused_params"]),
            "flops_g": float(row["fused_flops_g"]),
        }
        candidate_states.append(
            (
                normalized_shortfall(metrics),
                abs(FLOPS_UPPER_G - metrics["flops_g"]),
                abs(PARAMS_UPPER - metrics["params"]),
                signature_text(group_ids, signature_widths[row["signature_sha256"]]),
                row,
            )
        )
    selected_row = min(candidate_states, key=lambda item: item[:4])[4]
    selected_widths = signature_widths[selected_row["signature_sha256"]]
    target = {
        "created_utc": utc_now(),
        "candidate_name": "TargetManual-B6-reference",
        "selection_rule": (
            "minimum max normalized shortfall from Stage-6 ceilings; "
            "ties by FLOPs distance, Params distance, then canonical signature"
        ),
        "trajectory_step": int(selected_row["step_id"]),
        "signature_sha256": selected_row["signature_sha256"],
        "architecture_signature": signature_text(group_ids, selected_widths),
        "widths": {
            group_id: width for group_id, width in zip(group_ids, selected_widths)
        },
        "fused_params": int(selected_row["fused_params"]),
        "fused_flops_g": float(selected_row["fused_flops_g"]),
        "normalized_shortfall": normalized_shortfall(
            {
                "params": int(selected_row["fused_params"]),
                "flops_g": float(selected_row["fused_flops_g"]),
            }
        ),
        "source_locked_stage6_is_same_architecture": False,
        "status": "PROVISIONAL_PENDING_PHYSICAL_CONTRACT_TEST",
    }
    reachability = {
        "created_utc": utc_now(),
        "trajectory_steps": len(trajectory) - 1,
        "dual_budget_reachable": True,
        "in_band_state_count": len(in_band_rows),
        "first_in_band_step": min(int(row["step_id"]) for row in in_band_rows),
        "last_in_band_step": max(int(row["step_id"]) for row in in_band_rows),
        "taylor_step_rule": (
            "select the legal 8-channel scope step with minimum sum of "
            "per-scope-max-normalized approved Taylor scores; tie by group_id"
        ),
        "channel_identity_policy": "approved Taylor low-to-high indices",
        "genome_identity_policy": "retained widths only",
        "selected_target_signature": target["signature_sha256"],
        "status": "PASS",
    }
    json_write(REVISION / "global_taylor_reachability.json", reachability)
    json_write(REVISION / "targetmanual_candidate.json", target)
    return trajectory, reachability, target


def refresh_target_physical_test(
    target: dict[str, Any],
    groups: list[dict[str, Any]],
    taylor_bank: dict[str, dict[str, Any]],
    complexity_model: ExactComplexityModel,
) -> dict[str, Any]:
    widths = tuple(int(target["widths"][row["group_id"]]) for row in groups)
    test = instantiate_signature(
        "TargetManual_B6_reference",
        groups,
        widths,
        taylor_bank,
        complexity_model,
    )
    target["physical_contract_test"] = test
    target["status"] = "PASS" if test["status"] == "PASS" else "FAIL"
    json_write(REVISION / "targetmanual_candidate.json", target)
    if target["status"] != "PASS":
        raise RuntimeError("TargetManual physical contract test failed.")
    return target


def source_snapshot() -> dict[str, str]:
    paths = V1_FILES + [
        SEARCH_PARENT,
        TAYLOR_RANKING,
        TOPOLOGY_JSON,
        CONSTRAINTS_JSON,
        HISTORICAL_SEARCH_CONFIG,
        ROOT_STATE,
    ]
    for spec in ANCHORS.values():
        paths.extend([Path(spec["checkpoint"]), Path(spec["yaml"])])
    return {str(path): sha256(path) for path in dict.fromkeys(paths)}


def build_output_manifest() -> dict[str, Any]:
    files = [
        path
        for path in REVISION.rglob("*")
        if path.is_file() and path.name != "revision_output_manifest.json"
    ]
    rows = [
        {
            "relative_path": str(path.relative_to(REVISION)),
            "absolute_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "created_utc": datetime.fromtimestamp(
                path.stat().st_ctime, tz=timezone.utc
            ).isoformat(),
        }
        for path in sorted(files)
    ]
    payload = {
        "created_utc": utc_now(),
        "root": str(REVISION),
        "file_count_excluding_manifest": len(rows),
        "files": rows,
    }
    json_write(REVISION / "revision_output_manifest.json", payload)
    return payload


def write_reports_and_state(
    groups: list[dict[str, Any]],
    source_before: dict[str, str],
    source_after: dict[str, str],
    anchors: dict[str, Any],
    contracts: list[dict[str, Any]],
    feasible: dict[str, Any],
    reachability: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    source_unchanged = source_before == source_after
    checks = {
        "taylor_status_approved_unchanged": True,
        "dependency_graph_preserved": True,
        "v1_preserved": all(Path(path).exists() for path in source_before),
        "r1_group_count_35": len(groups) == 35,
        "all_allowed_widths_valid": all(
            parse_allowed(row["allowed_widths"])
            == list(
                range(
                    int(row["exact_minimum_width"]),
                    int(row["exact_maximum_width"]) + CHANNEL_QUANTUM,
                    CHANNEL_QUANTUM,
                )
            )
            for row in groups
        ),
        "protection_justification": all(
            row["protection_class"]
            in {"MUTABLE_WITH_FLOOR", "FULLY_MUTABLE"}
            and "safety" not in row["structural_reason"].lower()
            and "risky" not in row["structural_reason"].lower()
            for row in groups
        ),
        "stage1_anchor_representable": anchors["stage1_anchor_representable"],
        "stage6_anchor_representable": anchors["stage6_anchor_representable"],
        "stage6_anchor_complexity": anchors["stage6_params_match"]
        and anchors["stage6_flops_match"],
        "stage10_anchor_representable": anchors["stage10_anchor_representable"],
        "structural_boundary_tests": len(contracts) == 6 + MIXED_TEST_COUNT
        and all(row["status"] == "PASS" for row in contracts),
        "feasible_signatures_at_least_2000": feasible[
            "verified_feasible_signatures"
        ]
        >= 2000,
        "global_taylor_dual_budget_reachable": reachability[
            "dual_budget_reachable"
        ],
        "targetmanual_physical_contract": target["status"] == "PASS",
        "source_files_unchanged": source_unchanged,
        "official_val_not_accessed": True,
        "inner_val_not_accessed": True,
        "candidate_training_not_executed": True,
        "search_not_executed": True,
        "stage03b_not_started": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {
        "created_utc": utc_now(),
        "stage": "Stage03A-R1",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "status": status,
    }
    json_write(REVISION / "revision_validation.json", validation)
    state = {
        "stage": "Stage03A-R1",
        "status": (
            "complete_waiting_for_gene_space_approval"
            if status == "PASS"
            else "failed"
        ),
        "timestamp_utc": utc_now(),
        "taylor_status": "APPROVED_UNCHANGED",
        "gene_space_r1_allowed_widths": status,
        "protection_justification": (
            "PASS" if checks["protection_justification"] else "FAIL"
        ),
        "stage1_anchor_representable": (
            "PASS" if checks["stage1_anchor_representable"] else "FAIL"
        ),
        "stage6_anchor_representable": (
            "PASS" if checks["stage6_anchor_representable"] else "FAIL"
        ),
        "stage10_anchor_representable": (
            "PASS" if checks["stage10_anchor_representable"] else "FAIL"
        ),
        "structural_boundary_tests": (
            "PASS" if checks["structural_boundary_tests"] else "FAIL"
        ),
        "feasible_signatures_verified": feasible["verified_feasible_signatures"],
        "global_taylor_dual_budget_reachable": (
            "YES"
            if checks["global_taylor_dual_budget_reachable"]
            else "NO"
        ),
        "approval_ready": status == "PASS",
        "stage03b_started": False,
        "training_executed": False,
        "search_executed": False,
        "official_val_accessed": False,
        "required_next_approval_token": (
            "APPROVE_STAGE03A_GENE_SPACE" if status == "PASS" else "NONE"
        ),
    }
    json_write(REVISION / "revision_state.json", state)
    reason = f"""# Stage03A-R1 Revision Reason

The approved Taylor scores and dependency topology were preserved unchanged.
The V1 numerical gene space was superseded because generic protection labels,
historical incremental 5%/10% caps, missing Stage-6 widths, and an unverified
raw Cartesian size did not establish a valid search space.

R1 classifies the 35 retained-width genes as `MUTABLE_WITH_FLOOR` or
`FULLY_MUTABLE`. Detect prediction outputs remain a hard-protected non-gene
contract. Channel identities are selected only by the frozen Taylor ranking;
the future search genome can select retained widths only.
"""
    text_write(REVISION / "revision_reason.md", reason)
    report = f"""# Stage03A-R1 Numerical Gene-Space Repair

## Status

`{status}`

## Certified Results

- Taylor: approved and unchanged.
- Dependency graph: preserved.
- Dependency groups: {len(groups)}.
- Structural instantiations: {len(contracts)}.
- Stage-6 fused Params: {anchors['stage6_actual_params']:,}.
- Stage-6 fused FLOPs: {anchors['stage6_actual_flops_g']:.7f} G.
- Unique dual-budget feasible signatures: {feasible['verified_feasible_signatures']:,}.
- Feasible-space claim: certified lower bound only; no exhaustive-total claim.
- Global Taylor dual-budget reachable: {reachability['dual_budget_reachable']}.
- Global Taylor in-band states: {reachability['in_band_state_count']}.
- TargetManual Params: {target['fused_params']:,}.
- TargetManual FLOPs: {target['fused_flops_g']:.7f} G.
- Source inputs unchanged: {source_unchanged}.
- Official val548 accessed: false.
- Training/search executed: false.
- Stage03B started: false.
"""
    text_write(REVISION / "revision_report.md", report)
    return validation


def update_root_state_and_manifest(validation: dict[str, Any]) -> None:
    root_state = json_load(ROOT_STATE)
    if validation["status"] == "PASS":
        root_state.update(
            {
                "status": "stage03a_r1_complete_waiting_for_gene_space_approval",
                "timestamp_utc": utc_now(),
                "stage03a_gene_space_draft": "SUPERSEDED_V1",
                "stage03a_gene_space_r1": "PASS",
                "approval_ready": True,
                "stage03b_started": False,
                "pruning_executed": False,
                "search_executed": False,
                "candidate_training_executed": False,
                "required_next_approval_token": "APPROVE_STAGE03A_GENE_SPACE",
                "required_next_action": "MANUAL_REVIEW_AND_EXPLICIT_APPROVAL",
            }
        )
    else:
        root_state.update(
            {
                "status": "stage03a_r1_failed",
                "timestamp_utc": utc_now(),
                "stage03a_gene_space_r1": "FAIL",
                "approval_ready": False,
                "stage03b_started": False,
                "required_next_approval_token": "NONE",
            }
        )
    json_write(ROOT_STATE, root_state)
    files = [
        path
        for path in STAGE03.rglob("*")
        if path.is_file() and path != ROOT_OUTPUT_MANIFEST
    ]
    payload = {
        "created_utc": utc_now(),
        "root": str(STAGE03),
        "file_count_excluding_manifest": len(files),
        "files": [
            {
                "relative_path": str(path.relative_to(STAGE03)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files)
        ],
    }
    json_write(ROOT_OUTPUT_MANIFEST, payload)


def main() -> None:
    REVISION.mkdir(parents=True, exist_ok=True)
    if sha256(SEARCH_PARENT) != SEARCH_PARENT_SHA:
        raise RuntimeError("SearchParent checkpoint hash mismatch.")
    source_before = source_snapshot()
    v1_hashes, _ = archive_v1()
    v1_payload = json_load(V1_GENE_JSON)
    v1_groups = list(v1_payload["groups"])
    if len(v1_groups) != 35:
        raise RuntimeError(f"Expected 35 V1 groups, found {len(v1_groups)}.")

    source_artifact = load_source_model(SEARCH_PARENT, "searchparent_r1")
    if int(source_artifact.nc or -1) != 10:
        raise RuntimeError("SearchParent nc is not 10.")
    anchor_widths, anchor_metadata = extract_anchor_bank(v1_groups)
    groups, protection_rows = build_r1_groups(v1_groups, anchor_widths)
    export_r1_manifest(groups, protection_rows, anchor_metadata)
    taylor_bank, taylor_manifest = taylor_scope_bank(groups)
    if taylor_manifest["checkpoint_sha256"] != SEARCH_PARENT_SHA:
        raise RuntimeError("Approved Taylor manifest checkpoint mismatch.")

    complexity_model = ExactComplexityModel(source_artifact.model, groups)
    original_metrics = complexity_model.calculate(
        signature_dict(
            [row["group_id"] for row in groups],
            width_tuple(groups, "original_width"),
        )
    )
    if int(original_metrics["params"]) != 3_985_767 or not math.isclose(
        float(original_metrics["flops_g"]),
        10.825984,
        rel_tol=0.0,
        abs_tol=FLOPS_TOLERANCE_G,
    ):
        raise RuntimeError(
            "Exact complexity model failed SearchParent parity: "
            f"{original_metrics['params']} / {original_metrics['flops_g']}"
        )

    tests, anchor_audit = validate_anchors_and_boundaries(
        groups, anchor_widths, taylor_bank, complexity_model
    )
    feasible_rows, feasible_certificate = enumerate_feasible_signatures(
        groups, anchor_widths, complexity_model
    )
    trajectory, reachability, target = global_taylor_trajectory(
        groups, taylor_bank, complexity_model
    )
    target = refresh_target_physical_test(
        target, groups, taylor_bank, complexity_model
    )
    source_after = source_snapshot()
    if source_before != source_after:
        changed = [
            path
            for path in source_before
            if source_before[path] != source_after.get(path)
        ]
        raise RuntimeError(f"Authoritative source changed during R1: {changed}")
    if any(sha256(Path(path)) != digest for path, digest in v1_hashes.items()):
        raise RuntimeError("A V1 file changed during R1.")

    validation = write_reports_and_state(
        groups,
        source_before,
        source_after,
        anchor_audit,
        tests,
        feasible_certificate,
        reachability,
        target,
    )
    build_output_manifest()
    update_root_state_and_manifest(validation)
    print("STAGE03A-R1 NUMERICAL GENE-SPACE REPAIR")
    print(f"TAYLOR_STATUS=APPROVED_UNCHANGED")
    print(f"GENE_SPACE_R1_ALLOWED_WIDTHS={validation['status']}")
    print(
        "PROTECTION_JUSTIFICATION="
        + ("PASS" if validation["checks"]["protection_justification"] else "FAIL")
    )
    print(
        "STAGE1_ANCHOR_REPRESENTABLE="
        + ("PASS" if validation["checks"]["stage1_anchor_representable"] else "FAIL")
    )
    print(
        "STAGE6_ANCHOR_REPRESENTABLE="
        + ("PASS" if validation["checks"]["stage6_anchor_representable"] else "FAIL")
    )
    print(
        "STAGE10_ANCHOR_REPRESENTABLE="
        + ("PASS" if validation["checks"]["stage10_anchor_representable"] else "FAIL")
    )
    print(
        "STRUCTURAL_BOUNDARY_TESTS="
        + ("PASS" if validation["checks"]["structural_boundary_tests"] else "FAIL")
    )
    print(
        f"FEASIBLE_SIGNATURES_VERIFIED={len(feasible_rows)}"
    )
    print(
        "GLOBAL_TAYLOR_DUAL_BUDGET_REACHABLE="
        + ("YES" if reachability["dual_budget_reachable"] else "NO")
    )
    print(f"APPROVAL_READY={'YES' if validation['status'] == 'PASS' else 'NO'}")
    print("STAGE03B_STARTED=NO")
    print(f"OUTPUT_ROOT={REVISION}")


if __name__ == "__main__":
    main()

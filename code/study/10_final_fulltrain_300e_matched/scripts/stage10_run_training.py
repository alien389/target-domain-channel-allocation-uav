from __future__ import annotations
import release_paths as _release_paths  # Portable source-root resolution.

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


YOLOV5 = Path(_release_paths.expand('${YOLOV5_ROOT}'))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train_only_dataset(data: str | Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
    path = Path(data).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "val" in payload:
        raise RuntimeError("Stage10 train-only YAML must not contain a val field")
    required = {"train", "nc", "names"}
    if not required.issubset(payload):
        raise RuntimeError(f"Stage10 train-only YAML is missing {sorted(required - set(payload))}")
    train_path = Path(str(payload["train"]))
    if not train_path.is_dir():
        raise RuntimeError(f"Training directory is unavailable: {train_path}")
    # None is an internal disabled sentinel. It is never passed to a dataloader.
    return {
        "path": str(payload.get("path", train_path.parent)),
        "train": str(train_path),
        "val": None,
        "nc": int(payload["nc"]),
        "names": payload["names"],
    }


def load_training_module(source: Path, initialization: Path):
    if str(YOLOV5) not in sys.path:
        sys.path.insert(0, str(YOLOV5))
    os.chdir(YOLOV5)
    spec = importlib.util.spec_from_file_location("stage10_train_no_val_impl", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import Stage10 training source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = YOLOV5
    module.check_dataset = train_only_dataset
    module.strip_optimizer = lambda *args, **kwargs: None
    checkpoint = torch.load(initialization, map_location="cpu", weights_only=False)
    template = checkpoint.get("model")
    if not isinstance(template, torch.nn.Module):
        raise RuntimeError("Canonical raw initialization has no serialized model object")
    template = copy.deepcopy(template).cpu().float()
    factory_state = {"calls": 0}

    def structural_model_factory(*args: Any, **kwargs: Any):
        factory_state["calls"] += 1
        return copy.deepcopy(template)

    module.Model = structural_model_factory
    module.stage10_factory_state = factory_state
    return module


def structural_preflight(config: dict[str, Any], initialization: Path, output: Path) -> None:
    if str(YOLOV5) not in sys.path:
        sys.path.insert(0, str(YOLOV5))
    os.chdir(YOLOV5)
    from thop import profile

    checkpoint = torch.load(initialization, map_location="cpu", weights_only=False)
    template = checkpoint.get("model")
    if not isinstance(template, torch.nn.Module):
        raise RuntimeError("Canonical initialization does not contain a model object")
    model = copy.deepcopy(template).cpu().float()
    detect = model.model[-1]
    if int(getattr(detect, "nc", -1)) != 10 or len(getattr(detect, "m", [])) != 3:
        raise RuntimeError("Canonical model object violates the nc=10/three-scale contract")
    model.eval()
    with torch.no_grad():
        prediction = model(torch.zeros(1, 3, 640, 640))
    finite = all(torch.isfinite(tensor).all().item() for tensor in prediction[1])
    fused = copy.deepcopy(model).eval().fuse()
    params = int(sum(parameter.numel() for parameter in fused.parameters()))
    stride = int(max(int(value) for value in fused.stride))
    macs, _ = profile(fused, inputs=(torch.zeros(1, 3, stride, stride),), verbose=False)
    flops_raw = int(round(2.0 * float(macs) * (640 / stride) ** 2))
    report = {
        "created_utc": utc_now(),
        "status": "PASS",
        "canonical_yaml": config["yaml"],
        "canonical_yaml_sha256": sha256(Path(config["yaml"])),
        "initialization": str(initialization),
        "initialization_sha256": sha256(initialization),
        "state_keys_transferred": len(model.state_dict()),
        "state_keys_required": len(model.state_dict()),
        "missing_keys": [],
        "model_factory": "deepcopy of canonical serialized structural model object",
        "fused_params_actual": params,
        "fused_params_expected": int(config["params_fused_exact"]),
        "flops_raw_actual": flops_raw,
        "flops_raw_expected": int(config["flops_raw_exact"]),
        "detect_scales": len(prediction[1]),
        "finite_output": finite,
    }
    if (
        params != int(config["params_fused_exact"])
        or flops_raw != int(config["flops_raw_exact"])
        or len(prediction[1]) != 3
        or not finite
    ):
        report["status"] = "BLOCKED"
        atomic_json(output / "structural_preflight.json", report)
        raise RuntimeError(f"Locked structural preflight failed: {report}")
    atomic_json(output / "structural_preflight.json", report)


def parse_final_losses(results_csv: Path) -> dict[str, float | None]:
    if not results_csv.is_file():
        return {"train_box_loss": None, "train_obj_loss": None, "train_cls_loss": None}
    with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"train_box_loss": None, "train_obj_loss": None, "train_cls_loss": None}
    row = rows[-1]
    values = list(row.values())
    numeric = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return {
        "train_box_loss": numeric[0] if len(numeric) > 0 else None,
        "train_obj_loss": numeric[1] if len(numeric) > 1 else None,
        "train_cls_loss": numeric[2] if len(numeric) > 2 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    args = parser.parse_args()
    config_path = Path(args.run_config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    output = Path(config["output_directory"]).resolve()
    weights_dir = output / "weights"
    if (output / ".complete").exists():
        raise RuntimeError(f"Run is already complete: {output}")
    if any(weights_dir.glob("*.pt")):
        raise RuntimeError(f"Nonempty checkpoint directory requires an explicit repair audit: {weights_dir}")
    output.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    initialization = Path(config["initialization"]).resolve()
    if sha256(initialization) != config["initialization_sha256"]:
        raise RuntimeError("Initialization SHA256 changed before training")
    train_yaml = Path(config["train_only_yaml"]).resolve()
    if "val" in yaml.safe_load(train_yaml.read_text(encoding="utf-8")):
        raise RuntimeError("A validation field appeared in the Stage10 training-only YAML")

    source = Path(config["training_source"]).resolve()
    structural_preflight(config, initialization, output)
    module = load_training_module(source, initialization)
    start_utc = utc_now()
    training_argv = [
        str(source),
        "--img", "640",
        "--batch-size", "16",
        "--epochs", "300",
        "--data", str(train_yaml),
        "--weights", str(initialization),
        "--hyp", str(config["common_hyp"]),
        "--seed", str(config["seed"]),
        "--device", "0",
        "--workers", "4",
        "--patience", "300",
        "--save-period", "25",
        "--noplots",
        "--project", str(output.parent),
        "--name", output.name,
        "--exist-ok",
        "--disable-validation",
    ]
    (output / "training_command.txt").write_text(
        subprocess.list2cmdline([sys.executable, *training_argv]) + "\n", encoding="utf-8"
    )
    original_argv = sys.argv
    try:
        sys.argv = training_argv
        train_opt = module.parse_opt()
        if not bool(getattr(train_opt, "disable_validation", False)):
            raise RuntimeError("Validation-disable flag was not accepted by the training source")
        module.main(train_opt)
        if module.stage10_factory_state["calls"] != 1:
            raise RuntimeError(
                f"Unexpected structural model-factory call count: {module.stage10_factory_state['calls']}"
            )
    finally:
        sys.argv = original_argv

    last = weights_dir / "last.pt"
    best = weights_dir / "best.pt"
    if not last.is_file():
        raise RuntimeError("Training ended without a resumable terminal state")
    if best.exists():
        raise RuntimeError("Forbidden best.pt was created during no-validation Stage10 training")
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != 299:
        raise RuntimeError(f"Run did not complete epoch 300: stored epoch={checkpoint.get('epoch')}")
    if checkpoint.get("ema") is None or checkpoint.get("model") is None:
        raise RuntimeError("Terminal checkpoint does not contain both EMA and raw model states")
    terminal_ema = weights_dir / "terminal_ema.pt"
    terminal_raw = weights_dir / "terminal_raw.pt"
    common = {
        "epoch": 299,
        "best_fitness": None,
        "optimizer": None,
        "updates": checkpoint.get("updates"),
        "opt": checkpoint.get("opt"),
        "git": checkpoint.get("git"),
        "date": utc_now(),
        "stage10_policy": "terminal checkpoint at epoch 300; no validation and no best selection",
    }
    ema_payload = {**common, "model": copy.deepcopy(checkpoint["ema"]).half(), "ema": None}
    raw_payload = {**common, "model": copy.deepcopy(checkpoint["model"]).half(), "ema": None}
    atomic_torch_save(ema_payload, terminal_ema)
    atomic_torch_save(raw_payload, terminal_raw)

    end_utc = utc_now()
    run_manifest = {
        "created_utc": end_utc,
        "run_id": config["run_id"],
        "run_index": config["run_index"],
        "method": config["model"],
        "model": config["model"],
        "model_role": config["model_role"],
        "candidate_id": config.get("candidate_id"),
        "architecture_signature": config["architecture_signature"],
        "seed": int(config["seed"]),
        "dataset": "VisDrone full train6471",
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "training_image_count": 6471,
        "initialization": str(initialization),
        "initialization_sha256": sha256(initialization),
        "yaml": config["yaml"],
        "yaml_sha256": config["yaml_sha256"],
        "genome_sha256": config.get("genome_sha256"),
        "channel_index_map_sha256": config.get("channel_index_map_sha256"),
        "params_fused_exact": int(config["params_fused_exact"]),
        "flops_raw_exact": int(config["flops_raw_exact"]),
        "epochs_requested": 300,
        "epochs_completed": 300,
        "imgsz": 640,
        "batch_size": 16,
        "optimizer": "SGD",
        "hyperparameters": config["common_hyp"],
        "hyperparameters_sha256": config["common_hyp_sha256"],
        "amp": True,
        "ema": True,
        "autoanchor": True,
        "workers": 4,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "terminal_ema": str(terminal_ema),
        "terminal_ema_sha256": sha256(terminal_ema),
        "terminal_raw": str(terminal_raw),
        "terminal_raw_sha256": sha256(terminal_raw),
        "resumable_state": str(last),
        "resumable_state_sha256": sha256(last),
        "final_losses": parse_final_losses(output / "results.csv"),
        "validation_dataloader_created": False,
        "accuracy_evaluation_executed": False,
        "best_pt_selection_used": False,
        "early_stopping_used": False,
        "kd_executed": False,
        "teacher_loaded": False,
        "meta_inner_val1294_accessed": False,
        "official_val548_accessed": False,
        "heldout1610_accessed": False,
        "exit_status": "COMPLETE",
    }
    atomic_json(output / "run_manifest.json", run_manifest)
    hashes = {
        "created_utc": end_utc,
        "initialization": sha256(initialization),
        "terminal_ema": sha256(terminal_ema),
        "terminal_raw": sha256(terminal_raw),
        "resumable_state": sha256(last),
        "run_manifest": sha256(output / "run_manifest.json"),
        "training_command": sha256(output / "training_command.txt"),
    }
    atomic_json(output / "checkpoint_hashes.json", hashes)
    (output / ".complete").write_text("STAGE10_RUN=COMPLETE\n", encoding="utf-8")
    print(f"STAGE10_RUN_COMPLETE={config['run_id']}")


if __name__ == "__main__":
    main()

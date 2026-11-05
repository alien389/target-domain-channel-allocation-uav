from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class LoadedModelArtifact:
    model: torch.nn.Module
    weights_path: str
    yolov5_root: str
    checkpoint_type: str
    model_source: str
    fused: bool
    model_id: str
    yaml: Dict[str, Any]
    names: Any
    nc: Optional[int]
    stride: Any
    save: Any

    def to_metadata(self) -> Dict[str, Any]:
        stride_value = None
        if self.stride is not None:
            try:
                stride_value = [int(x) for x in self.stride]
            except Exception:
                stride_value = self.stride
        return {
            "weights_path": self.weights_path,
            "yolov5_root": self.yolov5_root,
            "checkpoint_type": self.checkpoint_type,
            "model_source": self.model_source,
            "fused": self.fused,
            "model_id": self.model_id,
            "nc": self.nc,
            "stride": stride_value,
            "save": list(self.save) if isinstance(self.save, (list, tuple)) else self.save,
        }


def _bootstrap_yolov5(yolov5_root: str) -> Path:
    root = Path(yolov5_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"YOLOv5 root does not exist: {root}")
    expected = root / "models" / "yolo.py"
    if not expected.exists():
        raise FileNotFoundError(f"Invalid YOLOv5 root: missing {expected}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _load_raw_checkpoint(weights: str) -> Any:
    try:
        return torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(weights, map_location="cpu")


def _resolve_model_id(weights: str, explicit_model_id: Optional[str] = None) -> str:
    if explicit_model_id and explicit_model_id.strip():
        return explicit_model_id.strip()
    return Path(weights).stem


def _load_standard_detection_model(weights: str, device: torch.device, fuse: bool) -> Optional[torch.nn.Module]:
    from models.experimental import attempt_load

    try:
        return attempt_load(weights, device=device, inplace=True, fuse=fuse)
    except Exception:
        return None


def _reconstruct_from_state_dict(checkpoint: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    from models.yolo import Model

    if "model_state_dict" not in checkpoint or "yaml" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict' and 'yaml'.")

    model = Model(cfg=checkpoint["yaml"], ch=3, nc=checkpoint.get("nc"), anchors=None).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if checkpoint.get("names") is not None:
        model.names = checkpoint["names"]
    if checkpoint.get("stride") is not None and hasattr(model, "stride"):
        model.stride = torch.tensor(checkpoint["stride"])
    model.float().eval()
    return model


def load_detection_model(
    weights: str,
    yolov5_root: str,
    device: str = "cpu",
    fuse: bool = True,
    model_id: Optional[str] = None,
) -> LoadedModelArtifact:
    root = _bootstrap_yolov5(yolov5_root)
    device_t = torch.device(device)
    resolved_weights = str(Path(weights).resolve())
    resolved_model_id = _resolve_model_id(resolved_weights, model_id)

    model = _load_standard_detection_model(resolved_weights, device_t, fuse=fuse)
    if model is not None:
        return LoadedModelArtifact(
            model=model.eval(),
            weights_path=resolved_weights,
            yolov5_root=str(root),
            checkpoint_type="standard_yolov5_checkpoint",
            model_source="attempt_load",
            fused=bool(fuse),
            model_id=resolved_model_id,
            yaml=getattr(model, "yaml", {}) or {},
            names=getattr(model, "names", None),
            nc=getattr(model, "nc", None),
            stride=getattr(model, "stride", None),
            save=getattr(model, "save", []),
        )

    checkpoint = _load_raw_checkpoint(resolved_weights)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format for structural loading: {resolved_weights}")

    model = _reconstruct_from_state_dict(checkpoint, device_t)
    model.yaml_file = Path(resolved_weights).name
    return LoadedModelArtifact(
        model=model,
        weights_path=resolved_weights,
        yolov5_root=str(root),
        checkpoint_type="structural_state_dict_checkpoint",
        model_source="yaml_plus_model_state_dict",
        fused=False,
        model_id=resolved_model_id,
        yaml=getattr(model, "yaml", {}) or checkpoint.get("yaml", {}) or {},
        names=getattr(model, "names", checkpoint.get("names")),
        nc=getattr(model, "nc", checkpoint.get("nc")),
        stride=getattr(model, "stride", checkpoint.get("stride")),
        save=getattr(model, "save", []),
    )


def load_training_checkpoint_model(
    weights: str,
    yolov5_root: str,
    device: str = "cpu",
    prefer_ema: bool = True,
    model_id: Optional[str] = None,
) -> LoadedModelArtifact:
    root = _bootstrap_yolov5(yolov5_root)
    device_t = torch.device(device)
    resolved_weights = str(Path(weights).resolve())
    resolved_model_id = _resolve_model_id(resolved_weights, model_id)

    checkpoint = _load_raw_checkpoint(resolved_weights)
    if isinstance(checkpoint, dict):
        selected_model = checkpoint.get("ema") if prefer_ema and checkpoint.get("ema") is not None else checkpoint.get("model")
        selected_source = "ema" if prefer_ema and checkpoint.get("ema") is not None else "model"
        if isinstance(selected_model, torch.nn.Module):
            model = copy.deepcopy(selected_model).to(device_t).float()
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            model.eval()
            return LoadedModelArtifact(
                model=model,
                weights_path=resolved_weights,
                yolov5_root=str(root),
                checkpoint_type="training_checkpoint_module",
                model_source=f"checkpoint:{selected_source}",
                fused=False,
                model_id=resolved_model_id,
                yaml=getattr(model, "yaml", {}) or {},
                names=getattr(model, "names", None),
                nc=getattr(model, "nc", None),
                stride=getattr(model, "stride", None),
                save=getattr(model, "save", []),
            )

        if "model_state_dict" in checkpoint and "yaml" in checkpoint:
            model = _reconstruct_from_state_dict(checkpoint, device_t)
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            model.eval()
            return LoadedModelArtifact(
                model=model,
                weights_path=resolved_weights,
                yolov5_root=str(root),
                checkpoint_type="training_state_dict_checkpoint",
                model_source="yaml_plus_model_state_dict",
                fused=False,
                model_id=resolved_model_id,
                yaml=getattr(model, "yaml", {}) or checkpoint.get("yaml", {}) or {},
                names=getattr(model, "names", checkpoint.get("names")),
                nc=getattr(model, "nc", checkpoint.get("nc")),
                stride=getattr(model, "stride", checkpoint.get("stride")),
                save=getattr(model, "save", []),
            )

    return load_detection_model(
        weights=resolved_weights,
        yolov5_root=str(root),
        device=device,
        fuse=False,
        model_id=resolved_model_id,
    )

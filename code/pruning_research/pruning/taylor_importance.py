from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import yaml

from pruning_research.core.model_loader import load_training_checkpoint_model
from pruning_research.pruning.importance_scores import (
    iter_prunable_topology_rows,
    rank_channel_scores,
    resolve_scope_modules,
)
from pruning_research.utils.io_utils import ensure_dir, save_csv, save_json


def _load_json(path: str | Path) -> Dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Taylor hook received unsupported output type: {type(output).__name__}")


def reduce_channel_scores(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    score_reduction: str = "mean_abs",
) -> torch.Tensor:
    interaction = activation.detach() * gradient.detach()
    score_reduction = str(score_reduction).strip().lower()

    if score_reduction == "mean_abs":
        return interaction.abs().mean(dim=(0, 2, 3))
    if score_reduction == "sum_abs":
        return interaction.abs().sum(dim=(0, 2, 3))
    raise KeyError(f"Unsupported Taylor score_reduction: {score_reduction}")


def _freeze_bn_running_stats(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            module.eval()
            count += 1
    return count


def _resolve_hyp(model: nn.Module, yolov5_root: str, hyp_yaml: Optional[str] = None) -> Dict[str, Any]:
    if hyp_yaml:
        with open(hyp_yaml, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    model_hyp = getattr(model, "hyp", None)
    if isinstance(model_hyp, dict) and model_hyp:
        return dict(model_hyp)

    fallback = Path(yolov5_root) / "data" / "hyps" / "hyp.scratch-low.yaml"
    with open(fallback, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _select_topology_rows(
    channel_topology: Dict[str, Any],
    pruning_constraints: Dict[str, Any],
    include_blocks: Optional[Iterable[str]] = None,
    exclude_blocks: Optional[Iterable[str]] = None,
    max_blocks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = iter_prunable_topology_rows(channel_topology=channel_topology, pruning_constraints=pruning_constraints)
    include_set = {str(item).strip() for item in include_blocks or [] if str(item).strip()}
    exclude_set = {str(item).strip() for item in exclude_blocks or [] if str(item).strip()}

    if include_set:
        rows = [row for row in rows if str(row["block_name"]) in include_set]
    if exclude_set:
        rows = [row for row in rows if str(row["block_name"]) not in exclude_set]
    if max_blocks is not None:
        rows = rows[: int(max_blocks)]
    return rows


def collect_taylor_scores_one_run(
    weights: str,
    yolov5_root: str,
    profile_dir: str,
    output_dir: str,
    data_yaml: str,
    model_id: str,
    imgsz: int,
    batch_size: int,
    workers: int,
    device: str = "cpu",
    prefer_ema: bool = True,
    split: str = "train",
    max_batches: int = 200,
    start_batch_offset: int = 0,
    score_reduction: str = "mean_abs",
    freeze_bn_running_stats: bool = True,
    seed: int = 42,
    hyp_yaml: Optional[str] = None,
    include_blocks: Optional[Iterable[str]] = None,
    exclude_blocks: Optional[Iterable[str]] = None,
    max_blocks: Optional[int] = None,
) -> Dict[str, Any]:
    profile_root = Path(profile_dir).resolve()
    target_dir = ensure_dir(Path(output_dir).resolve())
    channel_topology = _load_json(profile_root / "channel_topology.json")
    pruning_constraints = _load_json(profile_root / "pruning_constraints.json")

    artifact = load_training_checkpoint_model(
        weights=weights,
        yolov5_root=yolov5_root,
        device=device,
        prefer_ema=prefer_ema,
        model_id=model_id,
    )
    model = artifact.model.to(torch.device(device)).float()
    model.train()

    if freeze_bn_running_stats:
        frozen_bn_count = _freeze_bn_running_stats(model)
    else:
        frozen_bn_count = 0

    topology_rows = _select_topology_rows(
        channel_topology=channel_topology,
        pruning_constraints=pruning_constraints,
        include_blocks=include_blocks,
        exclude_blocks=exclude_blocks,
        max_blocks=max_blocks,
    )
    if not topology_rows:
        raise ValueError("No Taylor target blocks were selected after include/exclude filtering.")

    from utils.dataloaders import create_dataloader
    from utils.general import check_dataset, check_img_size
    from utils.loss import ComputeLoss

    data_dict = check_dataset(data_yaml)
    if split not in data_dict:
        raise KeyError(f"Requested Taylor split '{split}' is not available in dataset config.")

    stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
    resolved_imgsz = int(check_img_size(imgsz, s=stride))
    hyp = _resolve_hyp(model=model, yolov5_root=yolov5_root, hyp_yaml=hyp_yaml)

    dataloader, dataset = create_dataloader(
        path=data_dict[split],
        imgsz=resolved_imgsz,
        batch_size=batch_size,
        stride=stride,
        single_cls=False,
        hyp=hyp,
        augment=False,
        cache=False,
        pad=0.0,
        rect=False,
        rank=-1,
        workers=workers,
        image_weights=False,
        quad=False,
        prefix=f"Taylor {split}: ",
        shuffle=False,
        seed=seed,
    )

    compute_loss = ComputeLoss(model)

    scope_states: Dict[Tuple[str, str], Dict[str, Any]] = {}
    hooks: List[Any] = []

    for topology_row in topology_rows:
        scope_modules = resolve_scope_modules(model=model, topology_row=topology_row)
        for scope_name, scope_info in scope_modules.items():
            state = {
                "block_name": str(topology_row["block_name"]),
                "stage": str(topology_row["stage"]),
                "module_type": str(topology_row["module_type"]),
                "scope_name": scope_name,
                "scope_type": str(scope_info["scope_type"]),
                "anchor_module": str(scope_info["anchor_module"]),
                "width": int(scope_info["width"]),
                "score_sum": torch.zeros(int(scope_info["width"]), dtype=torch.float64),
                "observation_count": 0,
                "activation": None,
            }

            def _forward_hook(_module, _inputs, output, target_state=state):
                tensor = _as_tensor_output(output)
                target_state["activation"] = tensor
                tensor.retain_grad()

            hooks.append(scope_info["module"].register_forward_hook(_forward_hook))
            scope_states[(state["block_name"], scope_name)] = state

    processed_batches = 0
    observed_images = 0
    last_loss_value: Optional[float] = None

    try:
        for batch_index, (images, targets, _paths, _shapes) in enumerate(dataloader):
            if batch_index < int(start_batch_offset):
                continue
            if processed_batches >= int(max_batches):
                break

            images = images.to(model.device if hasattr(model, "device") else torch.device(device), non_blocking=True)
            images = images.float() / 255.0
            targets = targets.to(images.device)

            model.zero_grad(set_to_none=True)
            predictions = model(images)
            loss, _loss_items = compute_loss(predictions, targets)
            loss.backward()

            for state in scope_states.values():
                activation = state["activation"]
                if activation is None or activation.grad is None:
                    raise RuntimeError(
                        f"Taylor capture missing activation/gradient for {state['block_name']}::{state['scope_name']}"
                    )
                channel_scores = reduce_channel_scores(
                    activation=activation,
                    gradient=activation.grad,
                    score_reduction=score_reduction,
                )
                state["score_sum"] += channel_scores.detach().cpu().double()
                state["observation_count"] += 1
                state["activation"] = None

            processed_batches += 1
            observed_images += int(images.shape[0])
            last_loss_value = float(loss.detach().cpu())
            model.zero_grad(set_to_none=True)

    finally:
        for hook in hooks:
            hook.remove()

    if processed_batches <= 0:
        raise RuntimeError(
            f"Taylor collection processed zero batches. start_batch_offset={start_batch_offset}, max_batches={max_batches}"
        )

    scope_summary_rows: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []
    importance_bank: Dict[str, Any] = {}

    for topology_row in topology_rows:
        block_name = str(topology_row["block_name"])
        block_payload = {
            "block_name": block_name,
            "stage": str(topology_row["stage"]),
            "module_type": str(topology_row["module_type"]),
            "scope_rankings": {},
        }

        related_states = [
            state for key, state in scope_states.items() if key[0] == block_name
        ]
        for state in sorted(related_states, key=lambda item: item["scope_name"]):
            if int(state["observation_count"]) <= 0:
                raise RuntimeError(f"Taylor state for {block_name}/{state['scope_name']} was never observed.")

            scores = (state["score_sum"] / float(state["observation_count"])).tolist()
            max_score = max(scores) if scores else 0.0
            normalized_scores = [float(score / max_score) if max_score > 0 else 0.0 for score in scores]
            ranked_scope = rank_channel_scores(
                scores=scores,
                scope_type=str(state["scope_type"]),
                anchor_module=str(state["anchor_module"]),
            )
            ranked_scope["scores"] = [float(score) for score in scores]
            ranked_scope["normalized_scores"] = normalized_scores
            ranked_scope["observation_count"] = int(state["observation_count"])
            block_payload["scope_rankings"][state["scope_name"]] = ranked_scope

            scope_summary_rows.append(
                {
                    "block_name": block_name,
                    "stage": state["stage"],
                    "module_type": state["module_type"],
                    "scope_name": state["scope_name"],
                    "anchor_module": state["anchor_module"],
                    "width": int(state["width"]),
                    "observation_count": int(state["observation_count"]),
                    "score_reduction": score_reduction,
                    "min_score": ranked_scope["min_score"],
                    "mean_score": ranked_scope["mean_score"],
                    "max_score": ranked_scope["max_score"],
                }
            )

            ordered = list(ranked_scope["indices_low_to_high"])
            rank_map = {int(channel_index): rank + 1 for rank, channel_index in enumerate(ordered)}
            for channel_index, channel_score in enumerate(scores):
                channel_rows.append(
                    {
                        "block_name": block_name,
                        "stage": state["stage"],
                        "module_type": state["module_type"],
                        "scope_name": state["scope_name"],
                        "anchor_module": state["anchor_module"],
                        "channel_index": int(channel_index),
                        "score": float(channel_score),
                        "normalized_score": float(normalized_scores[channel_index]),
                        "rank_low_to_high": int(rank_map[int(channel_index)]),
                    }
                )

        importance_bank[block_name] = block_payload

    payload = {
        "model_id": artifact.model_id,
        "weights": artifact.weights_path,
        "importance_metric": "taylor_first_order",
        "score_reduction": score_reduction,
        "split": split,
        "start_batch_offset": int(start_batch_offset),
        "max_batches": int(max_batches),
        "processed_batches": int(processed_batches),
        "observed_images": int(observed_images),
        "freeze_bn_running_stats": bool(freeze_bn_running_stats),
        "frozen_bn_count": int(frozen_bn_count),
        "prefer_ema": bool(prefer_ema),
        "device": str(device),
        "imgsz": int(resolved_imgsz),
        "batch_size": int(batch_size),
        "workers": int(workers),
        "seed": int(seed),
        "last_loss": last_loss_value,
        "importance_bank": importance_bank,
        "protocol": {
            "importance_family": "taylor_first_order",
            "data_split": split,
            "loss_source": "official_yolov5_training_loss",
            "score_reduction": score_reduction,
            "bn_policy": "train_mode_with_frozen_running_stats" if freeze_bn_running_stats else "plain_train_mode",
            "note": "Channel saliency is computed from activation-gradient interaction on structurally prunable scopes.",
        },
    }

    scope_csv = save_csv(scope_summary_rows, target_dir / "raw_taylor_scores.csv")
    channel_csv = save_csv(channel_rows, target_dir / "raw_taylor_channel_scores.csv")
    raw_json = save_json(payload, target_dir / "raw_taylor_scores.json")

    summary = {
        "model_id": artifact.model_id,
        "raw_taylor_scores_json": str(raw_json),
        "raw_taylor_scores_csv": str(scope_csv),
        "raw_taylor_channel_scores_csv": str(channel_csv),
        "output_dir": str(target_dir),
        "processed_batches": int(processed_batches),
        "observed_images": int(observed_images),
        "start_batch_offset": int(start_batch_offset),
        "max_batches": int(max_batches),
        "target_block_count": len(topology_rows),
        "target_scope_count": len(scope_summary_rows),
        "frozen_bn_count": int(frozen_bn_count),
        "last_loss": last_loss_value,
    }
    summary_json = save_json(summary, target_dir / "taylor_collection_summary.json")
    summary["taylor_collection_summary_json"] = str(summary_json)
    return summary

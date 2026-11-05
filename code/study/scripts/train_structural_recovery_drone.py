from __future__ import annotations
import release_paths as _release_paths  # Portable source-root resolution.

import copy
import sys
from pathlib import Path

import torch


YOLOV5_ROOT = Path(_release_paths.expand('${YOLOV5_ROOT}'))


def main() -> int:
    if str(YOLOV5_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLOV5_ROOT))
    import train_drone

    opt = train_drone.parse_opt()
    if opt.resume:
        raise RuntimeError("Stage04 structural Recovery must start fresh; resume is forbidden.")
    if str(opt.cfg).strip():
        raise RuntimeError("Stage04 must preserve the raw structural model object; --cfg is forbidden.")
    checkpoint_path = Path(opt.weights).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    template = checkpoint.get("model")
    if not isinstance(template, torch.nn.Module):
        raise RuntimeError(f"Raw checkpoint does not contain a model object: {checkpoint_path}")
    template = copy.deepcopy(template).cpu().float()
    detect = template.model[-1]
    if int(getattr(detect, "nc", -1)) != 10 or len(getattr(detect, "m", [])) != 3:
        raise RuntimeError("Raw structural checkpoint violates the nc=10/three-scale contract.")
    source_shapes = {
        key: tuple(value.shape) for key, value in template.state_dict().items()
    }
    factory_calls = 0

    def structural_model_factory(*args, **kwargs):
        nonlocal factory_calls
        factory_calls += 1
        return copy.deepcopy(template)

    train_drone.Model = structural_model_factory
    train_drone.main(opt)
    if factory_calls != 1:
        raise RuntimeError(
            f"Unexpected structural model factory call count: {factory_calls}"
        )
    best = Path(opt.project).resolve() / opt.name / "weights" / "best.pt"
    last = Path(opt.project).resolve() / opt.name / "weights" / "last.pt"
    for path in (best, last):
        trained_checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
        trained_model = trained_checkpoint.get("ema") or trained_checkpoint.get("model")
        if not isinstance(trained_model, torch.nn.Module):
            raise RuntimeError(f"Training output has no model object: {path}")
        output_shapes = {
            key: tuple(value.shape)
            for key, value in trained_model.state_dict().items()
        }
        if output_shapes != source_shapes:
            raise RuntimeError(
                f"Structural tensor-shape contract changed during Recovery: {path}"
            )
    print("STRUCTURAL_RECOVERY_MODEL_PRESERVED=YES", flush=True)
    print("FRESH_OPTIMIZER_SCHEDULER_EMA=YES", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Load a released tensor-only YOLOv5 checkpoint. No pickle or execution of config."""
import argparse
import json
from pathlib import Path


def load_model(artifact, graph_path):
    import torch
    from safetensors.torch import load_file
    from models import common, yolo

    allowed = {k: getattr(torch.nn, k) for k in (
        "SiLU", "BatchNorm2d", "ModuleList", "Sequential", "Conv2d", "MaxPool2d", "Upsample")}
    allowed.update({k: getattr(common, k) for k in (
        "Bottleneck", "C3", "C3Lite", "Concat", "Conv", "DWConvLite", "SPPF")})
    allowed.update(Detect=yolo.Detect, DetectionModel=yolo.DetectionModel)
    tensors = load_file(str(artifact), device="cpu")
    used = set()
    graph = json.loads(Path(graph_path).read_bytes())
    if graph["format"] != "locked-yolov5-graph-v1":
        raise ValueError("Wrong graph format")

    def value(x):
        if isinstance(x, dict):
            if set(x) == {"tensor"}:
                k = x["tensor"]
                used.add(k)
                return tensors[k]
            if set(x) == {"tuple"}:
                return tuple(value(v) for v in x["tuple"])
            if set(x) == {"int_map"}:
                return {int(k): v for k, v in x["int_map"].items()}
            raise ValueError("Unknown attribute encoding")
        if isinstance(x, list):
            return [value(v) for v in x]
        return x

    def make(node):
        cls = allowed[node["type"]]
        m = cls.__new__(cls)
        torch.nn.Module.__init__(m)
        for k, v in node["attributes"].items():
            if k.startswith("_") and k != "_reversed_padding_repeated_twice":
                raise ValueError("Unexpected private attribute")
            setattr(m, k, value(v))
        for k, v in node["parameters"].items():
            t = value(v)
            m.register_parameter(k, None if t is None else torch.nn.Parameter(t, requires_grad=False))
        for k, v in node["buffers"].items():
            m.register_buffer(k, value(v), persistent=k not in node["nonpersistent"])
        for k, sub in node["children"].items():
            m.add_module(k, None if sub is None else make(sub))
        return m

    model = make(graph["root"])
    if used != set(tensors):
        raise ValueError("Unused or missing material tensor")
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("safetensors")
    p.add_argument("graph")
    a = p.parse_args()
    m = load_model(a.safetensors, a.graph)
    print(f"CPU model loaded structurally; {len(m.state_dict())} state keys; no forward executed")


if __name__ == "__main__":
    main()

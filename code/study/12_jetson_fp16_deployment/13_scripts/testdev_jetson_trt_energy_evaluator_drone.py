# DRONE_JETSON_EDIT: Drone/VisDrone Jetson copy generated from the read-only COCO evaluator.
# DRONE_JETSON_EDIT: Measurement logic is retained; dataset-specific defaults are max_det=500 and output width=15.
#!/usr/bin/env python3
"""
Jetson Xavier NX TensorRT engine evaluator with onboard VDD_IN energy logging.
Final Q1 protocol version: isolated memory mode + YOLOv5 raw-output shape guard.

Purpose
-------
- Evaluate YOLOv5-family TensorRT .engine files on Jetson.
- Modes:
  * check: verify TensorRT/PyCUDA/engine bindings/VDD_IN.
  * idle: measure idle VDD_IN power using tegrastats.
  * timing_e2e: decode -> preprocess -> TensorRT forward -> NMS/postprocess.
  * timing_forward: preloaded preprocessed inputs -> H2D -> TensorRT execute -> D2H.
  * memory: engine load + warmup + peak RSS.
- Energy:
  * Source: VDD_IN from onboard INA3221 exposed by tegrastats.
  * Sampling: default 0.1 s = 10 Hz.
  * Net energy = gross energy - idle_power_w * runtime_s.

Assumptions
-----------
- Batch size = 1.
- Input = NCHW FP32 or FP16, normally [1,3,640,640].
- Output is YOLOv5 raw prediction, normally [1,25200,15] for the 10-class Drone/VisDrone models.
- Engine has exactly one input and at least one output.
"""

from __future__ import annotations
import release_paths as _release_paths  # Reviewed release-only path relocation.

import argparse
import csv
import gc
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import psutil

try:
    import tensorrt as trt
except Exception as exc:
    trt = None
    _TRT_IMPORT_ERROR = exc
else:
    _TRT_IMPORT_ERROR = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except Exception as exc:
    cuda = None
    _PYCUDA_IMPORT_ERROR = exc
else:
    _PYCUDA_IMPORT_ERROR = None


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}

MASTER_COLUMNS = [
    "Model",
    "Display Name",
    "Backend",
    "Precision",
    "Input Size",
    "Batch Size",
    "Engine_Size_MiB",
    "E2E_Mean_ms",
    "E2E_Median_ms",
    "E2E_P95_ms",
    "E2E_Std_ms",
    "FPS_E2E",
    "E2E_Avg_VDDIN_W",
    "E2E_Net_VDDIN_W",
    "E2E_Gross_Energy_J_per_Image",
    "E2E_Net_Energy_J_per_Image",
    "Decode_Mean_ms",
    "Preprocess_Mean_ms",
    "Forward_E2E_Mean_ms",
    "Postprocess_Mean_ms",
    "ForwardOnly_Mean_ms",
    "ForwardOnly_Median_ms",
    "ForwardOnly_P95_ms",
    "ForwardOnly_Std_ms",
    "FPS_ForwardOnly",
    "ForwardOnly_Avg_VDDIN_W",
    "ForwardOnly_Net_VDDIN_W",
    "ForwardOnly_Gross_Energy_J_per_Infer",
    "ForwardOnly_Net_Energy_J_per_Infer",
    "RSS_Before_Load_MB",
    "RSS_After_Load_MB",
    "RSS_Post_Warmup_MB",
    "Peak_RSS_MB",
    "Peak_RSS_Delta_MB",
    "System_RAM_Peak_MB",
    "System_RAM_Peak_Percent",
    "System_RAM_Peak_Delta_MB",
    "CUDA_Unified_Peak_MB",
    "CUDA_Unified_Peak_Percent",
    "CUDA_Unified_Peak_Delta_MB",
    "CUDA_Unified_Min_Free_MB",
    "Swap_Peak_MB",
    "Temp_Max_C",
    "Timing_Images",
    "Warmup_E2E",
    "Timed_E2E",
    "Warmup_Forward",
    "Timed_Forward",
    "Runs",
]


@dataclass
class EngineSpec:
    model_id: str
    engine_path: str
    display_name: str
    precision: str = ""


def finite_numeric_array(values: Optional[Sequence[float]], name: str = "values") -> np.ndarray:
    """Return a flat finite float64 array without ambiguous NumPy truth tests."""
    if values is None:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.empty(0, dtype=np.float64)
    arr = arr.reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Infinity")
    return arr


def percentile(values: Optional[Sequence[float]], q: float) -> Optional[float]:
    arr = finite_numeric_array(values, "percentile values")
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def summarize(values: Optional[Sequence[float]], prefix: str) -> Dict[str, float]:
    arr = finite_numeric_array(values, f"{prefix} values")
    if arr.size == 0:
        return {}
    mean_v = float(arr.mean())
    std_v = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci = 1.96 * std_v / math.sqrt(arr.size) if arr.size > 1 else 0.0
    p90 = percentile(arr, 90)
    p95 = percentile(arr, 95)
    p99 = percentile(arr, 99)
    return {
        f"{prefix}_mean": mean_v,
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p90": 0.0 if p90 is None else p90,
        f"{prefix}_p95": 0.0 if p95 is None else p95,
        f"{prefix}_p99": 0.0 if p99 is None else p99,
        f"{prefix}_std": std_v,
        f"{prefix}_cv": float(std_v / mean_v) if mean_v > 0 else 0.0,
        f"{prefix}_ci95_low": mean_v - ci,
        f"{prefix}_ci95_high": mean_v + ci,
    }


def round_num(v: Any, ndigits: int = 6) -> Any:
    if isinstance(v, (float, np.floating)):
        return round(float(v), ndigits)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def json_default(value: Any) -> Any:
    """Serialize NumPy values explicitly and fail for unknown scientific types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def append_or_update_master(path: str, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if p.exists():
        with open(p, "r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    norm = {c: row.get(c, "") for c in MASTER_COLUMNS}
    updated = False
    for i, r in enumerate(rows):
        if r.get("Model") == norm["Model"]:
            rows[i] = norm
            updated = True
            break
    if not updated:
        rows.append(norm)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def portable_basename(raw_path: str) -> str:
    raw_path = raw_path.strip()
    if "\\" in raw_path or (len(raw_path) >= 2 and raw_path[1] == ":"):
        return PureWindowsPath(raw_path).name
    return Path(raw_path).name


def load_image_files(entry: str, image_root: Optional[str] = None) -> List[str]:
    ep = Path(entry)
    root = Path(image_root).resolve() if image_root else None
    files: List[str] = []
    if ep.is_dir():
        for f in sorted(ep.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
                files.append(str(f.resolve()))
    elif ep.is_file():
        if ep.suffix.lower() == ".txt":
            for line in ep.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw:
                    continue
                cand = Path(raw)
                if root is not None:
                    cand = root / portable_basename(raw)
                elif not cand.is_absolute():
                    cand = (ep.parent / cand).resolve()
                files.append(str(cand))
        elif ep.suffix.lower() in IMAGE_SUFFIXES:
            files.append(str(ep.resolve()))
        else:
            raise ValueError(f"Unsupported timing-list entry: {entry}")
    else:
        raise FileNotFoundError(f"Could not resolve timing-list/image entry: {entry}")

    out = []
    seen = set()
    for f in files:
        k = os.path.normcase(os.path.normpath(f))
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def letterbox(image: np.ndarray, new_shape: int | Tuple[int, int] = 640, color=(114, 114, 114),
              auto: bool = False, scale_fill: bool = False, scaleup: bool = True, stride: int = 32):
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)


def preprocess_bgr(image_bgr: np.ndarray, imgsz: int, stride: int, dtype: np.dtype = np.float32):
    img, ratio, pad = letterbox(image_bgr, new_shape=imgsz, stride=stride, auto=False)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    if dtype == np.float16:
        img = img.astype(np.float16)
    img = np.expand_dims(img, axis=0)
    return img, ratio, pad


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    a1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    a2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-7)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou_one_to_many(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thres]
    return keep


def scale_boxes(img1_shape: Tuple[int, int], boxes: np.ndarray, img0_shape: Tuple[int, int], ratio_pad=None):
    if boxes.size == 0:
        return boxes
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        ratio, pad = ratio_pad
        gain = ratio[0]
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])
    return boxes


def nms_yolov5(prediction: np.ndarray, conf_thres=0.25, iou_thres=0.45, max_det=500,
               multi_label=False, agnostic=False) -> List[np.ndarray]:
    if prediction.ndim == 2:
        prediction = prediction[None, ...]
    if prediction.ndim != 3:
        raise ValueError(f"Expected [B,N,C] or [B,C,N] output, got {prediction.shape}")
    # DRONE_JETSON_EDIT: Drone raw outputs are [B,25200,15]. Some export paths may
    # produce [B,15,25200]; transpose that form before NMS while retaining COCO compatibility.
    if prediction.shape[1] in (15, 85, 84, 6) and prediction.shape[2] > prediction.shape[1]:
        prediction = np.transpose(prediction, (0, 2, 1))
    prediction = prediction.astype(np.float32, copy=False)
    outputs = []
    for xi in range(prediction.shape[0]):
        x = prediction[xi]
        x = x[x[:, 4] > conf_thres]
        if x.shape[0] == 0:
            outputs.append(np.zeros((0, 6), dtype=np.float32))
            continue
        x = x.copy()
        x[:, 5:] *= x[:, 4:5]
        if multi_label:
            i, j = np.where(x[:, 5:] > conf_thres)
            if len(i) == 0:
                outputs.append(np.zeros((0, 6), dtype=np.float32))
                continue
            boxes = xywh2xyxy(x[i, :4])
            scores = x[i, j + 5]
            classes = j.astype(np.float32)
        else:
            cls = np.argmax(x[:, 5:], axis=1)
            scores = x[np.arange(x.shape[0]), cls + 5]
            keep = scores > conf_thres
            if not np.any(keep):
                outputs.append(np.zeros((0, 6), dtype=np.float32))
                continue
            boxes = xywh2xyxy(x[keep, :4])
            scores = scores[keep]
            classes = cls[keep].astype(np.float32)
        order = scores.argsort()[::-1]
        if order.size > 30000:
            order = order[:30000]
        boxes, scores, classes = boxes[order], scores[order], classes[order]
        if agnostic:
            keep_idx = nms_numpy(boxes, scores, iou_thres)
        else:
            keep_idx = []
            for c in np.unique(classes):
                idx = np.where(classes == c)[0]
                keep_c = nms_numpy(boxes[idx], scores[idx], iou_thres)
                keep_idx.extend(idx[j] for j in keep_c)
            keep_idx = sorted(set(keep_idx), key=lambda j: scores[j], reverse=True)
        keep_idx = keep_idx[:max_det]
        det = np.concatenate([boxes[keep_idx], scores[keep_idx, None], classes[keep_idx, None]], axis=1).astype(np.float32)
        outputs.append(det)
    return outputs


TEGRASTATS_EXECUTABLE = "/usr/bin/tegrastats"


def _sudo_tegrastats_command(*args: str) -> List[str]:
    return ["sudo", "-n", TEGRASTATS_EXECUTABLE, *args]


def _stop_sudo_tegrastats() -> None:
    try:
        subprocess.run(
            _sudo_tegrastats_command("--stop"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except Exception:
        pass


class VDDINLogger:
    def __init__(self, out_csv: str, sample_period: float = 0.1):
        self.out_csv = str(out_csv)
        self.sample_period = float(sample_period)
        self.interval_ms = max(20, int(round(sample_period * 1000)))
        self.samples: List[Dict[str, Any]] = []
        self.energy_j = 0.0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = None
        self._error: Optional[BaseException] = None
        Path(self.out_csv).parent.mkdir(parents=True, exist_ok=True)

    def _loop(self):
        pattern = re.compile(r"\bVDD_IN\s+(\d+)mW(?:/(\d+)mW)?")
        try:
            if not Path(TEGRASTATS_EXECUTABLE).is_file():
                raise RuntimeError(f"tegrastats not found: {TEGRASTATS_EXECUTABLE}")
            _stop_sudo_tegrastats()
            cmd = _sudo_tegrastats_command("--interval", str(self.interval_ms))
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if self._proc.stdout is None:
                raise RuntimeError("failed to read sudo tegrastats stdout")
            last = time.perf_counter()
            with open(self.out_csv, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["elapsed_s", "power_w", "dt_s", "energy_increment_j", "energy_cumulative_j", "raw_line"],
                )
                writer.writeheader()
                for line in self._proc.stdout:
                    if self._stop.is_set():
                        break
                    m = pattern.search(line)
                    if not m:
                        continue
                    now = time.perf_counter()
                    dt = now - last
                    last = now
                    elapsed = now - (self.start_time or now)
                    power_w = float(m.group(1)) / 1000.0
                    e_inc = power_w * dt if dt > 0 else 0.0
                    self.energy_j += e_inc
                    row = {
                        "elapsed_s": elapsed,
                        "power_w": power_w,
                        "dt_s": dt,
                        "energy_increment_j": e_inc,
                        "energy_cumulative_j": self.energy_j,
                        "raw_line": line.strip(),
                    }
                    self.samples.append(row)
                    writer.writerow(row)
                    f.flush()
        except BaseException as exc:
            self._error = exc
        finally:
            _stop_sudo_tegrastats()
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    pass

    def start(self):
        self.start_time = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        _stop_sudo_tegrastats()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("VDD_IN logger thread did not stop within 10 seconds")
        self.end_time = time.perf_counter()
        if self._error is not None:
            raise RuntimeError(f"VDD_IN logger failed: {self._error}")

    def summary(self, idle_power_w: Optional[float], n_items: int) -> Dict[str, Any]:
        runtime_s = float((self.end_time or time.perf_counter()) - (self.start_time or time.perf_counter()))
        powers = [float(s["power_w"]) for s in self.samples]
        avg = float(np.mean(powers)) if powers else None
        std = float(np.std(powers, ddof=1)) if len(powers) > 1 else 0.0
        min_power = float(np.min(powers)) if powers else None
        max_power = float(np.max(powers)) if powers else None
        median_power = float(np.median(powers)) if powers else None
        gross_e = float(self.energy_j)
        gross_per = gross_e / n_items if n_items > 0 else None
        net_power = None
        net_e = None
        net_per = None
        if idle_power_w is not None and avg is not None:
            net_power = avg - float(idle_power_w)
            net_e = gross_e - float(idle_power_w) * runtime_s
            net_per = net_e / n_items if n_items > 0 else None
        return {
            "energy_sample_count": len(self.samples),
            "energy_runtime_s": runtime_s,
            "avg_power_w": avg,
            "std_power_w": std,
            "min_power_w": min_power,
            "max_power_w": max_power,
            "median_power_w": median_power,
            "gross_energy_j": gross_e,
            "gross_energy_per_item_j": gross_per,
            "idle_power_w": idle_power_w,
            "net_power_w": net_power,
            "net_energy_j": net_e,
            "net_energy_per_item_j": net_per,
        }


def validate_energy_summary(summary: Dict[str, Any], expected_items: int) -> None:
    sample_count = summary.get("energy_sample_count")
    if not isinstance(sample_count, int) or sample_count < 2:
        raise RuntimeError(f"VDD_IN energy sample count is invalid: {sample_count!r}")
    for key in ("energy_runtime_s", "avg_power_w", "gross_energy_j", "gross_energy_per_item_j"):
        value = summary.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            raise RuntimeError(f"VDD_IN energy field {key} is invalid: {value!r}")
    for key in ("idle_power_w", "net_power_w", "net_energy_j", "net_energy_per_item_j"):
        value = summary.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"VDD_IN energy field {key} is invalid: {value!r}")
    if expected_items <= 0:
        raise RuntimeError(f"Expected timed item count must be positive: {expected_items}")


def read_temp_from_tegrastats_once(timeout_s: float = 2.0) -> Dict[str, Any]:
    if not Path(TEGRASTATS_EXECUTABLE).is_file():
        return {}
    p = None
    try:
        _stop_sudo_tegrastats()
        p = subprocess.Popen(
            _sudo_tegrastats_command("--interval", "1000"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        line = ""
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            if p.stdout is not None:
                line = p.stdout.readline()
                if line:
                    break
        _stop_sudo_tegrastats()
        try:
            p.wait(timeout=3)
        except Exception:
            pass
        vals = {}
        for key in ["thermal", "CPU", "GPU", "AUX", "AO", "PMIC"]:
            m = re.search(rf"\b{key}@([0-9.]+)C", line)
            if m:
                vals[f"{key.lower()}_temp_c"] = float(m.group(1))
        vdd = re.search(r"\bVDD_IN\s+(\d+)mW", line)
        if vdd:
            vals["vdd_in_w"] = float(vdd.group(1)) / 1000.0
        return vals
    except Exception:
        return {}
    finally:
        _stop_sudo_tegrastats()


class TrtEngineRunner:
    def __init__(self, engine_path: str):
        if trt is None:
            raise RuntimeError(f"TensorRT import failed: {_TRT_IMPORT_ERROR}")
        if cuda is None:
            raise RuntimeError(f"PyCUDA import failed: {_PYCUDA_IMPORT_ERROR}")
        self.engine_path = str(engine_path)
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self.stream = cuda.Stream()

        self.bindings = []
        self.host_inputs = {}
        self.device_inputs = {}
        self.host_outputs = {}
        self.device_outputs = {}
        self.input_names = []
        self.output_names = []

        self._binding_info = []
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            shape = tuple(engine.get_binding_shape(i))
            # Support explicit dynamic batch/shape by setting to [1,3,640,640] if needed.
            if is_input and any(dim < 0 for dim in shape):
                shape = (1, 3, 640, 640)
                self.context.set_binding_shape(i, shape)
            elif any(dim < 0 for dim in shape):
                shape = tuple(self.context.get_binding_shape(i))
            size = int(trt.volume(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))
            info = {"index": i, "name": name, "is_input": is_input, "dtype": str(dtype), "shape": list(shape), "nbytes": int(host_mem.nbytes)}
            self._binding_info.append(info)
            if is_input:
                self.input_names.append(name)
                self.host_inputs[name] = host_mem
                self.device_inputs[name] = dev_mem
                self.input_shape = shape
                self.input_dtype = dtype
            else:
                self.output_names.append(name)
                self.host_outputs[name] = host_mem
                self.device_outputs[name] = dev_mem

        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected one input binding, got {self.input_names}")
        if len(self.output_names) < 1:
            raise RuntimeError("Expected at least one output binding")

    @property
    def binding_info(self) -> List[Dict[str, Any]]:
        return self._binding_info

    def infer(self, inp: np.ndarray) -> np.ndarray:
        input_name = self.input_names[0]
        host_in = self.host_inputs[input_name]
        if inp.dtype != host_in.dtype:
            inp = inp.astype(host_in.dtype, copy=False)
        flat = np.ascontiguousarray(inp).ravel()
        if flat.size != host_in.size:
            raise ValueError(f"Input size mismatch. Got {flat.size}, expected {host_in.size}")
        np.copyto(host_in, flat)
        cuda.memcpy_htod_async(self.device_inputs[input_name], host_in, self.stream)
        if not self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle):
            raise RuntimeError("TensorRT execute_async_v2 returned false")
        outputs = []
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.host_outputs[name], self.device_outputs[name], self.stream)
        self.stream.synchronize()
        for name in self.output_names:
            idx = next(x["index"] for x in self._binding_info if x["name"] == name)
            shape = tuple(self.context.get_binding_shape(idx))
            if any(dim < 0 for dim in shape):
                shape = tuple(x for x in next(b["shape"] for b in self._binding_info if b["name"] == name))
            arr = np.array(self.host_outputs[name]).reshape(shape)
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(f"TensorRT output {name!r} contains NaN or Infinity")
            outputs.append(arr)
        # YOLOv5 engines here should have one output. If more, return first and log binding info.
        return outputs[0]


def preload_inputs(image_paths: Sequence[str], imgsz: int, stride: int, dtype: np.dtype) -> List[Dict[str, Any]]:
    data = []
    for p in image_paths:
        img0 = cv2.imread(p)
        if img0 is None:
            raise FileNotFoundError(f"Failed to decode image during preload: {p}")
        inp, ratio, pad = preprocess_bgr(img0, imgsz=imgsz, stride=stride, dtype=dtype)
        data.append({"path": p, "input": inp, "orig_shape": img0.shape[:2], "ratio": ratio, "pad": pad})
    return data


def measure_idle(duration_s: float, sample_period: float, out_csv: str) -> Dict[str, Any]:
    logger = VDDINLogger(out_csv=out_csv, sample_period=sample_period)
    logger.start()
    time.sleep(duration_s)
    logger.stop()
    s = logger.summary(idle_power_w=None, n_items=max(1, len(logger.samples)))
    return {
        "duration_s": duration_s,
        "sample_period_s": sample_period,
        "sample_count": s["energy_sample_count"],
        "idle_power_w_mean": s["avg_power_w"],
        "idle_power_w_std": s["std_power_w"],
        "idle_power_w_min": s["min_power_w"],
        "idle_power_w_max": s["max_power_w"],
        "idle_power_w_median": s["median_power_w"],
        "trace_csv": out_csv,
    }


def bytes_to_mb(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def system_memory_snapshot() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "system_ram_total_mb": bytes_to_mb(vm.total),
        "system_ram_used_mb": bytes_to_mb(vm.used),
        "system_ram_available_mb": bytes_to_mb(vm.available),
        "system_ram_used_percent": float(vm.percent),
        "swap_used_mb": bytes_to_mb(sw.used),
        "swap_total_mb": bytes_to_mb(sw.total),
        "swap_used_percent": float(sw.percent),
    }


def cuda_memory_snapshot() -> Dict[str, Any]:
    if cuda is None:
        return {
            "cuda_visible_unified_memory_available": False,
            "cuda_visible_unified_memory_error": f"PyCUDA import failed: {_PYCUDA_IMPORT_ERROR}",
        }
    try:
        free_b, total_b = cuda.mem_get_info()
        used_b = total_b - free_b
        return {
            "cuda_visible_unified_memory_available": True,
            "cuda_total_visible_memory_mb": bytes_to_mb(total_b),
            "cuda_free_memory_mb": bytes_to_mb(free_b),
            "cuda_used_memory_mb": bytes_to_mb(used_b),
            "cuda_used_memory_percent": float(used_b / total_b * 100.0) if total_b else None,
        }
    except Exception as exc:
        return {
            "cuda_visible_unified_memory_available": False,
            "cuda_visible_unified_memory_error": repr(exc),
        }


def memory_trace_row(phase: str, iteration: Optional[int], process: psutil.Process, start_time: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "elapsed_s": time.perf_counter() - start_time,
        "phase": phase,
        "iteration": iteration,
        "process_rss_mb": process.memory_info().rss / (1024 * 1024),
    }
    row.update(system_memory_snapshot())
    row.update(cuda_memory_snapshot())
    return row


def write_memory_trace(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def max_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return max(vals) if vals else None


def min_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return min(vals) if vals else None


def evaluate_e2e(spec: EngineSpec, runner: TrtEngineRunner, image_paths: Sequence[str], args) -> Dict[str, Any]:
    process = psutil.Process(os.getpid())
    dtype = np.dtype(runner.input_dtype)
    decode_all, pre_all, fwd_all, post_all, total_all = [], [], [], [], []
    run_rows = []
    sample_rows = []

    for r in range(args.runs):
        for i in range(args.warmup_e2e):
            p = image_paths[i % len(image_paths)]
            img0 = cv2.imread(p)
            if img0 is None:
                raise FileNotFoundError(f"Failed to decode warmup image: {p}")
            inp, ratio, pad = preprocess_bgr(img0, args.imgsz, args.stride, dtype=dtype)
            pred = runner.infer(inp)
            det = nms_yolov5(pred, args.conf_thres, args.iou_thres, args.max_det, args.multi_label)
            if det and det[0].shape[0]:
                _ = scale_boxes((args.imgsz, args.imgsz), det[0][:, :4], img0.shape[:2], ratio_pad=(ratio, pad))

        logger = None
        if args.energy:
            if args.idle_power_w is None:
                raise ValueError("--idle-power-w is required with --energy")
            logger = VDDINLogger(str(Path(args.output_dir) / f"{spec.model_id}_e2e_energy_run{r+1}_trace.csv"), args.energy_sample_period)
            logger.start()

        d_run, p_run, f_run, pp_run, t_run = [], [], [], [], []
        for i in range(args.timed_e2e):
            img_path = image_paths[i % len(image_paths)]
            t0 = time.perf_counter()
            img0 = cv2.imread(img_path)
            t1 = time.perf_counter()
            if img0 is None:
                raise FileNotFoundError(f"Failed to decode image: {img_path}")
            inp, ratio, pad = preprocess_bgr(img0, args.imgsz, args.stride, dtype=dtype)
            t2 = time.perf_counter()
            pred = runner.infer(inp)
            t3 = time.perf_counter()
            det = nms_yolov5(pred, args.conf_thres, args.iou_thres, args.max_det, args.multi_label)
            if det and det[0].shape[0]:
                _ = scale_boxes((args.imgsz, args.imgsz), det[0][:, :4], img0.shape[:2], ratio_pad=(ratio, pad))
            t4 = time.perf_counter()

            vals = ((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t4 - t3) * 1000, (t4 - t0) * 1000)
            d_run.append(vals[0]); p_run.append(vals[1]); f_run.append(vals[2]); pp_run.append(vals[3]); t_run.append(vals[4])
            decode_all.append(vals[0]); pre_all.append(vals[1]); fwd_all.append(vals[2]); post_all.append(vals[3]); total_all.append(vals[4])
            sample_rows.append({
                "run": r + 1,
                "iteration": i + 1,
                "decode_ms": vals[0],
                "preprocess_ms": vals[1],
                "forward_inside_e2e_ms": vals[2],
                "postprocess_ms": vals[3],
                "e2e_ms": vals[4],
            })

        e = {}
        if logger is not None:
            logger.stop()
            e = logger.summary(args.idle_power_w, args.timed_e2e)
            validate_energy_summary(e, args.timed_e2e)

        total_s = summarize(t_run, "e2e_ms")
        fwd_s = summarize(f_run, "forward_ms")
        dec_s = summarize(d_run, "decode_ms")
        pre_s = summarize(p_run, "preprocess_ms")
        post_s = summarize(pp_run, "postprocess_ms")
        temp = read_temp_from_tegrastats_once()

        run_rows.append({
            "run": r + 1,
            "timed_iters": args.timed_e2e,
            "idle_power_w": args.idle_power_w,
            "energy_sample_count": int(e.get("energy_sample_count", 0)),
            "energy_runtime_s": round_num(e.get("energy_runtime_s")),
            "avg_vddin_w": round_num(e.get("avg_power_w")),
            "net_vddin_w": round_num(e.get("net_power_w")),
            "gross_energy_j": round_num(e.get("gross_energy_j")),
            "net_energy_j": round_num(e.get("net_energy_j")),
            "gross_energy_per_image_j": round_num(e.get("gross_energy_per_item_j")),
            "net_energy_per_image_j": round_num(e.get("net_energy_per_item_j")),
            "decode_mean_ms": round_num(dec_s.get("decode_ms_mean")),
            "preprocess_mean_ms": round_num(pre_s.get("preprocess_ms_mean")),
            "forward_mean_ms": round_num(fwd_s.get("forward_ms_mean")),
            "postprocess_mean_ms": round_num(post_s.get("postprocess_ms_mean")),
            "e2e_mean_ms": round_num(total_s.get("e2e_ms_mean")),
            "e2e_median_ms": round_num(total_s.get("e2e_ms_median")),
            "e2e_p95_ms": round_num(total_s.get("e2e_ms_p95")),
            "e2e_std_ms": round_num(total_s.get("e2e_ms_std")),
            "fps_e2e": round_num(1000.0 / total_s["e2e_ms_mean"] if total_s.get("e2e_ms_mean") else 0.0),
            "rss_mb_after_run": round_num(process.memory_info().rss / (1024 * 1024)),
            "thermal_temp_c": temp.get("thermal_temp_c"),
            "gpu_temp_c": temp.get("gpu_temp_c"),
        })

    metrics = {}
    metrics.update(summarize(decode_all, "decode_ms"))
    metrics.update(summarize(pre_all, "preprocess_ms"))
    metrics.update(summarize(fwd_all, "forward_ms"))
    metrics.update(summarize(post_all, "postprocess_ms"))
    metrics.update(summarize(total_all, "e2e_ms"))
    metrics["fps_e2e"] = 1000.0 / metrics["e2e_ms_mean"] if metrics.get("e2e_ms_mean", 0) > 0 else 0.0
    metrics["idle_power_w"] = args.idle_power_w
    for k in ["avg_vddin_w", "net_vddin_w", "gross_energy_per_image_j", "net_energy_per_image_j"]:
        vals = [row[k] for row in run_rows if row.get(k) not in (None, "")]
        if vals:
            metrics[k] = float(np.mean(np.asarray(vals, dtype=np.float64)))
            metrics[f"{k}_std"] = float(np.std(np.asarray(vals, dtype=np.float64), ddof=1)) if len(vals) > 1 else 0.0
    sample_path = Path(args.output_dir) / f"{spec.model_id}_e2e_latency_samples.csv"
    write_csv(str(sample_path), sample_rows)
    return {"metrics": metrics, "run_rows": run_rows, "latency_samples_csv": str(sample_path)}


def evaluate_forward(spec: EngineSpec, runner: TrtEngineRunner, preloaded: Sequence[Dict[str, Any]], args) -> Dict[str, Any]:
    vals_all = []
    run_rows = []
    sample_rows = []
    for r in range(args.runs):
        for i in range(args.warmup_forward):
            _ = runner.infer(preloaded[i % len(preloaded)]["input"])

        logger = None
        if args.energy:
            if args.idle_power_w is None:
                raise ValueError("--idle-power-w is required with --energy")
            logger = VDDINLogger(str(Path(args.output_dir) / f"{spec.model_id}_forward_energy_run{r+1}_trace.csv"), args.energy_sample_period)
            logger.start()

        vals = []
        for i in range(args.timed_forward):
            arr = preloaded[i % len(preloaded)]["input"]
            t0 = time.perf_counter()
            _ = runner.infer(arr)
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0
            vals.append(ms)
            vals_all.append(ms)
            sample_rows.append({
                "run": r + 1,
                "iteration": i + 1,
                "forward_only_ms": ms,
            })

        e = {}
        if logger is not None:
            logger.stop()
            e = logger.summary(args.idle_power_w, args.timed_forward)
            validate_energy_summary(e, args.timed_forward)

        s = summarize(vals, "forward_only_ms")
        temp = read_temp_from_tegrastats_once()
        run_rows.append({
            "run": r + 1,
            "timed_iters": args.timed_forward,
            "idle_power_w": args.idle_power_w,
            "energy_sample_count": int(e.get("energy_sample_count", 0)),
            "energy_runtime_s": round_num(e.get("energy_runtime_s")),
            "avg_vddin_w": round_num(e.get("avg_power_w")),
            "net_vddin_w": round_num(e.get("net_power_w")),
            "gross_energy_j": round_num(e.get("gross_energy_j")),
            "net_energy_j": round_num(e.get("net_energy_j")),
            "gross_energy_per_infer_j": round_num(e.get("gross_energy_per_item_j")),
            "net_energy_per_infer_j": round_num(e.get("net_energy_per_item_j")),
            "forward_only_mean_ms": round_num(s.get("forward_only_ms_mean")),
            "forward_only_median_ms": round_num(s.get("forward_only_ms_median")),
            "forward_only_p95_ms": round_num(s.get("forward_only_ms_p95")),
            "forward_only_std_ms": round_num(s.get("forward_only_ms_std")),
            "fps_forward_only": round_num(1000.0 / s["forward_only_ms_mean"] if s.get("forward_only_ms_mean") else 0.0),
            "thermal_temp_c": temp.get("thermal_temp_c"),
            "gpu_temp_c": temp.get("gpu_temp_c"),
        })

    metrics = summarize(vals_all, "forward_only_ms")
    metrics["fps_forward_only"] = 1000.0 / metrics["forward_only_ms_mean"] if metrics.get("forward_only_ms_mean", 0) > 0 else 0.0
    metrics["idle_power_w"] = args.idle_power_w
    for k in ["avg_vddin_w", "net_vddin_w", "gross_energy_per_infer_j", "net_energy_per_infer_j"]:
        vals = [row[k] for row in run_rows if row.get(k) not in (None, "")]
        if vals:
            metrics[k] = float(np.mean(np.asarray(vals, dtype=np.float64)))
            metrics[f"{k}_std"] = float(np.std(np.asarray(vals, dtype=np.float64), ddof=1)) if len(vals) > 1 else 0.0
    sample_path = Path(args.output_dir) / f"{spec.model_id}_forward_latency_samples.csv"
    write_csv(str(sample_path), sample_rows)
    return {"metrics": metrics, "run_rows": run_rows, "latency_samples_csv": str(sample_path)}


def evaluate_memory(spec: EngineSpec, args) -> Dict[str, Any]:
    # DRONE_STAGE4_EDIT: Jetson uses unified physical memory, so report process RSS,
    # system RAM, and CUDA-visible unified/device memory as separate categories.
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    trace_rows: List[Dict[str, Any]] = []

    trace_rows.append(memory_trace_row("before_engine_load", None, process, start_time))
    rss_before = trace_rows[-1]["process_rss_mb"]
    system_before = trace_rows[-1]
    cuda_before = trace_rows[-1]

    runner = TrtEngineRunner(spec.engine_path)
    trace_rows.append(memory_trace_row("after_engine_load_and_buffer_allocation", None, process, start_time))
    rss_after_load = trace_rows[-1]["process_rss_mb"]
    system_after_load = trace_rows[-1]
    cuda_after_load = trace_rows[-1]

    dtype = np.dtype(runner.input_dtype)
    dummy = np.zeros(tuple(runner.input_shape), dtype=dtype)
    for _ in range(args.warmup_memory):
        _ = runner.infer(dummy)
    trace_rows.append(memory_trace_row("after_warmup", None, process, start_time))
    rss_post = trace_rows[-1]["process_rss_mb"]
    system_after_warmup = trace_rows[-1]
    cuda_after_warmup = trace_rows[-1]

    peak = rss_post
    for idx in range(args.timed_memory):
        _ = runner.infer(dummy)
        row = memory_trace_row("timed_loop", idx + 1, process, start_time)
        trace_rows.append(row)
        peak = max(peak, row["process_rss_mb"])

    system_peak_mb = max_optional([r.get("system_ram_used_mb") for r in trace_rows])
    system_peak_percent = max_optional([r.get("system_ram_used_percent") for r in trace_rows])
    system_min_available_mb = min_optional([r.get("system_ram_available_mb") for r in trace_rows])
    swap_peak_mb = max_optional([r.get("swap_used_mb") for r in trace_rows])
    cuda_available = any(bool(r.get("cuda_visible_unified_memory_available")) for r in trace_rows)
    cuda_peak_mb = max_optional([r.get("cuda_used_memory_mb") for r in trace_rows])
    cuda_peak_percent = max_optional([r.get("cuda_used_memory_percent") for r in trace_rows])
    cuda_min_free_mb = min_optional([r.get("cuda_free_memory_mb") for r in trace_rows])
    cuda_error = None
    for r in trace_rows:
        if r.get("cuda_visible_unified_memory_error"):
            cuda_error = r.get("cuda_visible_unified_memory_error")
            break

    trace_path = Path(args.output_dir) / f"{spec.model_id}_memory_trace.csv"
    write_memory_trace(trace_path, trace_rows)

    return {
        "metrics": {
            "rss_before_load_mb": rss_before,
            "rss_after_load_mb": rss_after_load,
            "rss_post_warmup_mb": rss_post,
            "peak_rss_mb": peak,
            "peak_rss_delta_mb": max(0.0, peak - rss_before),
            "system_ram_total_mb": system_before.get("system_ram_total_mb"),
            "system_ram_used_before_load_mb": system_before.get("system_ram_used_mb"),
            "system_ram_used_after_load_mb": system_after_load.get("system_ram_used_mb"),
            "system_ram_used_after_warmup_mb": system_after_warmup.get("system_ram_used_mb"),
            "system_ram_peak_mb": system_peak_mb,
            "system_ram_peak_percent": system_peak_percent,
            "system_ram_peak_delta_mb": max(0.0, system_peak_mb - system_before.get("system_ram_used_mb")) if system_peak_mb is not None and system_before.get("system_ram_used_mb") is not None else None,
            "system_ram_available_min_mb": system_min_available_mb,
            "swap_used_before_mb": system_before.get("swap_used_mb"),
            "swap_peak_mb": swap_peak_mb,
            "cuda_visible_unified_memory_available": cuda_available,
            "cuda_total_visible_memory_mb": cuda_before.get("cuda_total_visible_memory_mb"),
            "cuda_used_before_engine_load_mb": cuda_before.get("cuda_used_memory_mb"),
            "cuda_used_after_engine_load_mb": cuda_after_load.get("cuda_used_memory_mb"),
            "cuda_used_after_warmup_mb": cuda_after_warmup.get("cuda_used_memory_mb"),
            "cuda_unified_peak_mb": cuda_peak_mb,
            "cuda_unified_peak_percent": cuda_peak_percent,
            "cuda_unified_peak_delta_mb": max(0.0, cuda_peak_mb - cuda_before.get("cuda_used_memory_mb")) if cuda_peak_mb is not None and cuda_before.get("cuda_used_memory_mb") is not None else None,
            "cuda_unified_min_free_mb": cuda_min_free_mb,
            "cuda_visible_unified_memory_error": cuda_error,
            "memory_trace_csv": str(trace_path),
            "memory_sampling_policy": "sampled every timed memory iteration; no slower than 0.1 s for the accepted protocol",
            "memory_model_note": "Jetson Xavier NX uses shared/unified physical memory; CUDA-visible memory is not reported as discrete VRAM.",
        },
        "binding_info": runner.binding_info,
    }


def load_manifest(path: str) -> List[EngineSpec]:
    specs = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            model_id = row["model_id"].strip()
            engine_path = row["engine_path"].strip()
            specs.append(EngineSpec(
                model_id=model_id,
                engine_path=engine_path,
                display_name=row.get("display_name", model_id).strip() or model_id,
                precision=row.get("precision", "").strip(),
            ))
    return specs



def run_memory_subprocess(spec: EngineSpec, args) -> Dict[str, Any]:
    """Profile memory in an isolated Python process.

    This avoids contaminating RSS memory measurements with the TensorRT engine,
    bindings, and CUDA allocations already used for E2E/forward timing.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode", "memory",
        "--model-id", spec.model_id,
        "--engine-path", spec.engine_path,
        "--display-name", spec.display_name,
        "--precision", spec.precision,
        "--output-dir", str(args.output_dir),
        "--imgsz", str(args.imgsz),
        "--stride", str(args.stride),
        "--warmup-memory", str(args.warmup_memory),
        "--timed-memory", str(args.timed_memory),
        "--max-det", str(args.max_det),
        "--memory-sample-period", str(args.memory_sample_period),
    ]
    subprocess.run(cmd, check=True)
    log_path = Path(args.output_dir) / f"{spec.model_id}_system_log.json"
    with open(log_path, "r", encoding="utf-8") as f:
        child = json.load(f)
    mem = child.get("memory_report")
    if not mem:
        raise RuntimeError(f"Isolated memory subprocess did not produce memory_report for {spec.model_id}")
    return mem

def build_master_row(spec: EngineSpec, args, runner: Optional[TrtEngineRunner], e2e, fwd, mem) -> Dict[str, Any]:
    e = (e2e or {}).get("metrics", {})
    f = (fwd or {}).get("metrics", {})
    m = (mem or {}).get("metrics", {})
    temp_vals = []
    for report in [e2e, fwd]:
        if not report:
            continue
        for row in report.get("run_rows", []):
            for k in ["thermal_temp_c", "gpu_temp_c"]:
                if row.get(k) is not None:
                    temp_vals.append(float(row[k]))
    return {
        "Model": spec.model_id,
        "Display Name": spec.display_name,
        "Backend": "TensorRT",
        "Precision": spec.precision,
        "Input Size": args.imgsz,
        "Batch Size": 1,
        "Engine_Size_MiB": os.path.getsize(spec.engine_path) / (1024 * 1024),
        "E2E_Mean_ms": round_num(e.get("e2e_ms_mean")),
        "E2E_Median_ms": round_num(e.get("e2e_ms_median")),
        "E2E_P95_ms": round_num(e.get("e2e_ms_p95")),
        "E2E_Std_ms": round_num(e.get("e2e_ms_std")),
        "FPS_E2E": round_num(e.get("fps_e2e")),
        "E2E_Avg_VDDIN_W": round_num(e.get("avg_vddin_w")),
        "E2E_Net_VDDIN_W": round_num(e.get("net_vddin_w")),
        "E2E_Gross_Energy_J_per_Image": round_num(e.get("gross_energy_per_image_j")),
        "E2E_Net_Energy_J_per_Image": round_num(e.get("net_energy_per_image_j")),
        "Decode_Mean_ms": round_num(e.get("decode_ms_mean")),
        "Preprocess_Mean_ms": round_num(e.get("preprocess_ms_mean")),
        "Forward_E2E_Mean_ms": round_num(e.get("forward_ms_mean")),
        "Postprocess_Mean_ms": round_num(e.get("postprocess_ms_mean")),
        "ForwardOnly_Mean_ms": round_num(f.get("forward_only_ms_mean")),
        "ForwardOnly_Median_ms": round_num(f.get("forward_only_ms_median")),
        "ForwardOnly_P95_ms": round_num(f.get("forward_only_ms_p95")),
        "ForwardOnly_Std_ms": round_num(f.get("forward_only_ms_std")),
        "FPS_ForwardOnly": round_num(f.get("fps_forward_only")),
        "ForwardOnly_Avg_VDDIN_W": round_num(f.get("avg_vddin_w")),
        "ForwardOnly_Net_VDDIN_W": round_num(f.get("net_vddin_w")),
        "ForwardOnly_Gross_Energy_J_per_Infer": round_num(f.get("gross_energy_per_infer_j")),
        "ForwardOnly_Net_Energy_J_per_Infer": round_num(f.get("net_energy_per_infer_j")),
        "RSS_Before_Load_MB": round_num(m.get("rss_before_load_mb")),
        "RSS_After_Load_MB": round_num(m.get("rss_after_load_mb")),
        "RSS_Post_Warmup_MB": round_num(m.get("rss_post_warmup_mb")),
        "Peak_RSS_MB": round_num(m.get("peak_rss_mb")),
        "Peak_RSS_Delta_MB": round_num(m.get("peak_rss_delta_mb")),
        "System_RAM_Peak_MB": round_num(m.get("system_ram_peak_mb")),
        "System_RAM_Peak_Percent": round_num(m.get("system_ram_peak_percent")),
        "System_RAM_Peak_Delta_MB": round_num(m.get("system_ram_peak_delta_mb")),
        "CUDA_Unified_Peak_MB": round_num(m.get("cuda_unified_peak_mb")),
        "CUDA_Unified_Peak_Percent": round_num(m.get("cuda_unified_peak_percent")),
        "CUDA_Unified_Peak_Delta_MB": round_num(m.get("cuda_unified_peak_delta_mb")),
        "CUDA_Unified_Min_Free_MB": round_num(m.get("cuda_unified_min_free_mb")),
        "Swap_Peak_MB": round_num(m.get("swap_peak_mb")),
        "Temp_Max_C": round_num(max(temp_vals) if temp_vals else None),
        "Timing_Images": len(load_image_files(args.timing_list, args.image_root)) if args.timing_list else "",
        "Warmup_E2E": args.warmup_e2e,
        "Timed_E2E": args.timed_e2e,
        "Warmup_Forward": args.warmup_forward,
        "Timed_Forward": args.timed_forward,
        "Runs": args.runs,
    }


def evaluate_one(spec: EngineSpec, args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "idle":
        idle = measure_idle(args.idle_duration, args.energy_sample_period, str(out / "idle_vddin_trace.csv"))
        with open(out / "idle_vddin_summary.json", "w", encoding="utf-8") as f:
            json.dump(idle, f, indent=2, default=json_default)
        print(json.dumps(idle, indent=2, default=json_default))
        print(f'Use: --idle-power-w {idle["idle_power_w_mean"]:.6f}')
        return {"idle": idle}

    image_paths = []
    if args.mode in {"timing_e2e", "timing_forward", "memory", "all", "check"} and args.timing_list:
        image_paths = load_image_files(args.timing_list, args.image_root)

    if args.mode == "memory":
        mem = evaluate_memory(spec, args)
        write_csv(str(out / f"{spec.model_id}_memory_summary.csv"), [mem["metrics"]])
        with open(out / f"{spec.model_id}_system_log.json", "w", encoding="utf-8") as f:
            json.dump({"model": spec.__dict__, "idle_power_w": args.idle_power_w, "memory_report": mem, "environment": environment()}, f, indent=2, default=json_default)
        return {"memory_report": mem}

    runner = TrtEngineRunner(spec.engine_path)
    engine_binding_info = runner.binding_info

    if args.mode == "check":
        report = {"model": spec.__dict__, "engine_binding_info": engine_binding_info, "environment": environment(), "vddin_probe": read_temp_from_tegrastats_once()}
        with open(out / f"{spec.model_id}_check.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=json_default)
        print(json.dumps(report, indent=2, default=json_default))
        return report

    e2e = None
    fwd = None
    mem = None

    if args.mode in {"timing_e2e", "all"}:
        if not image_paths:
            raise RuntimeError("--timing-list is required")
        e2e = evaluate_e2e(spec, runner, image_paths, args)
        write_csv(str(out / f"{spec.model_id}_e2e_run_summary.csv"), e2e["run_rows"])

    if args.mode in {"timing_forward", "all"}:
        if not image_paths:
            raise RuntimeError("--timing-list is required")
        preload_count = min(args.forward_preload_count, len(image_paths)) if args.forward_preload_count > 0 else len(image_paths)
        preloaded = preload_inputs(image_paths[:preload_count], args.imgsz, args.stride, np.dtype(runner.input_dtype))
        fwd = evaluate_forward(spec, runner, preloaded, args)
        write_csv(str(out / f"{spec.model_id}_forward_run_summary.csv"), fwd["run_rows"])
        del preloaded
        gc.collect()

    if args.mode == "all":
        # Release timing runner before memory profiling. Memory is profiled in a
        # child process so RSS reflects a clean engine-load/warmup/run sequence.
        del runner
        gc.collect()
        mem = run_memory_subprocess(spec, args)
        write_csv(str(out / f"{spec.model_id}_memory_summary.csv"), [mem["metrics"]])

    report = {
        "protocol_statement": (
            "Jetson TensorRT engine evaluation separates end-to-end latency, forward-only latency, memory, "
            "and VDD_IN energy logging. VDD_IN is sampled from the onboard INA3221 monitor through tegrastats. "
            "Batch size is fixed to 1. Energy logging starts after warmup and stops immediately after the timed loop."
        ),
        "model": spec.__dict__,
        "idle_power_w": args.idle_power_w,
        "environment": environment(),
        "engine_binding_info": engine_binding_info,
        "e2e_report": e2e,
        "forward_report": fwd,
        "memory_report": mem,
    }
    with open(out / f"{spec.model_id}_system_log.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=json_default)

    if args.mode == "all":
        row = build_master_row(spec, args, None, e2e, fwd, mem)
        append_or_update_master(str(out / "jetson_xavier_nx_trt_master.csv"), row)

    return report


def environment() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "tensorrt_version": getattr(trt, "__version__", None) if trt is not None else None,
        "pycuda_available": cuda is not None,
        "tegrastats_path": shutil.which("tegrastats"),
        "nvpmodel": subprocess.getoutput("nvpmodel -q 2>/dev/null"),
        "cpu_online": Path("/sys/devices/system/cpu/online").read_text().strip() if Path("/sys/devices/system/cpu/online").exists() else None,
        "process_rss_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
        "system_ram_used_mb": psutil.virtual_memory().used / (1024 * 1024),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Jetson Xavier NX Drone TensorRT VDD_IN evaluator")
    p.add_argument("--mode", choices=["check", "idle", "timing_e2e", "timing_forward", "memory", "all"], default="check")
    p.add_argument("--model-id", type=str)
    p.add_argument("--engine-path", type=str)
    p.add_argument("--display-name", type=str)
    p.add_argument("--precision", type=str, default="")
    p.add_argument("--manifest-csv", type=str)
    p.add_argument("--timing-list", type=str)
    p.add_argument("--image-root", type=str)
    p.add_argument("--output-dir", type=str, default=_release_paths.expand('${JETSON_HOME}/od_benchmark_drone_jetson/results'))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--iou-thres", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=500)
    p.add_argument("--multi-label", action="store_true")
    p.add_argument("--warmup-e2e", type=int, default=100)
    p.add_argument("--timed-e2e", type=int, default=1000)
    p.add_argument("--warmup-forward", type=int, default=300)
    p.add_argument("--timed-forward", type=int, default=3000)
    p.add_argument("--forward-preload-count", type=int, default=300)
    p.add_argument("--warmup-memory", type=int, default=100)
    p.add_argument("--timed-memory", type=int, default=300)
    p.add_argument("--memory-sample-period", type=float, default=0.1)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--energy", action="store_true")
    p.add_argument("--idle-power-w", type=float)
    p.add_argument("--energy-sample-period", type=float, default=0.1)
    p.add_argument("--idle-duration", type=float, default=60.0)
    return p


def main():
    args = parser().parse_args()

    if args.mode != "idle":
        if args.manifest_csv:
            specs = load_manifest(args.manifest_csv)
        else:
            if not args.model_id or not args.engine_path:
                raise SystemExit("Single-engine usage requires --model-id and --engine-path")
            specs = [EngineSpec(
                model_id=args.model_id,
                engine_path=args.engine_path,
                display_name=args.display_name or args.model_id,
                precision=args.precision,
            )]
        for s in specs:
            if not Path(s.engine_path).exists():
                raise FileNotFoundError(s.engine_path)
            evaluate_one(s, args)
    else:
        dummy = EngineSpec(model_id="idle", engine_path="", display_name="idle", precision="")
        evaluate_one(dummy, args)


if __name__ == "__main__":
    main()

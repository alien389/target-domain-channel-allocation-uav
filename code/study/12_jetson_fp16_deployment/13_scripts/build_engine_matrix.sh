#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${JETSON_PROJECT_ROOT}
TRTEXEC=/usr/src/tensorrt/bin/trtexec
PYTHON=${JETSON_HOME}/venvs/jetson_trt/bin/python
STATE="$ROOT/08_state/engine_build_state.json"
LOG_ROOT="$ROOT/09_logs/engine_build"
ARCHIVE="$ROOT/15_archive/engine_build_partial"
MIN_FREE_BYTES=1610612736
completed=0
current="PREFLIGHT"

mkdir -p "$LOG_ROOT" "$ARCHIVE"

write_state() {
  local status="$1"
  "$PYTHON" - "$STATE" "$status" "$current" "$completed" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, status, current, completed = sys.argv[1:]
data = {
    "schema": "meta_jetson_engine_build_state_v1",
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "current_engine": current,
    "completed_engines": int(completed),
    "expected_engines": 16,
}
temp = path + ".tmp"
with open(temp, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temp, path)
PY
}

on_error() {
  write_state FAILED
}
trap on_error ERR

require_space() {
  local available
  available="$(df -B1 --output=avail "$ROOT" | tail -1 | tr -d ' ')"
  if [ "$available" -lt "$MIN_FREE_BYTES" ]; then
    echo "HARD_STOP_LOW_DISK available=$available threshold=$MIN_FREE_BYTES" >&2
    return 1
  fi
}

archive_partial() {
  local path="$1"
  if [ -e "$path" ]; then
    mv "$path" "$ARCHIVE/$(basename "$path").$(date -u +%Y%m%dT%H%M%SZ)"
  fi
}

build_trtexec() {
  local engine_id="$1" variant="$2" source="$3" destination="$4"
  local log_dir="$LOG_ROOT/$engine_id" partial="$destination.partial"
  mkdir -p "$log_dir" "$(dirname "$destination")"
  current="$engine_id"
  write_state BUILDING
  require_space
  if [ -s "$destination" ] && [ -s "$log_dir/engine.sha256" ]; then
    if echo "$(cat "$log_dir/engine.sha256")  $destination" | sha256sum -c - >/dev/null 2>&1; then
      completed=$((completed + 1))
      write_state REUSED_VALID
      return 0
    fi
    archive_partial "$destination"
  fi
  archive_partial "$partial"
  local -a command=("$TRTEXEC" "--onnx=$source" "--saveEngine=$partial")
  case "$variant" in
    fp32) command+=("--workspace=2048" "--buildOnly" "--profilingVerbosity=detailed") ;;
    fp16) command+=("--fp16" "--workspace=2048" "--buildOnly" "--profilingVerbosity=detailed") ;;
    selective_int8_fp16) command+=("--int8" "--fp16" "--workspace=2048" "--buildOnly" "--profilingVerbosity=detailed") ;;
    *) echo "Unknown trtexec variant: $variant" >&2; return 1 ;;
  esac
  printf '%q ' "${command[@]}" > "$log_dir/command.txt"
  printf '\n' >> "$log_dir/command.txt"
  printf '%s\n' "$source" > "$log_dir/source_onnx_path.txt"
  sha256sum "$source" > "$log_dir/source_onnx.sha256"
  "${command[@]}" > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"
  test -s "$partial"
  mv "$partial" "$destination"
  sha256sum "$destination" | awk '{print toupper($1)}' > "$log_dir/engine.sha256"
  stat -c '%s' "$destination" > "$log_dir/engine.size_bytes"
  date -u +%FT%TZ > "$log_dir/completed_utc.txt"
  touch "$log_dir/.complete"
  sync
  completed=$((completed + 1))
  write_state COMMITTED
}

build_aggressive() {
  local engine_id=meta_ap_aps_b6_seed42_trt_aggressive_int8
  local source="$ROOT/03_onnx/fp32/meta_ap_aps_b6_seed42_fp32.onnx"
  local destination="$ROOT/04_engines/aggressive_int8/$engine_id.engine"
  local partial="$destination.partial" log_dir="$LOG_ROOT/$engine_id"
  local cache="$ROOT/03_onnx/aggressive_int8_sources/meta_ap_aps_b6_seed42_entropy_calibration.cache"
  local metadata="$ROOT/03_onnx/aggressive_int8_sources/meta_ap_aps_b6_seed42_aggressive_build_metadata.json"
  mkdir -p "$log_dir" "$(dirname "$destination")"
  current="$engine_id"
  write_state BUILDING
  require_space
  if [ -s "$destination" ] && [ -s "$log_dir/engine.sha256" ]; then
    if echo "$(cat "$log_dir/engine.sha256")  $destination" | sha256sum -c - >/dev/null 2>&1; then
      completed=$((completed + 1))
      write_state REUSED_VALID
      return 0
    fi
    archive_partial "$destination"
  fi
  archive_partial "$partial"
  local -a command=(
    "$PYTHON" "$ROOT/13_scripts/build_calibrated_int8_trt.py"
    --onnx "$source"
    --engine "$partial"
    --cache "$cache"
    --calib-dir "$ROOT/02_locked_inputs/datasets/calibration1024/images"
    --imgsz 640
    --stride 32
    --workspace-mb 2048
    --max-calib-images 1024
    --fp16-fallback
  )
  printf '%q ' "${command[@]}" > "$log_dir/command.txt"
  printf '\n' >> "$log_dir/command.txt"
  sha256sum "$source" > "$log_dir/source_onnx.sha256"
  "${command[@]}" > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"
  test -s "$partial"
  mv "$partial" "$destination"
  "$PYTHON" - "$metadata" "$engine_id" "$source" "$destination" "$cache" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

metadata_path, engine_id, source, engine, cache = map(Path, sys.argv[1:])
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()
data = {
    "schema": "meta_jetson_aggressive_int8_build_metadata_v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "engine_id": str(engine_id),
    "policy": "aggressive calibrated INT8 PTQ with FP16 fallback",
    "calibrator": "IInt8EntropyCalibrator2",
    "calibration_images": 1024,
    "calibration_order": "deterministic lexical order from the locked 1024-image directory, matching the executed builder",
    "calibration_batch_size": 1,
    "workspace_mb": 2048,
    "fp16_fallback": True,
    "source_onnx": str(source),
    "source_onnx_sha256": digest(source),
    "engine": str(engine),
    "engine_sha256": digest(engine),
    "engine_size_bytes": engine.stat().st_size,
    "calibration_cache": str(cache),
    "calibration_cache_sha256": digest(cache),
    "status": "PASS",
}
temp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
temp.replace(metadata_path)
PY
  sha256sum "$destination" | awk '{print toupper($1)}' > "$log_dir/engine.sha256"
  stat -c '%s' "$destination" > "$log_dir/engine.size_bytes"
  date -u +%FT%TZ > "$log_dir/completed_utc.txt"
  touch "$log_dir/.complete"
  sync
  completed=$((completed + 1))
  write_state COMMITTED
}

test -x "$TRTEXEC"
test -x "$PYTHON"
if pgrep -x trtexec >/dev/null || pgrep -f build_calibrated_int8_trt.py >/dev/null; then
  echo "Another TensorRT build is active" >&2
  exit 1
fi
write_state PREFLIGHT_PASS

models=(meta_ap_aps_b6 matched_manual_b6 meta_ap_only_b6 random_search_b6 source_locked_stage6_b6)
for model in "${models[@]}"; do
  canonical="$ROOT/03_onnx/fp32/${model}_seed42_fp32.onnx"
  qdq="$ROOT/03_onnx/selective_qdq/${model}_seed42_selective_qdq.onnx"
  build_trtexec "${model}_seed42_trt_fp32" fp32 "$canonical" "$ROOT/04_engines/fp32/${model}_seed42_trt_fp32.engine"
  build_trtexec "${model}_seed42_trt_fp16" fp16 "$canonical" "$ROOT/04_engines/fp16/${model}_seed42_trt_fp16.engine"
  build_trtexec "${model}_seed42_trt_selective_int8_fp16" selective_int8_fp16 "$qdq" "$ROOT/04_engines/selective_int8_fp16/${model}_seed42_trt_selective_int8_fp16.engine"
done
build_aggressive

current="NONE"
write_state COMPLETE
touch "$ROOT/08_state/ENGINE_BUILD_COMPLETE"
sync
echo "ENGINE_BUILD_MATRIX_COMPLETE completed=$completed"

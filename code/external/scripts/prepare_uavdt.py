from __future__ import annotations

import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2

from stage15_common import (
    STAGE15_ROOT,
    UAVDT_ROOT,
    VISDRONE_CLASSES,
    file_tree,
    require,
    sha256,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


ATTR_ZIP = UAVDT_ROOT / "M_attr.zip"
IMAGE_ZIP = UAVDT_ROOT / "UAV-benchmark-M.zip"
TOOLKIT_ZIP = UAVDT_ROOT / "UAV-benchmark-MOTD_v1.0.zip"
ATTR_PREFIX = "M_attr/test/"
IMAGE_PREFIX = "UAV-benchmark-M/"
TOOLKIT_PREFIX = "UAV-benchmark-MOTD_v1.0/"


def read_zip_text(archive: zipfile.ZipFile, name: str) -> str:
    return archive.read(name).decode("utf-8", errors="replace")


def atomic_extract_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    info = archive.getinfo(member)
    if destination.is_file() and destination.stat().st_size == info.file_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with archive.open(info) as source, temp.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    temp.replace(destination)


def parse_numeric_rows(text: str, columns: int, source: str) -> list[list[int]]:
    rows: list[list[int]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == columns, f"Malformed UAVDT row {source}:{line_number}: {line}")
        rows.append([int(float(field)) for field in fields])
    return rows


def main() -> int:
    out = STAGE15_ROOT / "02_UAVDT_PREP"
    derived = out / "derived"
    images_root = derived / "images"
    gt_root = derived / "gt"
    evidence_root = derived / "official_evidence"
    for path in (ATTR_ZIP, IMAGE_ZIP, TOOLKIT_ZIP):
        require(path.is_file(), f"Missing UAVDT source package: {path}")

    with zipfile.ZipFile(ATTR_ZIP) as attrs, zipfile.ZipFile(TOOLKIT_ZIP) as toolkit:
        attribute_test_sequences = sorted(
            {
                match.group(1)
                for name in attrs.namelist()
                if (match := re.fullmatch(r"M_attr/test/(M\d+)\s*_attr\.txt", name))
            }
        )
        attribute_train_sequences = sorted(
            {
                match.group(1)
                for name in attrs.namelist()
                if (match := re.fullmatch(r"M_attr/train/(M\d+)_attr\.txt", name))
            }
        )
        require(len(attribute_test_sequences) == 20, "UAVDT M_attr does not identify 20 test sequences")
        require(len(attribute_train_sequences) == 30, "UAVDT M_attr does not identify 30 train sequences")
        require(not (set(attribute_test_sequences) & set(attribute_train_sequences)), "UAVDT train/test sequence overlap")

        overall_name = TOOLKIT_PREFIX + "utils/CalculateDetectionPR_overall.m"
        overall_text = read_zip_text(toolkit, overall_name)
        seq_match = re.search(r"seqDirs\s*=\s*\{([^;]+)\};", overall_text)
        len_match = re.search(r"seqLens\s*=\s*\[([^\]]+)\]", overall_text)
        require(seq_match is not None and len_match is not None, "Could not parse official UAVDT DET test selector")
        toolkit_sequences = re.findall(r"'([^']+)'", seq_match.group(1))
        toolkit_lengths = [int(value) for value in re.findall(r"\d+", len_match.group(1))]
        require(len(toolkit_sequences) == len(toolkit_lengths) == 20, "Official UAVDT selector length mismatch")
        require(toolkit_sequences == attribute_test_sequences, "M_attr test sequences differ from official DET evaluator")
        sequence_lengths = dict(zip(toolkit_sequences, toolkit_lengths))
        official_readme_name = TOOLKIT_PREFIX + "README.md"
        official_readme = read_zip_text(toolkit, official_readme_name)
        require("car(1), truck(2), bus(3)" in official_readme, "UAVDT class IDs not verified in local README")
        require("ignore areas" in official_readme.lower(), "UAVDT ignore-area evidence missing")
        eval_source_names = [
            overall_name,
            TOOLKIT_PREFIX + "utils/CalculateDetectionPR_obj.m",
            TOOLKIT_PREFIX + "utils/evalRes.m",
            TOOLKIT_PREFIX + "compOas.m",
            official_readme_name,
        ]
        for member in eval_source_names:
            atomic_extract_member(toolkit, member, evidence_root / Path(member).name)
        atomic_extract_member(attrs, "M_attr/readme.txt", evidence_root / "M_attr_readme.txt")

    expected_total = sum(sequence_lengths.values())
    require(expected_total == 16592, f"Locally verified UAVDT test sequence lengths unexpectedly sum to {expected_total}")

    sequence_entries: dict[str, list[zipfile.ZipInfo]] = {}
    with zipfile.ZipFile(IMAGE_ZIP) as images_zip:
        all_infos = images_zip.infolist()
        for sequence in toolkit_sequences:
            prefix = f"{IMAGE_PREFIX}{sequence}/"
            infos = sorted(
                [info for info in all_infos if info.filename.startswith(prefix) and info.filename.lower().endswith(".jpg")],
                key=lambda info: info.filename,
            )
            require(len(infos) == sequence_lengths[sequence], f"UAVDT image count mismatch for {sequence}")
            sequence_entries[sequence] = infos
            for info in infos:
                atomic_extract_member(images_zip, info.filename, images_root / sequence / Path(info.filename).name)

    gt_rows_by_sequence: dict[str, list[list[int]]] = {}
    ignore_rows_by_sequence: dict[str, list[list[int]]] = {}
    with zipfile.ZipFile(TOOLKIT_ZIP) as toolkit:
        for sequence in toolkit_sequences:
            whole_name = f"{TOOLKIT_PREFIX}GT/{sequence}_gt_whole.txt"
            ignore_name = f"{TOOLKIT_PREFIX}GT/{sequence}_gt_ignore.txt"
            whole_text = read_zip_text(toolkit, whole_name)
            ignore_text = read_zip_text(toolkit, ignore_name)
            gt_rows_by_sequence[sequence] = parse_numeric_rows(whole_text, 9, whole_name)
            ignore_rows_by_sequence[sequence] = parse_numeric_rows(ignore_text, 9, ignore_name)
            atomic_extract_member(toolkit, whole_name, gt_root / f"{sequence}_gt_whole.txt")
            atomic_extract_member(toolkit, ignore_name, gt_root / f"{sequence}_gt_ignore.txt")

    # This list is literal evidence from the bundled official evaluator. Other ignore files are empty/not applied.
    pass_match = re.search(r"pass\s*=\s*\{([^;]+)\};", overall_text)
    require(pass_match is not None, "Could not recover official UAVDT ignore-sequence policy")
    ignore_enabled_sequences = re.findall(r"'([^']+)'", pass_match.group(1))

    gt_by_frame: dict[tuple[str, int], list[list[int]]] = defaultdict(list)
    ignore_by_frame: dict[tuple[str, int], list[list[int]]] = defaultdict(list)
    invalid_boxes = 0
    outside_boxes = 0
    malformed_records = 0
    for sequence, rows in gt_rows_by_sequence.items():
        for row in rows:
            frame, _target, left, top, width, height, out_of_view, occlusion, category = row
            valid = 1 <= frame <= sequence_lengths[sequence] and width > 0 and height > 0 and category in (1, 2, 3)
            valid = valid and out_of_view in (1, 2, 3) and occlusion in (1, 2, 3, 4)
            if not valid:
                invalid_boxes += 1
            gt_by_frame[(sequence, frame)].append(row)
        for row in ignore_rows_by_sequence[sequence]:
            frame, _target, _left, _top, width, height, *_ = row
            if not (1 <= frame <= sequence_lengths[sequence] and width > 0 and height > 0):
                invalid_boxes += 1
            ignore_by_frame[(sequence, frame)].append(row)

    require(invalid_boxes == 0, f"UAVDT has {invalid_boxes} structurally invalid GT/ignore boxes")

    test_manifest = []
    gt_audit = []
    sequence_manifest = []
    coco_images = []
    coco_annotations = []
    ignore_payload: dict[str, list[list[int]]] = {}
    annotation_id = 1
    global_order = 0
    gt_object_count = 0
    ignore_region_count = 0
    source_class_for_target = {1: 3, 2: 5, 3: 8}
    for sequence in toolkit_sequences:
        sequence_gt_count = 0
        sequence_ignore_count = len(ignore_rows_by_sequence[sequence])
        ignore_region_count += sequence_ignore_count
        for frame in range(1, sequence_lengths[sequence] + 1):
            global_order += 1
            image_path = images_root / sequence / f"img{frame:06d}.jpg"
            require(image_path.is_file(), f"Missing extracted UAVDT frame: {image_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            require(image is not None, f"UAVDT image decode failed: {image_path}")
            height, width = image.shape[:2]
            frame_gt = gt_by_frame.get((sequence, frame), [])
            frame_ignore = ignore_by_frame.get((sequence, frame), []) if sequence in ignore_enabled_sequences else []
            car_count = truck_count = bus_count = 0
            frame_outside = 0
            for row in frame_gt:
                _frame, _target, left, top, box_width, box_height, _out, _occ, category = row
                if left < 0 or top < 0 or left + box_width > width or top + box_height > height:
                    frame_outside += 1
                    outside_boxes += 1
                if category == 1:
                    car_count += 1
                elif category == 2:
                    truck_count += 1
                elif category == 3:
                    bus_count += 1
                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": global_order,
                        "category_id": category,
                        "bbox": [left, top, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                        "out_of_view": row[6],
                        "occlusion": row[7],
                    }
                )
                annotation_id += 1
            sequence_gt_count += len(frame_gt)
            gt_object_count += len(frame_gt)
            image_hash = sha256(image_path)
            frame_key = f"{sequence}/img{frame:06d}.jpg"
            if frame_ignore:
                ignore_payload[frame_key] = [[row[2], row[3], row[4], row[5]] for row in frame_ignore]
            coco_images.append(
                {"id": global_order, "file_name": frame_key, "width": width, "height": height, "sequence": sequence, "frame": frame}
            )
            test_manifest.append(
                {
                    "order": global_order,
                    "sequence": sequence,
                    "frame": frame,
                    "image_id": frame_key,
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "image_sha256": image_hash,
                    "gt_objects": len(frame_gt),
                    "ignore_regions": len(frame_ignore),
                    "status": "PASS",
                }
            )
            gt_audit.append(
                {
                    "image_id": frame_key,
                    "gt_objects": len(frame_gt),
                    "car_objects": car_count,
                    "truck_objects": truck_count,
                    "bus_objects": bus_count,
                    "ignore_regions_applied": len(frame_ignore),
                    "boxes_extending_outside_image": frame_outside,
                    "invalid_boxes": 0,
                    "status": "PASS",
                }
            )
        sequence_manifest.append(
            {
                "sequence": sequence,
                "selector_M_attr_test": True,
                "selector_official_DET_evaluator": True,
                "expected_frames_from_official_evaluator": sequence_lengths[sequence],
                "actual_images": len(sequence_entries[sequence]),
                "gt_objects": sequence_gt_count,
                "ignore_rows_in_file": sequence_ignore_count,
                "ignore_filter_enabled_by_official_evaluator": sequence in ignore_enabled_sequences,
                "status": "PASS",
            }
        )

    require(global_order == 16592, f"UAVDT test manifest count mismatch: {global_order}")
    write_text(out / "UAVDT_TREE.txt", "\n".join(file_tree(UAVDT_ROOT)) + "\n")
    write_csv(out / "UAVDT_SEQUENCES_MANIFEST.csv", sequence_manifest)
    write_csv(out / "UAVDT_TEST_MANIFEST.csv", test_manifest)
    write_csv(out / "UAVDT_GT_AUDIT.csv", gt_audit)
    image_list = out / "uavdt_test_images.txt"
    write_text(image_list, "\n".join(row["image_path"] for row in test_manifest) + "\n")
    coco_path = out / "uavdt_test_coco_annotations.json"
    write_json(
        coco_path,
        {
            "info": {"description": "Stage15 standard UAVDT DET test GT adapter"},
            "licenses": [],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "truck"}, {"id": 3, "name": "bus"}],
        },
    )
    ignore_path = out / "uavdt_ignore_regions.json"
    write_json(
        ignore_path,
        {
            "schema": "stage15_uavdt_ignore_regions_v1",
            "enabled_sequences": ignore_enabled_sequences,
            "rule": "Remove a detection only when its full xywh box lies strictly inside an enabled ignore region, matching CalculateDetectionPR_overall.m.",
            "regions_by_image": ignore_payload,
        },
    )
    data_yaml = out / "uavdt_eval_data.yaml"
    write_text(
        data_yaml,
        "train: null\n"
        f"val: '{image_list.as_posix()}'\n"
        "nc: 10\n"
        "names: [pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor]\n",
    )
    mapping = {
        "schema": "stage15_uavdt_class_mapping_v1",
        "created_utc": utc_now(),
        "source_dataset": "VisDrone2019-DET",
        "target_dataset": "UAVDT",
        "mappings": [
            {"source_name": "car", "source_id_zero_based": 3, "target_name": "car", "target_id": 1},
            {"source_name": "truck", "source_id_zero_based": 5, "target_name": "truck", "target_id": 2},
            {"source_name": "bus", "source_id_zero_based": 8, "target_name": "bus", "target_id": 3},
        ],
        "source_ids_verified_from": str(STAGE15_ROOT / "00_PRECHECK" / "VISDRONE_CANONICAL_INFERENCE_CONFIG.json"),
        "target_ids_verified_from": str(evidence_root / "README.md"),
        "semantic_merges": False,
        "status": "PASS",
    }
    require([VISDRONE_CLASSES[index] for index in (3, 5, 8)] == ["car", "truck", "bus"], "VisDrone mapping assertion failed")
    write_json(out / "UAVDT_CLASS_MAPPING.json", mapping)
    ignore_report = f"""# UAVDT Ignore Policy

Status: **PASS**

The bundled official DET evaluator (`CalculateDetectionPR_overall.m`) was
recovered from `{TOOLKIT_ZIP}`. It removes detections whose complete bounding
box lies strictly inside a listed ignore region for its explicit 18-sequence
`pass` list. Stage15 reproduces that filtering before all metric aggregation.

- Ignore files retained unchanged: 20
- Ignore rows: {ignore_region_count}
- Enabled sequences: {', '.join(ignore_enabled_sequences)}
- Ground-truth out-of-view and occlusion fields are retained as metadata; the
  official overall evaluator does not exclude ordinary GT by these attributes.
- No GT or prediction box is clipped or silently removed.
"""
    write_text(out / "UAVDT_IGNORE_POLICY.md", ignore_report)
    prep = {
        "schema": "stage15_uavdt_prep_config_v1",
        "created_utc": utc_now(),
        "status": "PASS",
        "source_root": str(UAVDT_ROOT),
        "source_read_only": True,
        "source_archives": [
            {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in (ATTR_ZIP, IMAGE_ZIP, TOOLKIT_ZIP)
        ],
        "test_selector_evidence": {
            "M_attr_test_sequences": attribute_test_sequences,
            "official_evaluator_sequences": toolkit_sequences,
            "exact_match": True,
        },
        "test_sequences": 20,
        "test_frames": global_order,
        "expected_prompt_count": 15069,
        "verified_local_count": global_order,
        "count_classification": "VERIFIED_LOCAL_STANDARD_SPLIT_COUNT_DIFFERENT_FROM_EXPECTED",
        "count_explanation": "The bundled official CalculateDetectionPR_overall.m hard-codes the same 20 M_attr/test sequences and seqLens summing to 16,592; the image archive matches every seqLen exactly.",
        "gt_objects": gt_object_count,
        "boxes_extending_outside_image": outside_boxes,
        "outside_box_policy": "Retained exactly; out-of-view is an annotated UAVDT attribute and boxes are not clipped.",
        "invalid_boxes": invalid_boxes,
        "malformed_records": malformed_records,
        "reported_metrics": ["AP@[0.50:0.95]", "AP50", "AP75", "Recall", "Precision", "F1", "AP_car", "AP_truck", "AP_bus"],
        "APS_reported": False,
        "APS_decision": "PRE_PREDICTION_NOT_INCLUDED; no new small-object endpoint introduced for Stage15",
        "test_manifest": str(out / "UAVDT_TEST_MANIFEST.csv"),
        "test_manifest_sha256": sha256(out / "UAVDT_TEST_MANIFEST.csv"),
        "coco_annotation_path": str(coco_path),
        "coco_annotation_sha256": sha256(coco_path),
        "ignore_regions_path": str(ignore_path),
        "ignore_regions_sha256": sha256(ignore_path),
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": sha256(data_yaml),
        "target_to_source_class_ids": source_class_for_target,
    }
    write_json(out / "UAVDT_PREP_CONFIG.json", prep)
    report = f"""# UAVDT Dataset Audit

Status: **PASS**

- Raw source: `{UAVDT_ROOT}` (read-only ZIP packages)
- Train/test sequence identities from `M_attr`: 30 / 20, overlap 0
- Official DET evaluator test identities: exact match with `M_attr/test`
- Test sequences: 20
- Test frames verified against official `seqLens`: {global_order}
- Test ground-truth objects: {gt_object_count}
- Classes: car(1), truck(2), bus(3), verified from bundled README
- Invalid boxes: {invalid_boxes}
- Malformed records: {malformed_records}
- Boxes extending outside images: {outside_boxes}, retained without clipping

The often-cited expected count of 15,069 was not used as a selector. The local
official evaluator and image archive independently agree on 16,592 frames for
the same 20 standard test sequences, so Stage15 records the local verified
count rather than forcing an unsupported count.
"""
    write_text(out / "UAVDT_DATASET_AUDIT.md", report)
    write_text(out / "UAVDT_BLOCKING_ISSUES.md", "# UAVDT Blocking Issues\n\nNone. The verified local frame count differs from the prompt expectation and is explicitly documented.\n")
    print("UAVDT_PREP=PASS")
    print("UAVDT_TEST_SEQUENCES=20")
    print(f"UAVDT_TEST_FRAMES={global_order}")
    print(f"UAVDT_GT_OBJECTS={gt_object_count}")
    print("UAVDT_INFERENCE_STARTED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

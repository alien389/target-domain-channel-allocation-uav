from __future__ import annotations

import json
from pathlib import Path

import cv2

from stage15_common import (
    CARPK_ROOT,
    STAGE15_ROOT,
    VISDRONE_CLASSES,
    file_tree,
    require,
    sha256,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


def parse_annotation(path: Path) -> list[tuple[int, int, int, int, int]]:
    rows: list[tuple[int, int, int, int, int]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        require(len(fields) == 5, f"Malformed CARPK annotation {path}:{number}: {line}")
        rows.append(tuple(int(value) for value in fields))
    return rows


def main() -> int:
    out = STAGE15_ROOT / "01_CARPK_PREP"
    out.mkdir(parents=True, exist_ok=True)
    data = CARPK_ROOT / "CARPK_devkit" / "data"
    images_dir = data / "Images"
    annotations_dir = data / "Annotations"
    train_list = data / "ImageSets" / "train.txt"
    test_list = data / "ImageSets" / "test.txt"
    local_reader = CARPK_ROOT / "tool" / "draw_bounding_boxes.py"
    for path in (images_dir, annotations_dir, train_list, test_list, local_reader):
        require(path.exists(), f"Missing CARPK source evidence: {path}")

    train_ids = [line.strip() for line in train_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    test_ids = [line.strip() for line in test_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    require(len(train_ids) == len(set(train_ids)) == 989, "CARPK canonical train list is not 989 unique IDs")
    require(len(test_ids) == len(set(test_ids)) == 459, "CARPK canonical test list is not 459 unique IDs")
    require(not (set(train_ids) & set(test_ids)), "CARPK train/test ID overlap detected")

    image_paths = sorted(images_dir.glob("*.png"))
    annotation_paths = sorted(annotations_dir.glob("*.txt"))
    require(len(image_paths) == 1448, f"Expected 1448 CARPK images, found {len(image_paths)}")
    require(len(annotation_paths) == 1448, f"Expected 1448 CARPK annotations, found {len(annotation_paths)}")
    require({path.stem for path in image_paths} == set(train_ids) | set(test_ids), "CARPK images do not equal split union")
    require({path.stem for path in annotation_paths} == set(train_ids) | set(test_ids), "CARPK annotations do not equal split union")

    image_manifest = []
    test_manifest = []
    gt_audit = []
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    invalid_boxes = 0
    malformed_annotations = 0
    gt_objects = 0

    split_by_id = {item: "train" for item in train_ids}
    split_by_id.update({item: "test" for item in test_ids})
    test_order = {item: index + 1 for index, item in enumerate(test_ids)}
    for image_path in image_paths:
        image_id = image_path.stem
        annotation_path = annotations_dir / f"{image_id}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        require(image is not None, f"CARPK image decode failed: {image_path}")
        height, width = image.shape[:2]
        rows = parse_annotation(annotation_path)
        image_hash = sha256(image_path)
        annotation_hash = sha256(annotation_path)
        row_invalid = 0
        for x1, y1, x2, y2, class_id in rows:
            valid = class_id == 1 and 0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height
            if not valid:
                row_invalid += 1
                invalid_boxes += 1
            if split_by_id[image_id] == "test":
                # The bundled reader draws through R+1/B+1; preserve inclusive CARPK endpoints.
                box_width = x2 - x1 + 1
                box_height = y2 - y1 + 1
                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": test_order[image_id],
                        "category_id": 1,
                        "bbox": [x1, y1, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        image_manifest.append(
            {
                "image_id": image_id,
                "split": split_by_id[image_id],
                "image_path": str(image_path),
                "annotation_path": str(annotation_path),
                "width": width,
                "height": height,
                "image_sha256": image_hash,
                "annotation_sha256": annotation_hash,
                "gt_objects": len(rows),
            }
        )
        if split_by_id[image_id] == "test":
            gt_objects += len(rows)
            test_manifest.append(
                {
                    "order": test_order[image_id],
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "annotation_path": str(annotation_path),
                    "width": width,
                    "height": height,
                    "image_sha256": image_hash,
                    "annotation_sha256": annotation_hash,
                    "gt_objects": len(rows),
                    "status": "PASS" if row_invalid == 0 else "INVALID_GT",
                }
            )
            gt_audit.append(
                {
                    "image_id": image_id,
                    "gt_objects": len(rows),
                    "invalid_boxes": row_invalid,
                    "missing_annotation": False,
                    "malformed_records": 0,
                    "class_ids": "1" if rows else "",
                    "status": "PASS" if row_invalid == 0 else "FAIL",
                }
            )
            coco_images.append(
                {
                    "id": test_order[image_id],
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                }
            )

    require(invalid_boxes == 0, f"CARPK has {invalid_boxes} invalid test boxes")
    require(malformed_annotations == 0, "CARPK has malformed annotations")
    test_manifest.sort(key=lambda row: int(row["order"]))
    write_text(out / "CARPK_TREE.txt", "\n".join(file_tree(CARPK_ROOT)) + "\n")
    write_csv(out / "CARPK_IMAGES_MANIFEST.csv", image_manifest)
    write_csv(out / "CARPK_TEST_MANIFEST.csv", test_manifest)
    write_csv(out / "CARPK_GT_AUDIT.csv", gt_audit)
    test_image_list = out / "carpk_test_images.txt"
    write_text(test_image_list, "\n".join(row["image_path"] for row in test_manifest) + "\n")
    coco_path = out / "carpk_test_coco_annotations.json"
    write_json(
        coco_path,
        {
            "info": {"description": "Stage15 canonical CARPK test GT adapter"},
            "licenses": [],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "car"}],
        },
    )
    data_yaml = out / "carpk_eval_data.yaml"
    write_text(
        data_yaml,
        "train: null\n"
        f"val: '{test_image_list.as_posix()}'\n"
        "nc: 10\n"
        "names: [pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor]\n",
    )
    mapping = {
        "schema": "stage15_carpk_class_mapping_v1",
        "created_utc": utc_now(),
        "source_dataset": "VisDrone2019-DET",
        "source_class_name": "car",
        "source_class_id_zero_based": VISDRONE_CLASSES.index("car"),
        "target_dataset": "CARPK",
        "target_class_name": "car",
        "target_annotation_class_id": 1,
        "target_annotation_semantics": "The fifth bundled annotation field equals 1 for every verified test object.",
        "other_source_classes_mapped": False,
        "evidence_source": [
            str(STAGE15_ROOT / "00_PRECHECK" / "VISDRONE_CANONICAL_INFERENCE_CONFIG.json"),
            str(local_reader),
            str(out / "CARPK_GT_AUDIT.csv"),
        ],
        "status": "PASS",
    }
    write_json(out / "CARPK_CLASS_MAPPING.json", mapping)
    prep = {
        "schema": "stage15_carpk_prep_config_v1",
        "created_utc": utc_now(),
        "status": "PASS",
        "source_root": str(CARPK_ROOT),
        "source_read_only": True,
        "canonical_split_source": str(test_list),
        "canonical_split_source_sha256": sha256(test_list),
        "test_images": len(test_manifest),
        "test_gt_objects": gt_objects,
        "annotation_format": "x1 y1 x2 y2 class_id",
        "coordinate_convention": "integer inclusive x2/y2, proven by bundled draw loop through R+1/B+1",
        "coco_adapter": "xywh with width=x2-x1+1 and height=y2-y1+1; no clipping required",
        "invalid_box_handling": "hard stop; none found",
        "reported_metrics": ["AP@[0.50:0.95]", "AP50", "AP75", "Recall", "Precision", "F1"],
        "APS_reported": False,
        "APS_decision": "PRE_PREDICTION_NOT_INCLUDED; not required for the CARPK external endpoint",
        "test_manifest": str(out / "CARPK_TEST_MANIFEST.csv"),
        "test_manifest_sha256": sha256(out / "CARPK_TEST_MANIFEST.csv"),
        "coco_annotation_path": str(coco_path),
        "coco_annotation_sha256": sha256(coco_path),
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": sha256(data_yaml),
    }
    write_json(out / "CARPK_PREP_CONFIG.json", prep)
    report = f"""# CARPK Dataset Audit

Status: **PASS**

- Raw source: `{CARPK_ROOT}` (read-only)
- Total images/annotations: 1,448 / 1,448
- Canonical train/test lists: 989 / 459 unique IDs
- Train-test overlap: 0
- Canonical Stage15 test images: 459
- Test ground-truth objects: {gt_objects}
- Missing annotations: 0
- Malformed records: 0
- Invalid boxes: 0
- Scope: car only, VisDrone source class ID 3 (zero based)

The bundled annotation reader interprets records as `x1 y1 x2 y2 class`
and draws through both endpoints. The adapter therefore uses inclusive
width/height (`x2-x1+1`, `y2-y1+1`) without clipping or object removal.
"""
    write_text(out / "CARPK_DATASET_AUDIT.md", report)
    write_text(out / "CARPK_BLOCKING_ISSUES.md", "# CARPK Blocking Issues\n\nNone.\n")
    print("CARPK_PREP=PASS")
    print("CARPK_TEST_IMAGES=459")
    print(f"CARPK_GT_OBJECTS={gt_objects}")
    print("CARPK_INFERENCE_STARTED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


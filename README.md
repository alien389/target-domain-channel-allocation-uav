# Target-Domain Dependency-Aware Channel Allocation for Efficient UAV Object Detection

Source code, recorded configurations, software requirements and exact dataset
split definitions supporting the research study. The repository includes the
channel-allocation search and the additional comparative/ablation methods in a
single functional code tree. It contains **no image data, trained weights, raw
results, predictions, energy logs or private review materials**.

## Repository structure

| Location | Content |
|---|---|
| `code/study/` | Source preparation, dependency/Taylor scoring, population/search, F1–F3, architecture locking, final training, evaluation and hardware export |
| `code/baselines/` | DepGraph Group-L1, BN-gamma magnitude pruning, scratch comparison, allocation transplant and KD/NoKD continuation |
| `code/pruning_research/` | Shared dependency, channel importance and pruning implementation |
| `code/yolov5/` | Modified YOLOv5 code required by the study, with its upstream license |
| `code/external/` | CARPK/UAVDT data preparation and evaluation tools |
| `code/run_search.py`, `code/train_final_models.py`, `code/evaluate_models.py` | Entry points to the authentic research scripts |
| `code/prepare_data_splits.py` | Optional generation of localized JSON/YAML configurations |
| `configs/` | Recorded model, source and per-seed configurations |
| `data/` | Exact ordered image/frame identifiers defining the evaluated partitions |
| `requirements*.txt` | Reconstructed Python requirements and Jetson export dependencies |

## Research methods and source locations

| Method | Key source |
|---|---|
| Data split and source preparation | `code/study/scripts/stage01_data_split.py`, `stage02_prepare_search_parent.py` |
| Global-Taylor / 35-group dependency-aware search space | `code/study/scripts/stage03a_taylor_dependency_gene_space.py`, `code/pruning_research/pruning/` |
| Shared initialization and multi-fidelity search | `code/study/scripts/stage03b_gene_space_lock_shared_p0.py`, `code/study/05_full_fidelity1_search/`, `07_fidelity2_confirmation/`, `08a_fidelity3_proposed_lock/`, `08b_fidelity3_matched_controls/` |
| Frozen allocation candidates | `code/study/09_locked_architectures/` |
| Final three-seed training (300 epochs) | `code/study/10_final_fulltrain_300e_matched/scripts/stage10_run_training.py` |
| Val548 / held-out test-dev1610 evaluation | `code/study/scripts/stage11_evaluate_one.py`, `stage11b_evaluate_one.py` |
| Model export, selective quantization and hardware measurement | `code/study/12_jetson_fp16_deployment/13_scripts/` |
| DepGraph-L1 baseline | `code/baselines/depgraph/` |
| Global BN-gamma baseline | `code/baselines/bn_gamma/` |
| Weight-inheritance comparison | `code/baselines/scratch/` |
| Joint P3/P4/P5 transplant ablation | `code/baselines/transplant/` |
| KD versus NoKD continuation | `code/baselines/kd/` |
| CARPK / UAVDT transfer | `code/external/scripts/` |

The two conventional pruning comparisons differ in their channel-importance
criteria. BN-gamma magnitude pruning here does **not** include sparse
pretraining and is not the full Network Slimming method. The joint transplant
involves backbone compensation and is not a path-only causal experiment.

## Stage-script identities (not software versions)

The letters in `stage03a`, `stage03b`, `stage06a` and `stage11b` denote **different protocol steps or evaluation partitions**, not alternative copies of a script. Preserve their original filenames because other research scripts load them by filename.

| Script(s) | Role and usage |
|---|---|
| `stage01_data_split.py` | Prepare and audit the train-derived search split. |
| `stage02_prepare_search_parent.py` | Prepare the locked source-adapted SearchParent. |
| `stage02_explicit_inner_val_eval.py`, `run_visdrone_internal_eval_stage02.m` | Inner-val evaluation and its MATLAB VisDrone toolkit wrapper. The `.m` file is not an obsolete Python revision. |
| `visdrone_toolkit_python_port.py` | Python implementation of the VisDrone evaluation protocol, used by later evaluation scripts. |
| `stage03a_taylor_dependency_gene_space.py` | Original Stage03a implementation and **imported support module** for the subsequent corrected protocol and R1 repair. Its initial standalone draft must not be mistaken for the finalized Taylor/R1 pipeline. |
| `stage03a_resolve_taylor_protocol.py` | Resolve the historical Taylor protocol and exclude the superseded 512-image/32-batch evidence in favor of the documented corrected protocol. |
| `stage03a_r1_gene_space_repair.py` | Apply the R1 gene-space repair used by the locked allocation space; imports the Stage03a support module. |
| `stage03b_gene_space_lock_shared_p0.py` | Lock the repaired gene space and shared initial population. Depends on R1 repair outputs and code. |
| `stage04_shared_p0_fidelity1.py`, `evaluate_stage04_inner_val.py` | Run Stage04 F1 candidates and evaluate their inner-val predictions. |
| `train_structural_recovery_drone.py` | Shared training interface used by F1 and later search/recovery steps. |
| `stage06a_fidelity2_promotion_lock.py` | Lock candidate promotion from F1 to F2. This is a separate step, not version 6 of Stage03a. |
| `stage11_evaluate_one.py`, `stage11_statistics.py` | Evaluate and analyze **official VisDrone val548**. |
| `stage11b_evaluate_one.py`, `stage11b_statistics.py` | Evaluate and analyze **annotated held-out test-dev1610**. These do not replace the Stage11 val548 scripts. |

The retained source tree contains no byte-identical duplicate files. Retention denotes a source dependency or distinct scientific operation, **not** certification that every historical stage can be replayed from a fresh checkout. Original file hashes differ from the private historical sources where release-only path relocation was required.

## Python environment

The supplied `requirements.txt` is reconstructed from saved sources rather
than a verified historical package lock. Recorded Stage10 evidence includes
Python 3.10.19, PyTorch 2.10.0+cu128 and CUDA runtime 12.8; other historical
package versions are incomplete. Jetson deployment uses a separate TensorRT
8.5.2.2-era environment and compatible export libraries. The software
requirements are **not guaranteed to install as a single universal environment**.
Install a compatible PyTorch build for the target hardware before resolving
remaining requirements. `requirements-jetson.txt` is platform-specific.

From the repository root, the code import path is `code/` and the YOLOv5
modules are in `code/yolov5/`. On Linux/macOS an example is:

```sh
export PYTHONPATH="$PWD/code:$PWD/code/yolov5"
python code/run_search.py --stage f1 --source-interface
python code/train_final_models.py --stage stage10 --source-interface
python code/evaluate_models.py --stage testdev1610 --source-interface
```

These commands inspect source argument declarations only. They do not run
training or demonstrate full reproduction. To execute a stage, consult its
real command-line arguments and supply its authentic checkpoints, data and
intermediate artifacts. The launcher does not synthesize missing inputs.

## Datasets and exact splits

Data and annotations must be obtained separately from their providers. Only
ordered filenames/identifiers are redistributed.

| Dataset | Official source | Included exact study membership |
|---|---|---|
| VisDrone2019-DET | https://github.com/VisDrone/VisDrone-Dataset | `data/train6471_ordered_filenames.txt` (6,471), `data/meta_train5177_ordered_filenames.txt` (5,177), `data/meta_inner_val1294_ordered_filenames.txt` (1,294), `data/val548_ordered_filenames.txt` (548), `data/testdev1610_ordered_filenames.txt` (1,610) |
| CARPK | https://winstonhsu.info/drone-view-object-counting-dataset-carpk/ | `data/carpk459_ordered_filenames.txt` (459) |
| UAVDT | https://sites.google.com/view/grli-uavdt/ | `data/uavdt16592_ordered_image_ids.txt` (16,592 frames) and `data/uavdt20_sequences.csv` (20 sequences) |
| COCO 2017 | https://cocodataset.org/ | Pretrained source provenance only; no weights or raw records included |

The `data/archived_calibration1024_ordered_ids.txt` file records a saved
calibration candidate order. This alone cannot establish exactly which prior
TensorRT engine consumed those records. VisDrone *test-dev* in this study is
its publicly annotated test-dev subset, not the blind challenge submission
split. COCO-style `AP_S` is a supplementary object-size metric.

## Configuration and training policy

The research uses real recorded source/configuration files. Prepare local
inputs using actual dataset paths and external source checkpoints. Environment
root variables are resolved through `code/release_paths.py`; for instance
`STUDY_ROOT`, `YOLOV5_ROOT` and `DATASETS_ROOT`. Missing external roots
are errors rather than automatically downloaded assets. To materialize a new
localized JSON/YAML file without changing the recorded template:

```sh
python code/prepare_data_splits.py \
  --template configs/source_weight_inputs.json \
  --output local_source_weights.json
```

The original 300-epoch final model selection uses terminal EMA and does not
select the best epoch based on official validation. The added baselines use
their separate initialization and continuation rules recorded in source and
per-seed configs. Keep the train-derived 5,177/1,294 search split separate
from official val548 and annotated held-out test-dev1610 evaluations.

## Reproduction boundaries

This is a public **source-and-configuration release**, not a verified
bit-for-bit rerun of every historical stage from a fresh checkout. Original
model/data/checkpoint integrity safeguards in the scientific scripts remain
intact. Some executable source/config hashes refer to the historical private
files and may require reconciliation after the portability-only relocation.
No guard may be silently bypassed or restated as passed without genuine
evidence. Historical generated search artifacts, model weights, datasets,
and compatible hardware/software environments must be obtained separately.

## License and attribution

The author-owned research code is released under GNU AGPL-3.0, consistent
with the modified Ultralytics YOLOv5 code. Preserve the upstream source
notices and `LICENSE`. See `THIRD_PARTY_NOTICES.md` for the original source
and third-party information. Dataset files and pretrained weights are not
redistributed.

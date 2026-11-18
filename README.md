# Target-domain channel allocation for efficient UAV detection

Published R1 release and archival references

Manuscript: MJCSC-D-26-00121

GitHub Repository: 
https://github.com/alien389/target-domain-channel-allocation-uav

Complete R1 Evidence Release (16 archives and SHA-256 integrity file): 
https://github.com/alien389/target-domain-channel-allocation-uav/releases/tag/evidence-r1-v1.0.0

Zenodo R1 Version DOI (current release): 
https://doi.org/10.5281/zenodo.24347341

Zenodo Concept DOI (all versions):
https://doi.org/10.5281/zenodo.23977144

Previous v1.0.0 Version DOI (historical release): 
https://doi.org/10.5281/zenodo.23977145

The complete R1 experimental evidence archives are hosted as GitHub Release Assets. Zenodo preserves the corresponding tagged software repository snapshot. The Zenodo record does not independently contain the complete experimental evidence archives.

File names, sizes, and SHA-256 checksums are documented in the repository's reproducibility manifests and the release integrity file.


Research evidence for MJCSC-D-26-00121R1. This is a local, pre-publication
overlay for the existing source repository. The staged Release assets include
15 terminal-EMA checkpoints, per-image predictions for 30 external runs,
Table 14 prediction outputs, and the 160 original VDD_IN power traces plus the
pre-campaign idle trace.

Start with the [H2 Evidence Index](reproducibility/evidence-index.csv), which
maps H2-01 through H2-10 to repository files, Release assets and replay commands.
Use the [SHA-256 Manifest](reproducibility/manifests/files.csv) for exact
filenames, sizes and checksums. The Release asset set also includes a
`SHA256SUMS.txt` file.

## Core Reproducibility

Use the repository files and the following nine Release archives to replay the
reported evaluations and supporting calculations. Obtain provider annotations
separately as described below.

| Evidence | Core Release archives |
|---|---|
| Search and process-time records | `native-search-and-process-records.zip` |
| Power traces and timing | `raw-energy-originals.zip` |
| External predictions | `external-carpk-all-seeds.zip`, `external-uavdt-seed-42.zip`, `external-uavdt-seed-43.zip`, `external-uavdt-seed-44.zip` |
| Table 14 evaluator inputs and capture metadata | `table14-full-split-predictions-and-capture-metadata.zip` |
| Terminal-EMA checkpoints and corresponding source | `terminal-ema-15-safetensors.zip`, `yolov5-checkpoint-corresponding-source.zip` |

For a Core-only download, compare each downloaded archive with its SHA-256
entry in the manifest, then run the relevant replay command below.

## Advanced Tensor-Level Verification

The seven `table14-trt-tensor-verification-*.zip` archives preserve the raw
TensorRT tensor bits for lower-level checks. They are not needed to replay
Table 14 AP from the Core evaluator inputs. Use the
[Table 14 member index](reproducibility/manifests/table14-member-index.csv)
to find the archive and member for a given capture.

`verify_release.py --full` requires **all 16 archives**, including **all seven
Advanced tensor archives**. It is not a Core-only verification command.

## Verify and use

1. Install dependencies with `pip install -r reproducibility/requirements-audit.txt`.
   For the complete local candidate, keep `GITHUB_REPOSITORY_PATCH/` and
   `GITHUB_RELEASE_ASSETS/` as siblings. From the repository overlay,
   `python -B reproducibility/scripts/verify_release.py ..` checks the complete
   staged layout. Append `--full` to check archive hashes, CRCs, replay,
   energy integration and tensor equality.
2. Obtain source data separately from the [VisDrone2019-DET provider](https://github.com/VisDrone/VisDrone-Dataset),
   [CARPK provider](https://lafi.github.io/LPN/) and
   [UAVDT provider](https://sites.google.com/view/grli-uavdt). Match the
   [locked data roles](reproducibility/H2_DATASET_ROLE_AND_PROVENANCE.md) and
   [ordered split IDs](reproducibility/evaluation/splits/). No images, labels
   or ignore masks are distributed. Build local GT with `python -B reproducibility/scripts/prepare_local_gt.py DATASET SPLIT.csv ANNOTATIONS_DIR LOCAL_GT.json`.
   For UAVDT, use the provider's per-sequence `*_gt_whole.txt`; ignore filtering of predictions is already reflected in the archived COCO output.
3. For external AP, find the model, seed and dataset in the
   [prediction index](reproducibility/manifests/external-payloads.csv).
   Run `python -B reproducibility/scripts/payloads.py external ASSET.zip MEMBER OUT`
   to obtain exact `raw.jsonl`, `adapted.jsonl` and `coco.json`. Check
   `python -B reproducibility/scripts/external_coco_replay.py carpk LOCAL_GT.json OUT/coco.json`
   (replace `carpk` with `uavdt` as appropriate). Use the original
   converted local COCO GT, with 1-based IDs in split order. The COCO output
   already contains the archived adapter's retained detections.
4. For Table 14, run `python -B reproducibility/scripts/table14_replay.py . ../GITHUB_RELEASE_ASSETS --write OUT`.
   Outputs are sorted into historical rectangle PT, historical square TRT and
   common square PT/TRT arms. Run
   `python -B reproducibility/scripts/visdrone_toolkit_python_port.py --help`
   for the paired prediction/annotation/image directory CLI. Use each split's
   provider data and the recorded annotation projection and
   [backend protocol](reproducibility/evaluation/backend-parity/matched-backend-protocol.json).
   Raw PyTorch tensors were not retained.
5. Load a checkpoint with the [tensor-only guide](reproducibility/checkpoints/README.md).
   For energy, run `python -B reproducibility/scripts/replay_tables.py energy ../GITHUB_RELEASE_ASSETS/raw-energy-originals.zip`.
   Use `replay_tables.py search-cost .` for exact-match Kendall and process-time
   totals, and `replay_bootstrap.py reproducibility/analysis` for locked CIs.

The energy archive contains 160 unmodified VDD_IN traces and one pre-campaign
idle reference, not run-matched idle measurements. Gross energy integrates the
exact timed loop; revised net subtracts the same pre-campaign reference.
Table 7 is job process time, not measured GPU-active time. Prior test-dev
exposure is disclosed in the data-role record. Exploratory CIs use fixed
checkpoints and the stated image/sequence resampling units.

The previous [Zenodo record](https://doi.org/10.5281/zenodo.23977145) is a
different, immutable code/configuration snapshot. See [NOTICE](NOTICE.md) for
source and dataset attribution.

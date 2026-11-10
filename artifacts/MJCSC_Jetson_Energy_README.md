# Jetson Xavier NX: TensorRT power, energy, and latency evidence

## Scope

This directory provides the original accepted-run measurement evidence for the Jetson Xavier NX power-profiling analysis in *Target-Domain Dependency-Aware Channel Allocation for Efficient Unmanned Aerial Vehicle Object Detection* (MJCSC-D-26-00121R1).

- **Engines:** 16 TensorRT engine configurations.
- **Workload protocols:** end-to-end (E2E) processing and forward-only inference.
- **Accepted repetitions:** five for each engine and protocol (160 workload runs in total).
- **Device operating point:** 1109.25 MHz recorded GPU frequency, with EMC locked at 1866 MHz during the accepted measurements.
- **Idle reference:** one independently recorded, 300-second pre-campaign `VDD_IN` trace (`E01_pre_attempt1.csv`). Its time-weighted mean is **4.1284821492431 W**. This is one pre-campaign baseline, **not** a separately validated matched-session baseline for every later profiling run.

## Archived files

The accompanying file `MJCSC_A1_Original_160_Power_Traces_and_Single_Idle.zip` contains:

| Path inside the ZIP | Content |
| --- | --- |
| `Original_Evidence/A1_ALL_160_ACCEPTED_RUNS.csv` | All 160 accepted workload-run records and associated timing, power, energy, hardware-state and acquisition metadata. |
| `Original_Evidence/RAW_POWER_TRACES/` | 160 original time-series `VDD_IN` power traces, one per accepted workload run. |
| `Original_Evidence/RAW_IDLE_TRACES/E01_pre_attempt1.csv` | Original unmodified power trace for the chosen 300-second idle reference. |
| `Original_Evidence/RAW_IDLE_TRACES/E01_pre_attempt1.window.json` | Start and end bounds, completion status, and sampling-coverage metadata for the reference idle window. |
| `Original_Evidence/RAW_IDLE_TRACES/E01_pre_attempt1.logger.json` | Original idle logger metadata. |
| `Original_Evidence/idle_checkpoints/E01_pre.json` | Time-weighted idle-energy integration results and measurement-quality evidence. |
| `SHA256_Original_Evidence.csv` | Per-file SHA-256 integrity records. |

The original archive contains only the 160 accepted workload traces and the chosen pre-campaign idle raw trace. Rejected workload traces and subsequent idle raw traces are not included in this curated archive.

## Reproducing the reported energy quantities

The columns `gross_j_item`, `weighted_power_w`, `logger_duration_s`, `duration_s`, `t0_ns`, `t1_ns`, and `items` in `A1_ALL_160_ACCEPTED_RUNS.csv` provide the needed per-run measurements.

- **Gross energy per item** (`gross_j_item`) is the integrated measured `VDD_IN` board-input energy over `[t0_ns, t1_ns]`, divided by `items`.
- **Mean board-input power** (`weighted_power_w`) is the integrated gross energy over the same timed-loop interval, divided by `duration_s`.
- **Timed-loop duration** is `duration_s`, corresponding to `(t1_ns - t0_ns) / 1e9`.
- **Logger duration** is `logger_duration_s`, separately recorded from the sampling logger's boundaries. It must not be substituted for the timed-loop integration interval.
- **Reference-corrected net energy per item**, when required, is calculated as `gross_j_item - 4.1284821492431 * duration_s / items` in joules per item.

The CSV also retains historical diagnostic fields such as `matched_idle_w`, `net_j_item`, and `net_valid`. These fields record intermediate per-block diagnostic calculations, **not** the uniformly referenced net-energy values used in the revised manuscript. Derive those values from `gross_j_item`, `duration_s`, `items`, and the single reference stated above. Do not treat the historical diagnostic fields as an alternative primary analysis.

For every accepted engine/protocol combination, compute the sample mean and sample standard deviation from its five accepted runs. E2E and forward-only are independently executed workload measurements. Their difference does not isolate preprocessing or postprocessing energy causally.

## Verification and provenance

SHA-256 of the ZIP archive:

`ccd69dee30e8d0baa3a63b5ac5a946a0dd60c00be0b70d631aa0086a16ae40e1`

The archive includes the source-file checksum manifest. The raw traces are preserved without modification. The original profiling evidence is distinct from the historical process-memory campaign, whose results are reported separately in the manuscript.

## Data access

No third-party benchmark images or dataset binaries are redistributed here. The study's original data splits and associated evaluation scripts are documented elsewhere in the repository.

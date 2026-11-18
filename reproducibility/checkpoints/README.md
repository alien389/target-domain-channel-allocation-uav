# Verify and load a fixed terminal EMA

Download and verify `terminal-ema-15-safetensors.zip` and the paired
`yolov5-checkpoint-corresponding-source.zip` using the
[public file manifest](../manifests/files.csv). Extract each separately.
The [checkpoint index](checkpoint-index.csv) maps all five allocations and
three seeds to the sanitized derivative, original SHA-256, graph and lineage.
Install PyTorch and the extracted source tree's requirements. From the two
unpacked folders, set `PYTHONPATH` to `source/yolov5`, then run:

```sh
python source/load_checkpoint.py weights/meta-ap-aps-b6-seed-42.safetensors weights/meta-ap-aps-b6-seed-42.graph.json
```

The command only assembles an evaluation-mode CPU module; it does not run a
forward pass. Every stored state/buffer and auxiliary tensor is type-, shape-
and byte-equal to the previously verified native checkpoint inventory.
See the [corresponding-source notice](WEIGHT_LICENSE_AND_SOURCE_COMPLIANCE.md).

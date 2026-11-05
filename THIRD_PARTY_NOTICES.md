# Third-party source and licensing

This source release includes modified Ultralytics YOLOv5 code. The original
Ultralytics copyright notices and GNU Affero General Public License
version 3 (AGPL-3.0) are retained in the original source headers and the repository-root `LICENSE`.
The research author has authorized public release of the author-owned code.
The combined redistributed source is supplied under AGPL-3.0, subject to the
rights and notices retained in its component files. No ownership of third-party
software or datasets is asserted.

- Upstream YOLOv5: https://github.com/ultralytics/yolov5
- Local YOLOv5 source base commit: `88af13e3e25602344aab68d44f9d0b666dacf695`
- External `torch-pruning`: https://github.com/VainF/Torch-Pruning (not bundled)
- Data remain with their respective dataset providers (not bundled).

This repository contains research-specific changes to the original YOLOv5
source. The included code is a subset tailored to the experiments and does not
claim to implement every upstream export backend.

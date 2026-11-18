# Dataset roles and provenance

The authors confirm original acquisition and experimental use of **VisDrone2019-DET-test-dev (heldout1610)**. The prior independent local comparison found matching SHA-256 digests for all 1,610 images and all 1,610 corresponding annotation files between the two retained replicas. This was a local comparison, not a comparison with an external archive.

- [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset): search uses 5,177 train-derived images and 1,294 inner validation images. Final fitting uses all 6,471 training images. val548 and heldout1610 are second-paper evaluation splits. Earlier test-dev exposure in a fixed source-selected prior-work bank is disclosed; it was not used for second-paper ranking, promotion, training, architecture lock, threshold selection or winner selection. See the [archived split record](evaluation/backend-parity/heldout-provenance-audit.json).
- [CARPK](https://lafi.github.io/LPN/): canonical 459-image test set, evaluated zero shot.
- [UAVDT](https://sites.google.com/view/grli-uavdt): 16,592 test frames in 20 sequences, evaluated zero shot using the recorded ignore adapter. The exploratory analysis resamples sequences.
- [COCO](https://cocodataset.org/): source-domain pretraining and prior pruning/KD provenance. Not a target-domain evaluation split.

[Split IDs and fingerprints](evaluation/splits/) do not include third-party image or annotation bytes. Use the respective providers' data-access terms.

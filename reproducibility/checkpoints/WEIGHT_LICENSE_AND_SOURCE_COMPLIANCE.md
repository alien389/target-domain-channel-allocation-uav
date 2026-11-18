# Weight license, source and notices

**Conveyance route:** distribute the 15 author-generated terminal-EMA derivatives
and associated graphs on the open-source AGPL-3.0 route, with the matched
corresponding-source asset and notices available from the same release. Do not
remove the license, upstream credit or modification notices. This offer concerns
this study's weight derivatives and covered source, not third-party datasets,
unrelated binaries or commercial terms. See the accompanying full
[AGPLv3](../../LICENSE) and historical GPLv3 in the corresponding-source archive texts.

Ultralytics YOLOv5 is credited to Ultralytics and its contributors. The
[v7.0 pinned source](https://github.com/ultralytics/yolov5/tree/915bbf294bb74c859f0b41f1c23bc395014ea679)
has a GPLv3 notice; the [recorded later execution tree](https://github.com/ultralytics/yolov5/tree/88af13e3e25602344aab68d44f9d0b666dacf695)
has an AGPLv3 notice. [Ultralytics' public licensing terms](https://www.ultralytics.com/license)
specifically identify an AGPL route for trained YOLO models. That model policy
supports this open-source route; it is not inferred from a source-code license
alone and is not represented as a historical contract or retroactive change to
the pinned GPL source. Apply the original notices to the covered code as
appropriate and preserve the terms of each covered component, including the
GPLv3/AGPLv3 section 13 combination provisions where applicable.

The archived stock v7.0 initializers match the official `yolov5s.pt`
(`8B3B748C1E592DDD8868022E8732FDE20025197328490623CC16C6F24D0782EE`)
and `yolov5m.pt`
(`61D933360BA5A7733A36764996C800287D973889D875227F5BEEDD2473A97A56`) downloads.
The Lite-v3 COCO launcher names the former; the recorded source-domain KD2
configuration names the latter as its teacher. Stock weights and intermediate
ancestors are referenced for provenance but are not redistributed here.

Twelve checkpoints descend from Lite-v3 COCO to the target-adapted
SearchParent, then use the locked retained channels; three SourceLocked-B6
checkpoints descend from the source-domain Stage-6 COCO post-KD2 branch.
All 15 then follow the same 300-epoch VisDrone training lock without a teacher.
`checkpoint-index.csv` records the upstream and original state hashes
independently of the final published bytes.

The source asset contains pinned YOLOv5 source, frozen local model/runtime
changes, model graphs/configuration, final-horizon training source, structural
locking source, source-domain KD implementation and locked numerical config,
SearchParent training support, and the tensor-only reader. The source manifest
distinguishes exact source from path substitutions. The Stage-6 trainer is the
retained implementation matching the archived config/output convention and
runtime pin; no new invocation or historic trainer-file hash is claimed.
The checkpoint transformation removes path/training metadata and reserializes
tensors without numerical alteration. Redistribution of upstream stock
or provider data is not required for the portable loader.

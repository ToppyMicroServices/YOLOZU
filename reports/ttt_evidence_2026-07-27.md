# TTT evidence run — 2026-07-27

Status: **local diagnostic; efficacy not established; promotion ineligible**.

## Confirmed execution

- Dataset: COCO128 `train2017`, first 8 images in manifest order
- Shift: deterministic Gaussian blur, severity 3, seed 2026
- Methods: Tent, MIM, CoTTA, EATA, SAR
- Seeds: 11, 22, 33
- Input: 128 × 128, CPU
- Evaluation: real `pycocotools` COCO AP50:95; no proxy AP was used
- Protocols: Tent/MIM/EATA/SAR use sample reset; CoTTA uses continual stream
- Matrix: 30/30 child comparisons completed

The suite selected strict `content` dataset hashing and generated
checkpoint/config hashes, dataset content and order hashes,
COCO AP, clean retention, shifted-domain AP, calibration status, detector-output
collapse status, update ratio, subprocess latency, process peak RSS, and actual
forward/backward/optimizer counts.

## Checkpoints

Both checkpoints were trained from the current source on real COCO128 images for
two epochs with 30 optimizer steps per epoch and passed strict full-compatibility
preflight.

| Configuration | SHA-256 |
|---|---|
| Base checkpoint | `291d10f63908bd4b4746e910fb17ed677be16ff824fba77fe18b27285cffcdba` |
| Base resolved config | `17db6612011b5c8cd2cc456a40e1d8e3495286290f994531c900c7666adc91a0` |
| Base run metadata | `75a3d23d6c991fe196e1c300a6b770ca431580eb0deb01d783ac393cbd135ca5` |
| MIM checkpoint | `7014cd470624cb256668ea2a90690848fef778e44e720e7ca819eba757dccf31` |
| MIM resolved config | `590954532c3651ea669dfae2a3b73fbd248b30c427fe9253e9abe4700bdd83c5` |
| MIM run metadata | `23dcba0a24a867b67587829b97956a3fdde285c5023139862b06c43186f4000f` |
| Shift recipe | `b25503b760dbf3969183132b290839034a2730942e738693c1b45dce0bf92e88` |
| Full suite summary | `7c68280a2a9e52b46597e8b7187b6833d97b09de7e8bc2bb1a74c2385c0437aa` |

## Result

| Method | Protocol | Clean retention Δ mAP50:95 | Shifted Δ mAP50:95 | Worst shifted mAP50:95 |
|---|---|---:|---:|---:|
| Tent | sample reset | 0.000000 | 0.000000 | 0.000000 |
| MIM | sample reset | 0.000000 | 0.000000 | 0.000000 |
| CoTTA | continual stream | 0.000000 | 0.000000 | 0.000000 |
| EATA | sample reset | 0.000000 | 0.000000 | 0.000000 |
| SAR | sample reset | 0.000000 | 0.000000 | 0.000000 |

The MIM checkpoint produced nonzero clean COCO AP50:95
(`0.0018151815181518146`) in its clean runs; the adapted result was unchanged.
The base checkpoint emitted no retained detections under the fixed export
settings, and the shifted target produced no improvement. The generated
calibration field therefore reports `unavailable_no_detections` where
applicable instead of presenting zero ECE as measured calibration.

## Boundary and remaining gate

This run proves that the current-compatible paths execute and that the required
metrics are generated. It does **not** prove efficacy: the diagnostic
checkpoints are short COCO128 training runs, every improvement delta is zero,
and all artifacts were produced in one local environment.

The full child artifacts and checkpoint provenance are published in the
[TTT diagnostic evidence 2026-07-27 prerelease](https://github.com/ToppyMicroServices/YOLOZU/releases/tag/ttt-evidence-2026-07-27).
The archive SHA-256 is
`bb200d0c0a36447f0b6ed262a56ee09bef44ded8f10c55673243080fe1054068`;
its internal manifest verifies 261 files.

Release addressability is satisfied. Independent reproduction in a second
environment remains required before the Beads efficacy task can close.

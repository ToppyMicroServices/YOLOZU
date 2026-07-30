# Official BOP19 T-LESS Object 6DoF Evidence — 2026-07-30

Status: **official target-conditioned evaluation and semantic replay complete;
small non-zero official/task-native scores are not seed-robust; Research;
efficacy not established**.

## Dataset and evaluator provenance

The official T-LESS inputs were downloaded from the BOP benchmark dataset
distribution and verified before extraction:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `tless_base.zip` | 49,597 | `dd70ca884b7c471a530a952f70c5ab2c212f3d2c2f371be86397442b97d70a7e` |
| `tless_models.zip` | 33,468,762 | `6a29c59766b8d2af05e62e71739f0ca7243ad81bc3c7f9a24925504a8cb37928` |
| `tless_test_primesense_all.zip` | 825,276,992 | `1a18f6bbfb5ac4ced8529f7a35225adfed88c0f62ef38067933e2b541ef1d00b` |

`test_targets_bop19.json` contains 4,904 target rows and 6,423 target
instances. T-LESS data is CC BY 4.0. The pinned BOP toolkit commit is
`cea62d651c7e395b2e1962b9749e4e89693c6ac4`.

## Training and inference boundary

Three RT-DETR pose checkpoints were trained with seeds 11, 22, and 33 from
strict real `train_primesense` GT. The bounded training selection contains 378
images/instances across all 30 object classes and 1,272 visible projected CAD
keypoints. No official test GT was used by training or inference.

| Seed | Checkpoint SHA-256 | Export wall time | Peak RSS |
|---:|---|---:|---:|
| 11 | `897ec4cd71cb2c3e5131c523aa508ee59dbe01174835c467e3509672b8d73b84` | 15.45 s | 290,013,184 B |
| 22 | `d7f8488ac47a0a1c6ede4c7395877c59d1e69fa057eb8f7e681f35c1f69ddf1d` | 17.67 s | 277,889,024 B |
| 33 | `7a1cd6b117abf77fe39f8081e4358600cd55950a85b2bafe5027e83ae818b404` | 29.27 s | 291,684,352 B |

`tools/export_bop19_rtdetr_pose.py` emitted one target-conditioned estimate for
every official target instance. The full CSVs and per-export reports record
config, checkpoint, target, result, runtime, resource, source, and license
hashes.

## Evaluation

The official evaluator computes VSD, MSSD, and MSPD. A separate invocation of
the same pinned toolkit computes ADD, ADI, and symmetry-aware AD errors.
`tools/summarize_bop19_pose_evidence.py` combines these with matched rotation,
translation, ADD, ADD-S, and 0.1-diameter pose-success fields without replacing
unmeasurable values with zero.

## Observed result

| Seed | BOP19 AR | AR VSD | AR MSSD | AR MSPD | Matched / 6,423 | Rot. mean | Trans. mean | ADD mean | ADD-S / AD mean | Success @ 0.1d |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.00100161 | 0.00060719 | 0.00035809 | 0.00203955 | 2,402 | 120.23 deg | 76.24 mm | 90.06 mm | 43.27 mm | 0.00856298 |
| 22 | 0.00188282 | 0.00089989 | 0.00054492 | 0.00420364 | 2,730 | 116.82 deg | 75.78 mm | 91.59 mm | 44.14 mm | 0.01183248 |
| 33 | 0.00190980 | 0.00000000 | 0.00000000 | 0.00572941 | 23 | 115.33 deg | 154.30 mm | 166.53 mm | 97.29 mm | 0.00000000 |

The non-zero official recall and pose-success values are measured results, but
they are small and not seed-robust. Seed 33 has non-zero projection recall and
zero 0.1-diameter pose success. The large mean rotation error is incompatible
with a positive 6DoF claim.

## Independent semantic replay

The independent role re-read the immutable official score files, result CSVs,
export reports, native error files, and GT metadata into a fresh summary. It
matched all per-seed official and task-native values within `1e-9`.

- primary summary SHA-256:
  `3df0f8ca3371601de6bba97f0d04f51a29657b50d4aa9bf9401b87a6670ebc4c`;
- independent summary SHA-256:
  `4d94bc2846bb5c0c7b972bda31bfbc13d5e2ec32da23d65f702c0c17b6f3e49a`;
- semantic comparison: `same_seed_metrics_within_1e-9=true`.

The tracked machine-readable summaries are
[`primary`](bop19_tless_pose_primary_2026-07-30.json) and
[`independent`](bop19_tless_pose_independent_2026-07-30.json).

## Decision boundary

Completing the official protocol closes the earlier frame-holdout protocol gap.
The observed result remains `hold` and `not_established`: positive values are
too small and inconsistent across seeds to support a pose-quality claim. It
does not establish human 3D skeleton support. Promotion requires materially
positive, seed-robust official and task-native pose quality.

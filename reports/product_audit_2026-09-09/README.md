# YOLOZU product usability and performance audit

Date: 2026-09-09. Baseline: `b5f5601` (public package `4.7.0`).
Tracked by `YOLOZU-ll2.83`.

This audit follows the Stable install, proof, validation, export, and evaluation
path. It also checks the README, installation docs, TeX manual, and generated
references. It does not qualify every model/runtime or establish why particular
people did not adopt YOLOZU.

## Reproduced problems and changes

| Problem | Evidence before the change | Change |
|---|---|---|
| Invalid numeric dataset values can pass validation | NaN box coordinates pass strict validation; nonfinite image dimensions are not rejected consistently | Reject nonfinite numbers and malformed falsey label containers; retain warn-mode behavior |
| Repeated work in the proxy mAP evaluator | Ground-truth lookup, score sorting, and IoU work repeat across thresholds | Prepare matches once per class, with independent matching occupancy per threshold |
| Unnecessary validator memory | The iterable API immediately copies every record into a list | Validate each incoming record incrementally; retain empty-input rejection |
| Installed guide examples need checkout assets | Evaluation/export/debug routes refer to `data/smoke` or predictions they never create | Generate toy inputs with `doctor --proof` and execute the routes from a new directory |
| Doctor proof PNG is corrupt | Pillow can read its dimensions but full decode and IDAT checksum checks fail; export silently skips the overlay | Correct the embedded PNG and test verification, full decode, exported PNG, and HTML reference |
| First-use docs obscure the working path | Evaluation precedes installation, required inputs are unclear, and long Experimental details precede the useful next steps | Install-first instructions in three languages, input requirements, real COCO evaluation, and collapsible advanced details |
| Installation docs contain broken or obsolete guidance | An unclosed code fence turns later prose into code; the official Conda-channel route cannot provide the required Torch version | Close the fence; use dependency-resolving venv/pip setup; clarify MPS device visibility versus workload qualification |
| Manual content and checks drift | Wrong literal chapter numbers, nonexistent `yolozu eval-suite`, old pip-to-repo instructions, and overstated license scope | Use chapter references, installed `eval-coco`, an early working first-run path, and repository-code license wording |
| Manual audit misses invalid examples | Only chapter 04 command macros were checked, leaving shell listings unchecked | Audit all 21 chapter files and listings; regress unknown commands in temporary TeX fixtures |
| PDF command text is not safely copyable | Shell quotes extract as U+2019; inline double hyphens become en dashes; long tokens overflow | Literal breakable command rendering and straight listing quotes, followed by final PDF QA |

## Measurements

The retained [runtime benchmark](runtime-benchmark.json) compares baseline and
patched code in interleaved runs on macOS arm64 with Python 3.14.6.

- Proxy mAP, 2,000 images, 4,000 detections, 20 classes, 10 thresholds:
  median **1.735 seconds to 0.206 seconds**, an **8.44x** speedup.
  All 400 randomized metric comparisons were exactly equal.
- Dataset-validator API, 100,000 generated empty-label records with image I/O
  disabled: peak Python allocations measured by `tracemalloc` fell from
  **29,773,146 bytes to 1,588 bytes**. Error-list growth and caller allocations
  are outside this clean-input result. CLI manifest construction remains eager.

These measurements concern CPU evaluation/validation overhead. They do not show
faster model inference, better detection accuracy, or a speedup in official
COCOeval. Run [runtime_benchmark.py](runtime_benchmark.py) to reproduce the
synthetic comparison; its default baseline is pinned to the audited revision.

## Installed-package verification

The public `yolozu==4.7.0` wheel was installed from PyPI in a clean environment.
The repository's fresh-install harness passed all five Stable smoke steps.
Recorded step time totaled 11.859 seconds: environment creation 1.993 seconds,
installation 4.512 seconds, and the installed-package smoke path 5.354 seconds.
This is one run on one host and network, not a time guarantee. The initial
sandbox-denied network attempt is retained separately and is not a package defect.

The public wheel SHA-256 is
`5ff9073e7d88a1bb4bf5c9510c2812ba16b338d15071234a68fc086be16b7f3a`.
Its standard proof harness passed despite the corrupted PNG because it did not
fully decode the proof image. This audit adds that missing check.

The patched candidate wheel was separately installed with core and COCO
dependencies, without an editable install or `PYTHONPATH`. The
[installed journey](installed-journey-final/installed-journey.json) ran all 11
evaluate/export/debug commands, then official COCOeval and the explicit synthetic
demo. All 13 commands passed. It verified PNG checksums, full decoding, and the
promised export outputs. Installed source hashes are retained in that report.
The toy COCO result is approximately 1.0 by construction, not model-quality evidence.

The candidate still uses source version 4.7.0 for local testing; it is not the
published 4.7.0 wheel. Its SHA-256 is
`eab28f05a3f20363e2f1ca223a04171620a5781bc87684e70e0d9c36f3bf14ff`.
A new release must have its own version and public verification.

## Documentation and regression checks

The documented-command audit passed 123 public shell examples against the
available help/flag surface. The expanded manual audit passed 26 documented
command families across all 21 chapters. These are command-surface checks, not
execution of every model-dependent example. The manifest help audit checked
131 Python tools with no execution errors or missing declared flags.

The source and packaged manifests are byte-identical. Required manifest checks,
targeted regression tests, and repository-wide Ruff checks passed. The final
full suite ran **1,701 tests with zero failures, zero errors, and 18 skips**
(766.072 seconds of test execution). Skips cover unavailable CUDA and optional
dependencies, plus the no-Torch path in a Torch-equipped environment.
See [tests.json](tests.json) and the local `unittest.log` for exact counts and
skip reasons. The first
full run exposed two tests that hardcoded the old chapter numbers, a stale
example count, and an in-progress generated-reference mismatch. Those were
corrected before the final full run. A subsequent run exposed a bug in this
audit's test-recording harness: without a main guard, a spawned PDF worker
re-entered test discovery and exceeded its time limit. The harness was corrected;
the product PDF parser did not require a change. The failed harness log and JSON
are retained locally with `.harness-before-fix` suffixes.

The first PR CodeQL run found a side effect inside an `assert` in the installed
journey helper. Explicit checks now preserve both command handling and failure
detection under Python `-O`. The helper passed all 13 commands in normal mode
and in [optimized mode](installed-journey-ci-optimized/installed-journey.json);
an [invalid guide command](installed-journey-ci-rejected/installed-journey.json)
was rejected under `-O`. This correction only changes audit tooling, not the
package bytes covered by the full regression run.

## Manual artifact

The TeX source remains `manual/main.tex` plus `manual/chapters/*.tex`.
Before this audit, `docs/yolozu_manual.pdf` and `reports/yolozu_manual.pdf` were
identical 163-page June artifacts. Their SHA-256 was
`c5c6ff73635ba028f41e7b31fc296aa7b2cc64374b3272f3514cbceea2d91e0c`.
The ignored August local build was a different 172-page document.

The final PDF has 174 pages and SHA-256
`f3f5e05b9b85c5fd0d0f3f75f9d686d260ad94dda2b9355cb80868e0ffb54637`.
The build, tracked docs PDF, and ignored reports convenience copy are identical.
There are no overfull boxes or unresolved chapter references. LaTeX retains a
font-shape substitution warning, which is not a PDF structure error.

Final PDF structure, renderer, and visual-review evidence is recorded in
[qa-report.json](manual-qa/qa-report.json). qpdf 12.4.1 and Ghostscript 10.07.1
returned zero with no structural diagnostics. All 174 pages rendered with
Poppler 26.07.0, PDFium 153.0.7999.0, and macOS PDFKit on macOS 26.6.2.
Review covered all pages in contact sheets from each renderer and selected
full-size pages. No blocking visual defects were found; the QA report records
minor line-wrapping and pagination observations. Full-page PNGs and contact sheets from Poppler,
PDFium, and macOS PDFKit are retained locally under `manual-qa/`.
`verify_manual_pdf.py` and `render_pdfkit.swift` reproduce the mechanical checks.
Successful structure/rendering checks and visual review do not mean user approval
or that a new manual DOI version has been published.

## Adoption: evidence and next decisions

The latest committed [2026-09-03 snapshot](../../docs/adoption/2026-09-03-baseline.md)
records two GitHub views in 14 days and no qualifying external issues or
discussions. Confirmed first installation, first evaluation, and repeat use are
all unknown. Clone counts are automation-sensitive and cannot estimate people.
The snapshot is dated evidence, not a refreshed 2026-09-09 audience measurement.

The demonstrated defects create plausible friction, but no user study connects
them to abandonment. The small observed discovery signal also means that better
code alone may not increase use. The next useful product observation is a
consenting newcomer completing an evaluation on their own prediction artifact:
record time, the point where help was needed, and whether the report answered
their question. Use the existing
[observation kit](../../docs/adoption/design_partner_observation_kit.md); do not
collect raw private data or add telemetry merely to fill the unknowns.

Prioritize additional adapter or performance work when a reproducible user
workload identifies the missing capability or budget. GPU/MPS throughput,
real-model accuracy, Windows execution, customer workloads, and retention were
not qualified by this audit. Existing Experimental/Research boundaries remain
appropriate; this work does not promote those capabilities.

## Public surfaces inspected

The [official product page](https://www.toppymicros.com/yolozu/) and its
[web docs](https://www.toppymicros.com/yolozu/docs/) were checked; the docs URL
returned HTTP 200. PyPI's live JSON API reported 4.7.0. A web-search cache still
showed 4.6.0, so it was not used as current release evidence.
The [PyTorch 2.6 release note](https://pytorch.org/blog/pytorch2-6/) confirms the
end of official Conda-channel releases; the revised MPS setup links to that
source and the official MPS guidance.

This branch changes repository artifacts. A new PyPI release, DOI publication,
and a separately hosted product-page update are distinct publication steps.
Release verification is tracked by `YOLOZU-ll2.85`; consenting first-evaluation
observations are tracked by `YOLOZU-ll2.84`. The existing August feedback review
remains under `YOLOZU-ll2.82` and was not marked complete by this audit.

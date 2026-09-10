# YOLOZU LaTeX Manual

This folder contains a **direct TeX** (hand-written LaTeX) manual for YOLOZU.

## Source of truth

Edit `main.tex` and `chapters/*.tex`. The include order in `main.tex` determines
the printed chapter numbers; filename prefixes do not. Use chapter labels and
`\ref{...}` for cross-references.

`build/yolozu_manual.pdf` is an ignored local build. The tracked distribution
copy is `../docs/yolozu_manual.pdf`; `../reports/yolozu_manual.pdf` is an ignored
local convenience copy. A successful build does not update either copy automatically.

## First run

The installed package works without a source checkout. On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install yolozu
yolozu doctor --proof --output reports/first_run/doctor.json --proof-dir reports/first_run/proof
yolozu demo instance-seg --background synthetic --inference none --run-dir reports/first_run/demo
```

The proof and offline demo check installation and artifact processing with known
or synthetic predictions. They do not establish trained-model accuracy.

For the repository smoke path, run this from the repository root:

```bash
bash scripts/smoke.sh
```

This validates the committed offline assets under `data/smoke` and writes
`reports/smoke_coco_eval_dry_run.json`.

## Build

Requirements:
- A LaTeX distribution (MacTeX / TeX Live)
- `latexmk` (usually included with TeX Live)

Build PDF:

```bash
cd manual
make pdf
```

Clean:

```bash
cd manual
make clean
```

Output:
- `build/yolozu_manual.pdf`

## Editing

- Entry point: `main.tex`
- Chapters: `chapters/*.tex`

This manual is designed to mirror the repo docs organization (see `docs/README.md`) while being
printable/searchable as a single PDF.

## Verify and distribute

From the repository root, check the source before building:

```bash
python3 tools/audit_manual_cli_drift.py --json
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
python3 -m unittest tests.test_packaged_tools_manifest tests.test_manifest_docs_references tests.test_manual_cli_drift_audit
```

When distributing an update, validate the final PDF that will be copied to both
distribution paths. Require clean Ghostscript and `qpdf --check` results and
inspect every page with Poppler, macOS PDFKit/Quick Look, and Chromium PDFium.
Keep the renderer versions, page images, SHA-256, and `qa-report.json` with the
release evidence. An unavailable check remains unverified; a LaTeX build alone
is not PDF QA.

After QA passes, update both distribution copies from that exact PDF and verify
that all three SHA-256 values match. Do not label an older distribution PDF as
current merely because a newer local build exists. Zenodo publication is a
separate step described in `../docs/manual_doi_release.md`.

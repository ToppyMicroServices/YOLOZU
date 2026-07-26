# Web Docs Plan

YOLOZU's web docs should be the searchable learning surface. The PDF manual can
remain the complete reference, while web docs guide users through shorter paths.

Published surface: <https://www.toppymicros.com/yolozu/docs/>

## Information Architecture

| Section | Purpose | Source of truth |
|---|---|---|
| Start here | Isolated install, `doctor --proof`, strict validation, real eval | `README.md`, `docs/install.md`, `docs/cpu_only_dod.md`, `docs/python_api.md` |
| Tutorials | 30-minute and 2-hour guided flows | `docs/README.md`, `docs/evaluation_protocol_template.md` |
| Command reference | Generated CLI/tool reference | `tools/manifest.json`, `docs/tools_index.md` |
| Schema browser | Predictions, dataset/eval, training, SynthGen, research schemas | `docs/schema_governance.md`, `docs/schemas/` |
| Examples gallery | Report and overlay examples with expected files | `docs/assets/`, `reports/` examples |
| Glossary | Stable terms such as predictions interface contract, protocol hash, lane | `docs/predictions_schema.md`, `docs/production_readiness.md` |
| What can go wrong | Common failures and how to verify | `manual/chapters/11_troubleshooting.tex`, `docs/doctor_diagnostics.md` |
| Report reading guides | How to compare outputs and skipped lanes | `docs/benchmark_support_matrix.md`, `docs/evaluation_protocol_template.md` |

## Tutorial Path

### 30-minute path

1. Install `yolozu[coco]` in an isolated environment. A candidate wheel may be
   substituted for the PyPI package, with `pycocotools` installed alongside it.
2. Run `yolozu doctor --proof` to create the toy dataset and predictions used
   by every later command.
3. Strictly validate the generated dataset and predictions.
4. Run real COCOeval through concise `eval-coco` flags. Use `--dry-run` only as
   an explicit dependency-free validation/conversion fallback.
5. Use the same artifacts through the stable `yolozu.api` surface when
   embedding the evaluation in Python.
6. Read the report using the protocol and skip-reason checklist.

### 2-hour path

1. Complete the 30-minute path.
2. Export or import predictions from an external stack.
3. Run a CPU-only evaluation with a fixed protocol.
4. Compare two prediction artifacts.
5. Fill out `docs/evaluation_protocol_template.md` for the run.
6. Decide whether the next step is stable evaluation, backend parity, or a
   research lane.

## Generated Pages

| Page | Generator / input | Gate |
|---|---|---|
| Command reference | `tools/manifest.json` plus `--help` output | manifest/docs reference tests |
| Schema browser | `docs/schema_governance.md` plus `docs/schemas/*.json` | schema governance tests |
| Examples gallery | checked-in demo reports and assets | golden/report validation |
| Glossary | curated Markdown source | link checker / docs drift test |

## Acceptance Checklist

- Web docs must lead with "Evaluate existing predictions".
- Stable, bridge, benchmark, and research lanes must remain visually separate.
- Research pages must link back to the stable evaluation artifact they extend.
- Generated pages must fail CI when their source manifest/schema entries drift.
- Every configured source must resolve to a regular file inside the repository;
  absolute, parent-traversing, and symlink-escaping paths are rejected.
- Generated links must be internal links or credential-free HTTPS URLs.
- A non-empty output directory may be replaced only when its complete contents
  match generator-owned `provenance.json`; replacement is staged and renamed
  atomically.
- The CI candidate-wheel journey must execute real COCOeval. Missing
  `pycocotools` is a failure in that gate.
- No web page should be the only source for CLI flags or schema fields.

## Implemented Surface

`tools/generate_web_docs.py` builds the checked-in static bundle at
`docs/generated/web_docs/`.

The bundle includes:

1. a searchable hub that leads with "Evaluate existing predictions";
2. the self-contained 30-minute and two-hour tutorial paths;
3. a command reference generated from every `tools/manifest.json` entry;
4. a schema browser generated from every `docs/schemas/*.json` file;
5. a repository-backed example gallery and report-reading checklist;
6. a curated glossary;
7. an evidence-first failure guide; and
8. a stable typed-Python example sourced from `docs/python_api.md`; and
9. a search index spanning commands, schemas, lanes, examples, terms, and
   troubleshooting entries.

Stable, Bridge, Benchmark, and Research cards use separate visual states.
Research examples link back to a Stable artifact or the Stable tutorial.

The entry and completion links emit aggregate Plausible events containing only
the fixed page and target labels. Search text is not sent to analytics.

## Generation And Publication

Regenerate after changing the manifest, schemas, curated content, or source
assets:

```bash
python3 tools/generate_web_docs.py
```

CI uses a non-writing drift gate:

```bash
python3 tools/generate_web_docs.py --check --json
python3 -m unittest \
  tests.test_web_docs_generation \
  tests.test_web_docs_candidate_wheel
```

The candidate-wheel journey builds the checked-out commit, installs that wheel
in a nested environment, removes `PYTHONPATH`, verifies that `yolozu` imports
from the installed wheel, invokes the installed `yolozu` console script, and
executes the commands and Python example held in `docs/web_docs_content.json`
from a directory outside the checkout. CI installs
`requirements-locks/requirements-web-docs.lock` and sets
`YOLOZU_REQUIRE_REAL_COCO=1`, so the documented real COCOeval path cannot
silently fall back to a dry run. The nested environment deliberately reuses
those hash-locked runner dependencies so the gate remains offline; this
isolates the YOLOZU package origin, not dependency resolution. Public
fresh-install evidence is handled separately by
`scripts/fresh_install_journey.sh`.

`provenance.json` records SHA-256 hashes for the generator and every SSOT file
referenced by the rendered pages, including all manifested implementation/docs
links, schemas, curated content, stable artifacts, and copied images. The
generator accepts source inputs only from inside the repository and refuses to
delete a non-empty directory whose complete inventory is not generator-owned.
The generated bundle is published unchanged under `/yolozu/docs/` on the
ToppyMicroServices site. Repository Markdown, JSON Schemas, manifests, and
report artifacts remain the source of truth; the web pages link back to them
and do not define new flags or fields.

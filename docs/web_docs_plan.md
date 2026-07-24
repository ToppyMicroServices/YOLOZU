# Web Docs Plan

YOLOZU's web docs should be the searchable learning surface. The PDF manual can
remain the complete reference, while web docs guide users through shorter paths.

Published surface: <https://www.toppymicros.com/yolozu/docs/>

## Information Architecture

| Section | Purpose | Source of truth |
|---|---|---|
| Start here | Install, `doctor --proof`, demo, validate, eval | `README.md`, `docs/install.md`, `docs/cpu_only_dod.md` |
| Tutorials | 30-minute and 2-hour guided flows | `docs/README.md`, `docs/evaluation_protocol_template.md` |
| Command reference | Generated CLI/tool reference | `tools/manifest.json`, `docs/tools_index.md` |
| Schema browser | Predictions, dataset/eval, training, SynthGen, research schemas | `docs/schema_governance.md`, `docs/schemas/` |
| Examples gallery | Report and overlay examples with expected files | `docs/assets/`, `reports/` examples |
| Glossary | Stable terms such as predictions interface contract, protocol hash, lane | `docs/predictions_schema.md`, `docs/production_readiness.md` |
| What can go wrong | Common failures and how to verify | `manual/chapters/11_troubleshooting.tex`, `docs/doctor_diagnostics.md` |
| Report reading guides | How to compare outputs and skipped lanes | `docs/benchmark_support_matrix.md`, `docs/evaluation_protocol_template.md` |

## Tutorial Path

### 30-minute path

1. Install and run `yolozu doctor --proof`.
2. Run `yolozu demo instance-seg --progress`.
3. Validate the demo predictions and dataset.
4. Open the generated report and overlays.
5. Read the report using the protocol and skip-reason checklist.

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
- No web page should be the only source for CLI flags or schema fields.

## Implemented Surface

`tools/generate_web_docs.py` builds the checked-in static bundle at
`docs/generated/web_docs/`.

The bundle includes:

1. a searchable hub that leads with "Evaluate existing predictions";
2. the 30-minute and two-hour tutorial paths;
3. a command reference generated from every `tools/manifest.json` entry;
4. a schema browser generated from every `docs/schemas/*.json` file;
5. a repository-backed example gallery and report-reading checklist;
6. a curated glossary;
7. an evidence-first failure guide; and
8. a search index spanning commands, schemas, lanes, examples, terms, and
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
python3 -m unittest tests.test_web_docs_generation
```

`provenance.json` records SHA-256 hashes for every manifest, schema, curated
content file, and copied example image used by the bundle. The generated bundle
is published unchanged under `/yolozu/docs/` on the ToppyMicroServices site.
Repository Markdown, JSON Schemas, manifests, and report artifacts remain the
source of truth; the web pages link back to them and do not define new flags or
fields.

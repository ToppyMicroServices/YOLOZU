# Experimental algorithm scout

`yolozu scout-algorithms` is a maintainer-only monitored-source inbox. It does
not search the whole web or establish which algorithm is latest or best. The
only accepted source file is `docs/algorithm_intake/sources.json`.

The safe default validates the allowlist and prints a bounded JSON plan. It does
not open a network connection or create the output directory.

```bash
yolozu scout-algorithms \
  --sources docs/algorithm_intake/sources.json \
  --output-dir reports/algorithm_scout \
  --collection-date 2026-08-26 \
  --trigger workflow_dispatch
```

`--collect` enables the only network and write path. Collection accepts exact
allowlisted HTTPS scheme/host/path records on port 443. It rejects credentials,
queries, fragments, IP literals, non-public DNS or peer addresses, and redirects
outside the same explicit allowlist. No caller headers, cookies, or tokens are
accepted. TLS 1.2 is the minimum protocol version; normal certificate-chain,
hostname, and SNI validation remain enabled. A source gets 30 seconds, the run
gets 12 minutes, and the surrounding
15-minute workflow must keep three minutes for report finalization and failure
handling.

Fetched HTML, text, JSON, and PDF are untrusted input. The parser applies the
document, process, memory, time, and byte caps recorded in the Bead and report.
It never follows URLs found in fetched content, executes code, downloads weights,
installs packages, or copies raw pages into the repository. The dated managed
output retains bounded summaries, content provenance, explicit failed or missed
status, and `unknown` for unavailable metadata. A prior candidate absent from the
current response is retained only as `historical`, never relabeled as freshly
collected.

Exit codes are:

- `0`: a valid report with no enabled-source failure;
- `3`: a valid report with at least one failed or missed source;
- `2`: invalid or unsafe input rejected before collection; and
- `1`: bounded internal or finalization failure.

Retention follows repository history policy for the dated metadata reports. Raw
source documents, headers, response bodies, local paths, credentials, and parser
temporary files are not retained. Repeated runs merge candidate history and
deduplicate by source URL plus version or revision.

The repository workflow runs every Monday at 00:00 UTC (09:00 Asia/Tokyo),
with one non-cancelling concurrency group and a 15-minute job limit. The
collector itself stops after 12 minutes so validation, issue handling, and
finalization keep a three-minute reserve. A manual `workflow_dispatch` uses the
same bounded path but is not a substitute for observing the scheduled event.

For exit `0` or `3`, the workflow validates the managed report and uploads only
`docs/algorithm_intake/YYYY-MM-DD.json`, its privacy-safe aggregate Markdown
view, and the checksum manifest in `algorithm-scout-YYYY-MM-DD`. GitHub retains
that artifact for 30 days. Exit `1` or `2` uploads no report, raw body, cache, or
temporary file. Raw fetched bodies and parser cache exist only in the fresh
runner temporary directory and expire with the job.

Before a scheduled collection, the workflow reads back the previous expected
artifact using `actions: read`. If it is absent, the current report records the
missed date as unknown through `--missed-collection-date`; it does not fetch the
old date or call current data historical. Failures create or update the single
exact-title issue `Algorithm scout scheduled run failed` with a bounded generic
summary. A later successful scheduled run may close that exact issue after
readback. The workflow never commits a report, changes the registry, opens a
model PR, or promotes a bundle.

The output kind is `yolozu_algorithm_scout_report` with
`selectability=inbox_only`. It is deliberately different from the AlgorithmBundle
registry interface contract. The implemented non-executing screening stage can
turn one immutable candidate into pass, hold, or reject, but only through the
separate `CandidateScreeningRecord` interface contract. The packaged append-only
stream currently has two reviewed `hold` records and no pass. A current
repository-managed pass and a later registry action are both required before a
candidate can enter lifecycle, qualification, selection, or execution paths.

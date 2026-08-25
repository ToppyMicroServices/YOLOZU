# Adaptive algorithm scout foundation

Recorded: 2026-08-26 (Asia/Tokyo)

Bead: `YOLOZU-ll2.81.3.1`

## Result

The repository now has an Experimental read-only algorithm scout. Its default
mode validates the canonical official-source allowlist and prints a JSON plan
without network access or writes. `--collect` is the only path that can fetch or
publish a dated managed report.

The canonical allowlist monitors four public release endpoints: Detectron2,
MMDetection, Ultralytics, and YOLOX. The endpoints were read back through the
official GitHub API on 2026-08-26. This is a small monitored set, not evidence of
exhaustive coverage of current vision research.

A live smoke collection through `SafeHttpsTransport` ran from
2026-08-25T17:45:10Z to 2026-08-25T17:45:12Z (2026-08-26 02:45 JST). All four
sources completed, producing 74 deduplicated release records from 595,016 decoded
bytes. The managed report and checksum were written only to a temporary workspace
and were not committed. This proves endpoint and transport interoperability at
that time; it does not qualify any release or make it selectable.

## Trust and network boundary

`SafeHttpsTransport` accepts structured scheme/host/path records only. It rejects
credentials, queries, fragments, IP literals, non-443 destinations, non-public
IPv4/IPv6 DNS results, changed peer addresses, and redirects not separately
allowlisted. The socket connects to a vetted resolved address while TLS chain,
hostname, and SNI verification continue to use the original host. Callers cannot
provide headers, cookies, or tokens.

The implementation caps connect, read, per-source, and whole-collection time. It
also caps headers, transferred bytes, decoded bytes per source, and decoded bytes
for the run. Deadline exhaustion stops new fetches, records the remaining sources
as missed, finalizes a valid report, and returns exit 3.

## Untrusted parser boundary

Only HTML, UTF-8 text, JSON, and PDF are accepted. Archive/media magic,
content-type confusion, active or embedded HTML, DTD/entity declarations, and PDF
embedded files fail closed. HTML and JSON are bounded before an unbounded document
tree is built. PDF inspection runs in a fresh secret-free POSIX child process
group with network disabled, wall/CPU/address-space/PID/file/IPC limits, bounded
handoff, process-tree termination, reaping, and exact private temporary-directory
cleanup.

The report stores source identity, version/revision or `unknown`, release date,
task scope, local/hosted availability, license status, weight status, runtime
hints, collection status/timezone, bounded parser facts, and content SHA-256. It
does not retain raw pages, release bodies, papers, archives, weights, or embedded
instructions.

## Selection boundary

The output kind is `yolozu_algorithm_scout_report` and has
`selectability=inbox_only`. It has no `bundles` field and is rejected by the
AlgorithmBundle registry validator. Discovery cannot register, qualify,
recommend, execute, or promote a candidate. Those state changes remain separate
reviewed Beads work.

## Verification scope

Offline fixtures cover plan/no-write behavior, CLI help and exit codes, URL and
address rejection, DNS/peer and redirect checks, header/body/decompression caps,
content confusion, HTML/JSON/PDF bounds, explicit failure and deadline reports,
history deduplication with prior-only candidates marked `historical`, output symlink rejection before network access, managed
checksums, and the nonselection boundary. These fixtures verify the interface and
failure policy. They are not algorithm, runtime, performance, quality, support,
or adoption evidence.

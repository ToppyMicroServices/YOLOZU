#!/usr/bin/env python3
"""Detect redundancy across LaTeX manual chapters.

This is a lightweight static analysis tool:
- Finds repeated section/subsection headings across chapters
- Finds near-duplicate paragraphs via SimHash bucketing + difflib ratio

Usage:
  python3 tools/manual_redundancy_check.py
"""

from __future__ import annotations

import difflib
import glob
import hashlib
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CHAPTER_GLOB = "manual/chapters/*.tex"


COMMENT_RE = re.compile(r"(?m)^%.*$")
MATH_RE = re.compile(r"\$[^$]*\$")
# Intentionally conservative: strip commands without attempting to preserve all arguments.
CMD_RE = re.compile(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?(\{[^\}]*\})?")
HEADING_RE = re.compile(r"\\(sub)*section\*?\{([^}]*)\}")


def _normalize_tex(text: str) -> str:
    text = COMMENT_RE.sub("", text)
    text = text.replace("~", " ")
    text = MATH_RE.sub(" ", text)
    text = CMD_RE.sub(" ", text)
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\\+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9_]+", text) if t]


def _simhash(tokens: list[str], bits: int = 64) -> int:
    if not tokens:
        return 0
    weights = [0] * bits
    for token in tokens:
        h = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            bit = (h >> i) & 1
            weights[i] += 1 if bit else -1
    out = 0
    for i, w in enumerate(weights):
        if w >= 0:
            out |= 1 << i
    return out


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass(frozen=True)
class Paragraph:
    file: str
    start_line: int
    index: int
    raw: str
    norm: str
    simhash: int


def _iter_paragraphs(path: str) -> Iterable[Paragraph]:
    raw = Path(path).read_text(encoding="utf-8")

    # Split on blank lines, but keep start line numbers.
    # We do a manual scan to keep line positions stable.
    lines = raw.splitlines(keepends=True)

    buf: list[str] = []
    start_line = 1
    para_idx = 0

    def flush() -> Paragraph | None:
        nonlocal para_idx
        if not buf:
            return None
        chunk = "".join(buf).strip()
        if not chunk:
            return None
        norm = _normalize_tex(chunk)
        # Skip tiny chunks (headings, short callouts)
        if len(norm) < 350:
            return None
        tokens = _tokenize(norm)
        if len(tokens) < 80:
            return None
        para = Paragraph(
            file=path,
            start_line=start_line,
            index=para_idx,
            raw=chunk,
            norm=norm,
            simhash=_simhash(tokens),
        )
        para_idx += 1
        return para

    blank_re = re.compile(r"^\s*$")

    current_line = 1
    for line in lines:
        if blank_re.match(line):
            para = flush()
            if para:
                yield para
            buf = []
            start_line = current_line + 1
        else:
            buf.append(line)
        current_line += 1

    para = flush()
    if para:
        yield para


def _report_repeated_headings(files: list[str]) -> None:
    counts: Counter[str] = Counter()
    where: defaultdict[str, set[str]] = defaultdict(set)

    for path in files:
        text = Path(path).read_text(encoding="utf-8")
        for m in HEADING_RE.finditer(text):
            heading = m.group(2).strip()
            counts[heading] += 1
            where[heading].add(os.path.basename(path))

    repeated = [(h, c, sorted(where[h])) for h, c in counts.items() if c > 1]
    repeated.sort(key=lambda x: (-x[1], x[0]))

    print("\n== Repeated headings (across chapters) ==")
    if not repeated:
        print("(none)")
        return

    for heading, count, fileset in repeated[:80]:
        print(f"{count}\t{heading}\t{', '.join(fileset)}")


def _report_near_duplicate_paragraphs(files: list[str]) -> None:
    paragraphs: list[Paragraph] = []
    for path in files:
        paragraphs.extend(list(_iter_paragraphs(path)))

    # Bucket by high bits to reduce comparisons.
    buckets: defaultdict[int, list[int]] = defaultdict(list)
    for idx, p in enumerate(paragraphs):
        buckets[p.simhash >> 52].append(idx)  # 12-bit bucket

    candidates: list[tuple[float, int, int, int]] = []
    for bucket_ids in buckets.values():
        if len(bucket_ids) < 2:
            continue
        for i in range(len(bucket_ids)):
            for j in range(i + 1, len(bucket_ids)):
                a = paragraphs[bucket_ids[i]]
                b = paragraphs[bucket_ids[j]]
                if a.file == b.file:
                    continue
                ham = _hamming(a.simhash, b.simhash)
                if ham > 10:
                    continue
                ratio = difflib.SequenceMatcher(None, a.norm, b.norm).ratio()
                if ratio >= 0.80:
                    candidates.append((ratio, ham, bucket_ids[i], bucket_ids[j]))

    candidates.sort(reverse=True)

    print("\n== Near-duplicate paragraphs (ratio >= 0.80) ==")
    if not candidates:
        print("(none found at this threshold)")
        return

    for ratio, ham, ia, ib in candidates[:40]:
        a = paragraphs[ia]
        b = paragraphs[ib]
        print(
            f"{ratio:.2f} (ham={ham})\t"
            f"{os.path.basename(a.file)}:{a.start_line}\t<->\t"
            f"{os.path.basename(b.file)}:{b.start_line}"
        )


def main() -> None:
    files = sorted(glob.glob(CHAPTER_GLOB))
    if not files:
        raise SystemExit(f"No files matched: {CHAPTER_GLOB}")

    print(f"Scanned {len(files)} chapter files.")
    _report_repeated_headings(files)
    _report_near_duplicate_paragraphs(files)


if __name__ == "__main__":
    main()

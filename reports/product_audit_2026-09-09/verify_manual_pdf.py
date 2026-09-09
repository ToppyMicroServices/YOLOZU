"""Reproduce this audit's local final-PDF checks (requires macOS PDFKit).

Run with the bundled document Python, then INPUT.pdf and a new output directory.
This is audit evidence tooling, not a supported YOLOZU CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageStat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Exact final PDF to check")
    parser.add_argument("output", type=Path, help="New directory for logs, images, and qa-report.json")
    args = parser.parse_args()
    source = args.pdf.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo_root = Path(__file__).resolve().parents[2]
    source_label = str(source.relative_to(repo_root)) if source.is_relative_to(repo_root) else source.name
    report = {"source": source_label, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "platform": platform.platform(), "checks": {}, "renderers": {},
              "visual_review": "pending", "user_review": "not_performed", "ok": False}

    def run(name: str, args: list[str]) -> str:
        result = subprocess.run(args, capture_output=True, text=True, timeout=600)
        message = result.stdout + result.stderr
        (output / f"{name}.log").write_text(message)
        report["checks"][name] = {"exit_code": result.returncode, "log": f"{name}.log"}
        if result.returncode != 0:
            raise RuntimeError(f"{name} failed: {message[-2000:]}")
        return message

    try:
        report["structure_tools"] = {
            "qpdf": run("qpdf-version", ["qpdf", "--version"]).splitlines()[0],
            "ghostscript": run("ghostscript-version", ["gs", "--version"]).strip(),
        }
        for name, args in (
            ("qpdf-check", ["qpdf", "--check", str(source)]),
            ("ghostscript-check", ["gs", "-dBATCH", "-dNOPAUSE", "-dSAFER", "-dPDFSTOPONERROR", "-sDEVICE=nullpage", str(source)]),
        ):
            result = run(name, args)
            diagnostics = result.replace(
                "No syntax or stream encoding errors found; the file may still contain\nerrors that qpdf cannot detect", ""
            )
            if re.search(r"(?i)warning|repaired|incorrect xref|recursive dict|error", diagnostics):
                raise RuntimeError(f"{name} emitted a structural diagnostic")
        run("pdfinfo", ["pdfinfo", str(source)])
        poppler_version = run("poppler-version", ["pdftoppm", "-v"]).strip()
        (output / "poppler").mkdir()
        run("poppler-render", ["pdftoppm", "-r", "72", "-png", str(source), str(output / "poppler/page")])
        if (output / "poppler-render.log").read_text().strip():
            raise RuntimeError("Poppler emitted rendering diagnostics")
        report["renderers"]["poppler"] = poppler_version
        document = pdfium.PdfDocument(source)
        pages = len(document)
        report["pages"] = pages
        (output / "pdfium").mkdir()
        for i in range(pages):
            page = document[i]
            bitmap = page.render(scale=1)
            bitmap.to_pil().save(output / f"pdfium/page-{i + 1:03d}.png")
            bitmap.close()
            page.close()
        document.close()
        report["renderers"]["pdfium"] = str(pdfium.PDFIUM_INFO)
        swift_file = Path(__file__).with_name("render_pdfkit.swift")
        report["renderers"]["pdfkit"] = run("pdfkit-render", [
            "swift", "-module-cache-path", str(output / "swift-cache"),
            str(swift_file), str(source), str(output / "pdfkit"),
        ]).strip()
        page_checks = []
        for renderer in report["renderers"]:
            images = sorted((output / renderer).glob("page-*.png"))
            if len(images) != pages:
                raise RuntimeError(f"{renderer}: expected {pages} pages, found {len(images)}")
            for path in images:
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    stddev = max(ImageStat.Stat(rgb).stddev)
                    if min(rgb.size) < 500 or stddev < 0.5:
                        raise RuntimeError(f"{path}: blank or undersized render")
                    page_checks.append({"renderer": renderer, "page": path.name,
                                        "size": list(rgb.size), "stddev": round(stddev, 3)})
            # Contact sheets aid whole-document review; full-size pages remain retained.
            for start in range(0, pages, 20):
                sheet = Image.new("RGB", (1200, 1835), "#dddddd")
                draw = ImageDraw.Draw(sheet)
                for j, path in enumerate(images[start:start + 20]):
                    with Image.open(path) as img:
                        img = img.convert("RGB")
                        img.thumbnail((284, 335))
                        x, y = (j % 4) * 300 + 8, (j // 4) * 367 + 24
                        sheet.paste(img, (x, y))
                        draw.text((x, y - 18), f"{renderer}: {start + j + 1}", fill="black")
                sheet.save(output / f"{renderer}-contact-{start // 20 + 1:02d}.png")
        report["page_checks"] = page_checks
        report["ok"] = True
    except Exception as exc:
        report["error"] = str(exc)
    (output / "qa-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "page_checks"}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

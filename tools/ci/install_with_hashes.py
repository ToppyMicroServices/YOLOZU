#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _wheel_metadata(path: Path) -> tuple[str, str]:
    name, version, *_ = parse_wheel_filename(path.name)
    return canonicalize_name(name), str(version)


def _write_hash_locked_requirements(wheels: Iterable[Path], output: Path) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for wheel in sorted(wheels):
        name, version = _wheel_metadata(wheel)
        key = (name, version)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{name}=={version} --hash=sha256:{_sha256(wheel)}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def _download_exact_requirements(
    python_exe: str,
    requirement_files: list[Path],
    wheelhouse: Path,
    *,
    index_url: str = "",
    extra_index_urls: list[str] | None = None,
) -> list[Path]:
    before = {p.name for p in wheelhouse.glob("*.whl")}
    cmd = [python_exe, "-m", "pip", "download", "--dest", str(wheelhouse), "--only-binary=:all:"]
    if index_url:
        cmd.extend(["--index-url", index_url])
    for extra_index_url in extra_index_urls or []:
        cmd.extend(["--extra-index-url", extra_index_url])
    for req in requirement_files:
        cmd.extend(["-r", str(req)])
    _run(cmd)
    after = [p for p in wheelhouse.glob("*.whl") if p.name not in before]
    if not after:
        raise SystemExit("no wheels downloaded; exact-version input lock may be empty")
    return after


def _install_hash_locked_requirements(
    python_exe: str,
    requirement_files: list[Path],
    wheelhouse: Path,
    *,
    index_url: str = "",
    extra_index_urls: list[str] | None = None,
) -> None:
    wheels = _download_exact_requirements(
        python_exe,
        requirement_files,
        wheelhouse,
        index_url=index_url,
        extra_index_urls=extra_index_urls,
    )
    lock_path = wheelhouse / "resolved.requirements.lock"
    lines = _write_hash_locked_requirements(wheels, lock_path)
    if not lines:
        raise SystemExit("generated requirement lock is empty")
    _run(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "--ignore-installed",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "--require-hashes",
            "-r",
            str(lock_path),
        ]
    )


def _build_local_wheel(python_exe: str, source: Path, wheelhouse: Path) -> Path:
    before = {p.name for p in wheelhouse.glob("*.whl")}
    _run(
        [
            python_exe,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheelhouse),
            str(source),
        ]
    )
    new_wheels = [p for p in wheelhouse.glob("*.whl") if p.name not in before]
    if len(new_wheels) != 1:
        raise SystemExit(f"expected exactly one local wheel, found {len(new_wheels)}")
    return new_wheels[0]


def _install_local_wheel(python_exe: str, source: Path, wheelhouse: Path) -> None:
    wheel = _build_local_wheel(python_exe, source, wheelhouse)
    name, version = _wheel_metadata(wheel)
    lock_path = wheelhouse / "local.requirements.lock"
    lock_path.write_text(
        f"{name}=={version} --hash=sha256:{_sha256(wheel)}\n",
        encoding="utf-8",
    )
    _run(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "--ignore-installed",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "--no-deps",
            "--require-hashes",
            "-r",
            str(lock_path),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Install exact-version dependency locks by downloading wheels, generating a hash-locked "
            "temporary requirements file, and installing with --require-hashes."
        )
    )
    p.add_argument(
        "--requirements",
        action="append",
        default=[],
        help="Exact-version top-level requirements lock file. Repeatable.",
    )
    p.add_argument(
        "--install-local-wheel",
        action="store_true",
        help="Build the current repo (or --local-wheel-source) as a wheel and install it with a local hash lock.",
    )
    p.add_argument(
        "--local-wheel-source",
        default=".",
        help="Path to the local package source used when --install-local-wheel is enabled.",
    )
    p.add_argument(
        "--wheelhouse",
        default="",
        help="Wheelhouse directory. Defaults to a temporary directory that is cleaned up on success.",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for pip invocations. Defaults to the current interpreter.",
    )
    p.add_argument(
        "--index-url",
        default="",
        help="Primary package index URL passed through to pip download.",
    )
    p.add_argument(
        "--extra-index-url",
        action="append",
        default=[],
        help="Extra package index URL passed through to pip download. Repeatable.",
    )
    p.add_argument(
        "--keep-wheelhouse",
        action="store_true",
        help="Keep the wheelhouse directory after a successful run.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.requirements and not args.install_local_wheel:
        raise SystemExit("nothing to do: pass --requirements and/or --install-local-wheel")

    temp_dir = None
    if args.wheelhouse:
        wheelhouse = Path(args.wheelhouse).resolve()
        wheelhouse.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="yolozu_hash_install_")
        wheelhouse = Path(temp_dir)

    try:
        req_files = [Path(p).resolve() for p in args.requirements]
        if req_files:
            _install_hash_locked_requirements(
                args.python,
                req_files,
                wheelhouse,
                index_url=args.index_url,
                extra_index_urls=args.extra_index_url,
            )
        if args.install_local_wheel:
            _install_local_wheel(args.python, Path(args.local_wheel_source).resolve(), wheelhouse)
        print(f"wheelhouse={wheelhouse}")
        return 0
    finally:
        if temp_dir and not args.keep_wheelhouse:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Release announcement helper for GitHub Actions.

Runs on `release: published` to generate a deterministic post bundle, and can
optionally post to LinkedIn / X / Reddit when corresponding secrets exist.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleaseInfo:
    tag: str
    name: str
    url: str
    body: str


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate and optionally post release announcements.")
    p.add_argument(
        "--event-json",
        default=os.environ.get("GITHUB_EVENT_PATH"),
        help="GitHub event JSON path (default: $GITHUB_EVENT_PATH).",
    )
    p.add_argument("--out-dir", default="reports/announce", help="Output directory (default: reports/announce).")
    p.add_argument("--post", action="store_true", help="Actually post to configured SNS targets (default: off).")
    p.add_argument(
        "--platform",
        action="append",
        choices=("linkedin", "x", "reddit"),
        default=[],
        help="Platform to post (repeatable). Default: all configured platforms.",
    )
    p.add_argument("--strict-post", action="store_true", help="Fail if any configured post fails.")
    p.add_argument(
        "--reddit-subreddit",
        default=os.environ.get("YOLOZU_REDDIT_SUBREDDIT", ""),
        help="Reddit subreddit (default: $YOLOZU_REDDIT_SUBREDDIT).",
    )
    p.add_argument(
        "--reddit-kind",
        choices=("link", "text"),
        default=os.environ.get("YOLOZU_REDDIT_KIND", "link"),
        help="Reddit submission kind (default: link).",
    )
    p.add_argument(
        "--x-max-len",
        type=int,
        default=280,
        help="Max length for X text truncation (default: 280).",
    )
    return p


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_release_from_event(event_json: Path) -> ReleaseInfo:
    payload = _read_json(event_json)
    release = payload.get("release") or {}
    tag = str(release.get("tag_name") or "").strip()
    name = str(release.get("name") or "").strip() or tag
    url = str(release.get("html_url") or "").strip()
    body = str(release.get("body") or "")
    if not tag or not url:
        raise SystemExit("event json is missing release.tag_name or release.html_url")
    return ReleaseInfo(tag=tag, name=name, url=url, body=body)


def _normalize_body(body: str, *, max_chars: int = 1200) -> str:
    text = (body or "").strip()
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _x_text(info: ReleaseInfo, *, max_len: int) -> str:
    base = f"YOLOZU {info.tag} released: {info.url}"
    if len(base) <= max_len:
        return base
    return base[: max_len - 1].rstrip() + "…"


def _markdown_bundle(info: ReleaseInfo) -> str:
    body = _normalize_body(info.body)
    lines = [f"# {info.name}", "", f"- Tag: `{info.tag}`", f"- Release: {info.url}"]
    if body:
        lines.extend(["", "## Notes", "", body])
    return "\n".join(lines).strip() + "\n"


def _write_bundle(out_dir: Path, info: ReleaseInfo, *, x_max_len: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "kind": "release_announcement_bundle",
        "timestamp": _now_utc(),
        "release": {"tag": info.tag, "name": info.name, "url": info.url},
        "texts": {
            "x": _x_text(info, max_len=x_max_len),
            "markdown": _markdown_bundle(info),
        },
    }
    (out_dir / "announcement.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "announcement.md").write_text(bundle["texts"]["markdown"], encoding="utf-8")
    return bundle


def _maybe_post_linkedin(info: ReleaseInfo, *, strict: bool) -> dict:
    token = os.environ.get("LINKEDIN_ACCESS_TOKEN", "").strip()
    author_urn = os.environ.get("LINKEDIN_AUTHOR_URN", "").strip()
    if not token or not author_urn:
        return {"platform": "linkedin", "configured": False, "ok": True, "skipped": True, "reason": "missing secrets"}

    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover
        msg = f"missing dependency: requests ({exc})"
        if strict:
            raise SystemExit(msg)
        return {"platform": "linkedin", "configured": True, "ok": False, "skipped": False, "error": msg}

    text = f"YOLOZU {info.tag} released.\n\n{info.url}"
    payload = {
        "author": author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": "ARTICLE",
                "media": [{"status": "READY", "originalUrl": info.url}],
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }
    headers = {"Authorization": f"Bearer {token}", "X-Restli-Protocol-Version": "2.0.0"}
    resp = requests.post("https://api.linkedin.com/v2/ugcPosts", json=payload, headers=headers, timeout=30)
    ok = 200 <= int(resp.status_code) < 300
    out = {
        "platform": "linkedin",
        "configured": True,
        "ok": ok,
        "skipped": False,
        "status_code": int(resp.status_code),
        "response_head": (resp.text or "")[:500],
    }
    if strict and not ok:
        raise SystemExit(f"LinkedIn post failed: http {resp.status_code}")
    return out


def _maybe_post_x(text: str, *, strict: bool) -> dict:
    api_key = os.environ.get("X_API_KEY", "").strip()
    api_secret = os.environ.get("X_API_KEY_SECRET", "").strip()
    access_token = os.environ.get("X_ACCESS_TOKEN", "").strip()
    access_secret = os.environ.get("X_ACCESS_TOKEN_SECRET", "").strip()
    if not (api_key and api_secret and access_token and access_secret):
        return {"platform": "x", "configured": False, "ok": True, "skipped": True, "reason": "missing secrets"}

    try:
        from requests_oauthlib import OAuth1Session  # type: ignore
    except Exception as exc:  # pragma: no cover
        msg = f"missing dependency: requests-oauthlib ({exc})"
        if strict:
            raise SystemExit(msg)
        return {"platform": "x", "configured": True, "ok": False, "skipped": False, "error": msg}

    sess = OAuth1Session(
        client_key=api_key,
        client_secret=api_secret,
        resource_owner_key=access_token,
        resource_owner_secret=access_secret,
    )
    resp = sess.post("https://api.twitter.com/2/tweets", json={"text": text}, timeout=30)
    ok = 200 <= int(resp.status_code) < 300
    out = {
        "platform": "x",
        "configured": True,
        "ok": ok,
        "skipped": False,
        "status_code": int(resp.status_code),
        "response_head": (resp.text or "")[:500],
    }
    if strict and not ok:
        raise SystemExit(f"X post failed: http {resp.status_code}")
    return out


def _maybe_post_reddit(title: str, body_md: str, url: str, *, subreddit: str, kind: str, strict: bool) -> dict:
    client_id = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    refresh = os.environ.get("REDDIT_REFRESH_TOKEN", "").strip()
    user_agent = os.environ.get("REDDIT_USER_AGENT", "YOLOZU-CI/1.0").strip()
    if not (client_id and client_secret and refresh and subreddit):
        return {"platform": "reddit", "configured": False, "ok": True, "skipped": True, "reason": "missing secrets"}

    try:
        import praw  # type: ignore
    except Exception as exc:  # pragma: no cover
        msg = f"missing dependency: praw ({exc})"
        if strict:
            raise SystemExit(msg)
        return {"platform": "reddit", "configured": True, "ok": False, "skipped": False, "error": msg}

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=refresh,
        user_agent=user_agent,
    )
    sub = reddit.subreddit(subreddit)
    if kind == "text":
        submission = sub.submit(title=title, selftext=body_md)
    else:
        submission = sub.submit(title=title, url=url)

    return {
        "platform": "reddit",
        "configured": True,
        "ok": True,
        "skipped": False,
        "submission_url": getattr(submission, "url", None),
        "permalink": getattr(submission, "permalink", None),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    event_json_raw = str(args.event_json or "").strip()
    if not event_json_raw:
        raise SystemExit("--event-json is required (or set GITHUB_EVENT_PATH)")

    event_path = Path(event_json_raw)
    if not event_path.is_file():
        raise SystemExit(f"event json not found: {event_path}")

    out_dir = Path(str(args.out_dir)).resolve()
    info = _load_release_from_event(event_path)
    bundle = _write_bundle(out_dir, info, x_max_len=int(args.x_max_len))

    report: dict = {
        "kind": "release_announcement_post_report",
        "timestamp": _now_utc(),
        "release": bundle["release"],
        "ok": True,
        "results": [],
    }

    if not bool(args.post):
        (out_dir / "post_report.json").write_text(
            json.dumps({**report, "note": "post disabled; bundle generated only"}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(str(out_dir))
        return 0

    requested = list(args.platform or [])
    if not requested:
        requested = ["linkedin", "x", "reddit"]

    strict = bool(args.strict_post)
    results: list[dict] = []
    if "linkedin" in requested:
        results.append(_maybe_post_linkedin(info, strict=strict))
    if "x" in requested:
        results.append(_maybe_post_x(bundle["texts"]["x"], strict=strict))
    if "reddit" in requested:
        title = f"YOLOZU {info.tag} released"
        body_md = textwrap.dedent(
            f"""\
            YOLOZU {info.tag} is out.

            Release: {info.url}
            """
        ).strip()
        results.append(
            _maybe_post_reddit(
                title=title,
                body_md=body_md,
                url=info.url,
                subreddit=str(args.reddit_subreddit or "").strip(),
                kind=str(args.reddit_kind),
                strict=strict,
            )
        )

    report["results"] = results
    report["ok"] = all(bool(r.get("ok")) for r in results if bool(r.get("configured")))
    (out_dir / "post_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    if strict and not bool(report["ok"]):
        return 1

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


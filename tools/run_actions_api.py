#!/usr/bin/env python3
from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_actions_api.py",
        description="Run YOLOZU Actions/OpenAPI server.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080).")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn worker count (default: 1).")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (local development only).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import uvicorn

    uvicorn.run(
        "yolozu.integrations.actions_api:app",
        host=str(args.host),
        port=int(args.port),
        workers=int(args.workers),
        reload=bool(args.reload),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

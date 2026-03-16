#!/usr/bin/env bash
set -euo pipefail

exec /src/YOLOZU/.clusterfuzzlite/build.sh "$@"

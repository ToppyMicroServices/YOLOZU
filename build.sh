#!/usr/bin/env bash
set -euo pipefail

exec /src/.clusterfuzzlite/build.sh "$@"

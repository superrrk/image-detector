#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Flask serves the same API with clear model error reporting.
exec python3 -m server.app

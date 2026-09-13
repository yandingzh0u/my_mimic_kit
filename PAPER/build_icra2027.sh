#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TECTONIC_BIN="${TECTONIC_BIN:-/home/y/.local/bin/tectonic}"

cd "$SCRIPT_DIR"
"$TECTONIC_BIN" --keep-logs --keep-intermediates icra2027_paper.tex

pages="$(pdfinfo icra2027_paper.pdf | awk '/^Pages:/ {print $2}')"
if [[ "$pages" != "8" ]]; then
  echo "Expected exactly 8 pages, got $pages." >&2
  exit 1
fi

echo "Built $SCRIPT_DIR/icra2027_paper.pdf (8 pages)."

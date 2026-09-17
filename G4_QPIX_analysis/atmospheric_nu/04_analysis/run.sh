#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
base="$(cd "$here/.." && pwd)"
source "$base/env.sh"

genie="$("$ATMO_PYTHON" - "$base/01_genie/output" <<'PY'
import sys
from pathlib import Path
files = list(Path(sys.argv[1]).glob("*.gtrac.root"))
if not files:
    raise SystemExit(1)
print(max(files, key=lambda p: p.stat().st_mtime).resolve())
PY
)"

"$ATMO_PYTHON" "$here/analyze.py" \
  --genie "$genie" \
  --g4 "$base/02_qpix_geant4/output/atmospheric_g4.root" \
  --rtd "$base/03_qpix_rtd/output/atmospheric_rtd.root" \
  --output "$here/output"

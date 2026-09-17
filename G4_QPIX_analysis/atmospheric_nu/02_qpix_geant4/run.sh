#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
source "$here/../env.sh"

events=100
seed=35791
force=0
while (($#)); do
  case "$1" in
    --events) events="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --force) force=1; shift ;;
    *) echo "Unknown option $1" >&2; exit 2 ;;
  esac
done

input="$("$ATMO_PYTHON" - "$ATMO_BASE/01_genie/output" <<'PY'
import sys
from pathlib import Path
files = list(Path(sys.argv[1]).glob("*.gtrac.root"))
if not files:
    raise SystemExit(1)
print(max(files, key=lambda p: p.stat().st_mtime).resolve())
PY
)"
[[ -s "$input" ]]

out="$here/output"
mkdir -p "$out"
root="$out/atmospheric_g4.root"
if ((! force)) && [[ -e "$root" ]]; then
  echo "Output exists; use --force" >&2
  exit 1
fi

sed -e "s|@INPUT@|$input|" \
    -e "s|@OUTPUT@|$root|" \
    -e "s|@EVENTS@|$events|" \
    -e "s|@SEED1@|$seed|" \
    -e "s|@SEED2@|$((seed+1))|" \
    "$here/atmospheric.mac.in" >"$out/atmospheric.mac"

: >"$out/run.log"
exec >>"$out/run.log" 2>&1

"$QPIX_G4_DIR/build/app/G4_QPIX" "$out/atmospheric.mac"

"$ATMO_PYTHON" - "$root" <<'PY'
import sys
import awkward as ak
import uproot

path = sys.argv[1]
with uproot.open(path) as source:
    tree = source["event_tree"]
    if tree.num_entries == 0:
        raise SystemExit("ERROR: Q_PIX_GEANT4 output contains no events")
    hits = tree["number_hits"].array(library="np")
    pdg = ak.flatten(tree["generator_final_particle_pdg_code"].array(library="ak"), axis=None)
    if int(hits.sum()) == 0 or len(pdg) == 0 or not ak.any(pdg != 0):
        raise SystemExit(
            "ERROR: Q_PIX_GEANT4 produced no physical GENIE deposits "
            "(zero hits or only PDG-0 dummy primaries)"
        )
    print(f"validated_events={tree.num_entries} total_geant4_hits={int(hits.sum())}")
PY

shasum -a 256 "$root"

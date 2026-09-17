#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
source "$here/../env.sh"

force=0
while (($#)); do
  case "$1" in
    --force) force=1; shift ;;
    *) echo "Unknown option $1" >&2; exit 2 ;;
  esac
done

input="$here/../02_qpix_geant4/output/atmospheric_g4.root"
out="$here/output"
mkdir -p "$out"
root="$out/atmospheric_rtd.root"
if ((! force)) && [[ -e "$root" ]]; then
  echo "Output exists; use --force" >&2
  exit 1
fi
(( force )) && rm -f -- "$root"

: >"$out/run.log"
exec >>"$out/run.log" 2>&1

"$QPIX_RTD_DIR/RTD/build/RTD" -i "$input" -o "$root" -t 6250 -s .4 -h 1e-8 -d 1 --nonoise

"$ATMO_PYTHON" - "$root" <<'PY'
import sys
import awkward as ak
import uproot

with uproot.open(sys.argv[1]) as source:
    tree = source["event_tree"]
    resets = tree["pixel_reset"].array(library="ak")
    total = int(ak.sum(ak.num(resets, axis=2)))
    if tree.num_entries == 0 or total == 0:
        raise SystemExit(
            "ERROR: QPIX-RTD output has no resets; inspect the Geant4 hit "
            "sanity check and RTD charge configuration"
        )
    print(f"validated_events={tree.num_entries} total_resets={total}")
PY

shasum -a 256 "$root"

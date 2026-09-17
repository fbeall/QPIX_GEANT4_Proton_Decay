#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
source "$here/../env.sh"

events=100
seed=24680
run=2001
emin=0.1
emax=10
force=0

while (($#)); do
  case "$1" in
    --events) events="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --energy-min) emin="$2"; shift 2 ;;
    --energy-max) emax="$2"; shift 2 ;;
    --force) force=1; shift ;;
    *) echo "Unknown option $1" >&2; exit 2 ;;
  esac
done

flux="$here/flux/dune-ally-20-12-solmin-genie.d"
[[ -s "$flux" ]] || "$ATMO_PYTHON" "$here/prepare_flux.py"

xsec="$here/xsec/argon40_G18_02a_00_000_4flavor.xml"
"$here/prepare_xsec.sh"
for pdg in 12 -12 14 -14; do
  grep -q "nu:${pdg};tgt:1000180400" "$xsec" || {
    echo "Incomplete spline file: missing PDG $pdg" >&2
    exit 1
  }
done

out="$here/output"
mkdir -p "$out"
ghep="$out/atmospheric.${run}.ghep.root"
gtrac="$out/atmospheric.${run}.gtrac.root"
if ((! force)) && { [[ -e "$ghep" ]] || [[ -e "$gtrac" ]]; }; then
  echo "Output exists; use --force" >&2
  exit 1
fi

: >"$out/run.log"
exec >>"$out/run.log" 2>&1

spec="HAKKM:$flux[14],$flux[-14],$flux[12],$flux[-12]"
cmd=(gevgen_atmo -n "$events" -r "$run" -E "$emin,$emax" -f "$spec" -g 1000180400 -o "$out/atmospheric" --seed "$seed" --tune G18_02a_00_000 --cross-sections "$xsec" --message-thresholds "$GENIE/config/Messenger_laconic.xml")
printf 'command='
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
gntpc -i "$ghep" -o "$gtrac" -f rootracker -c
shasum -a 256 "$flux" "$xsec" "$ghep" "$gtrac"

#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
source "$here/../env.sh"

out="$here/xsec"
mkdir -p "$out"
xsec="$out/argon40_G18_02a_00_000_4flavor.xml"
seed_xsec="${ATMO_XSEC_SEED:-$PROGRAMS_DIR/DUNE-Flux-Files/gxsec.xml}"
if [[ ! -s "$seed_xsec" ]]; then
  seed_xsec="/Users/fb_local/Downloads/ProtonDecayWorkflowBundle/DUNE-Flux-Files/gxsec.xml"
fi

force=0
[[ "${1:-}" == "--force" ]] && force=1

complete=1
if [[ -s "$xsec" ]]; then
  for pdg in 12 -12 14 -14; do
    grep -q "nu:${pdg};tgt:1000180400" "$xsec" || complete=0
  done
else
  complete=0
fi

if (( complete && ! force )); then
  echo "Complete spline file already exists: $xsec"
  exit 0
fi

if [[ ! -s "$seed_xsec" ]]; then
  echo "Missing seed cross-section file: $seed_xsec" >&2
  exit 1
fi

tmp="$out/.argon40_4flavor.$$.xml"
trap 'rm -f -- "$tmp"' EXIT
log="$out/prepare_xsec.log"

echo "Generating four-flavor Ar-40 splines. This one-time GENIE calculation can take tens of minutes."
gmkspl -p 12,-12,14,-14 -t 1000180400 -n 100 -e 20 \
  -o "$tmp" --tune G18_02a_00_000 \
  --input-cross-sections "$seed_xsec" \
  --message-thresholds "$GENIE/config/Messenger_laconic.xml" >"$log" 2>&1 || {
    echo "Spline generation failed; see $log" >&2
    exit 1
  }

for pdg in 12 -12 14 -14; do
  grep -q "nu:${pdg};tgt:1000180400" "$tmp" || {
    echo "Missing PDG $pdg splines" >&2
    exit 1
  }
done

mv -f -- "$tmp" "$xsec"
trap - EXIT
shasum -a 256 "$xsec"

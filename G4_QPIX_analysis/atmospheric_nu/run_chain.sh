#!/usr/bin/env bash
set -euo pipefail

base="$(cd "$(dirname "$0")" && pwd)"
source "$base/env.sh"

events=100
seed=24680
run=2001
force=0
plots=0

while (($#)); do
  case "$1" in
    --events) events="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --force) force=1; shift ;;
    --plots-only) plots=1; shift ;;
    *)
      echo "Usage: $0 [--events N] [--seed N] [--run N] [--force] [--plots-only]" >&2
      exit 2
      ;;
  esac
done

extra=()
(( force )) && extra+=(--force)

if ((! plots)); then
  "$base/01_genie/run.sh" --events "$events" --seed "$seed" --run "$run" "${extra[@]}"
  "$base/02_qpix_geant4/run.sh" --events "$events" --seed "$((seed + 100))" "${extra[@]}"
  "$base/03_qpix_rtd/run.sh" "${extra[@]}"
fi

"$base/04_analysis/run.sh"
echo "Atmospheric-neutrino chain complete."

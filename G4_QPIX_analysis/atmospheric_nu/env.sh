#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${ZSH_VERSION:-}" ]]; then
  script_path="${(%):-%x}"
else
  script_path="${BASH_SOURCE[0]}"
fi
ATMO_BASE="$(cd "$(dirname "$script_path")" && pwd)"
QPIX_G4_DIR="$(cd "$ATMO_BASE/../.." && pwd)"
PROGRAMS_DIR="$(cd "$QPIX_G4_DIR/.." && pwd)"

export ATMO_BASE QPIX_G4_DIR PROGRAMS_DIR
export GENIE="${GENIE:-$PROGRAMS_DIR/GENIE/Generator}"
export ROOTSYS="${ROOTSYS:-$PROGRAMS_DIR/root_install_6.30.08_pythia6}"
export G4_INSTALL="${G4_INSTALL:-$PROGRAMS_DIR/geant4_install_10.7.4}"
export QPIX_RTD_DIR="${QPIX_RTD_DIR:-$PROGRAMS_DIR/qpixrtd}"
export ATMO_PYTHON="${ATMO_PYTHON:-$QPIX_G4_DIR/uproot_env/bin/python}"

export PATH="$ROOTSYS/bin:$GENIE/bin:$G4_INSTALL/bin:$PATH"

for geant4_setup in "$G4_INSTALL/bin/geant4.sh" "$PROGRAMS_DIR/geant4_build_10.7.4/InstallTreeFiles/geant4.sh"; do
  if [[ -f "$geant4_setup" ]]; then
    old_pwd="$PWD"
    cd "$(dirname "$geant4_setup")"
    # shellcheck source=/dev/null
    source "./$(basename "$geant4_setup")"
    cd "$old_pwd"
    break
  fi
done

for maybe in \
  "$ROOTSYS/lib" \
  "$GENIE/lib" \
  "$G4_INSTALL/lib" \
  "$PROGRAMS_DIR/pythia6/v6_428/lib" \
  "$PROGRAMS_DIR/lhapdf-5.9.1/lib/.libs" \
  "$PROGRAMS_DIR/marley/build" \
  "$QPIX_G4_DIR/build/src" \
  "$QPIX_RTD_DIR/Library" \
  "$QPIX_RTD_DIR/Build/source"; do
  [[ -d "$maybe" ]] || continue
  export DYLD_LIBRARY_PATH="$maybe:${DYLD_LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$maybe:${LD_LIBRARY_PATH:-}"
done

export LHAPATH="${LHAPATH:-$PROGRAMS_DIR/lhapdf-5.9.1/share/lhapdf/PDFsets}"
export GXMLPATH="${GXMLPATH:-$GENIE/config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ATMO_BASE/.matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ATMO_BASE/.cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

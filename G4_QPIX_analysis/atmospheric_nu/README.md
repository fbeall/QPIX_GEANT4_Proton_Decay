# Atmospheric Neutrino Section 6 Workflow on macOS

This folder installs only Section 6 of `ProtonDecayWorkflowInstructions.md`: the atmospheric-neutrino background chain. It is self-contained under `G4_QPIX_analysis/atmospheric_nu/` and uses the existing local Q_PIX_GEANT4, GENIE, ROOT, Geant4, and qpixrtd installations.

## What Was Installed

- `01_genie/prepare_flux.py`: downloads and converts the DUNE Honda-format atmospheric flux table.
- `01_genie/prepare_xsec.sh`: validates or builds the four-flavor Ar-40 GENIE spline XML.
- `01_genie/run.sh`: runs `gevgen_atmo` and converts GHEP to gRooTracker.
- `02_qpix_geant4/atmospheric.mac.in` and `02_qpix_geant4/run.sh`: runs Q_PIX_GEANT4 with randomized atmospheric vertices in the 230 x 600 x 360 cm HD active volume.
- `03_qpix_rtd/run.sh`: runs qpixrtd on the atmospheric Geant4 output.
- `04_analysis/analyze.py` and `04_analysis/run.sh`: performs the Section 6 analysis checks and plots.

The Geant4 source was updated to support the Section 6 macro command `/inputs/randomize_vertex_position true`, to dispatch `Atmospheric` GENIE events through the GENIE primary generator, and to store GENIE vertex truth in cm.

## macOS Differences From the PDF

- The Ubuntu scripts used GNU `find -printf` and `sha256sum`; these scripts use Python newest-file selection and `shasum -a 256`.
- `env.sh` points at the local Mac paths:
  - `GENIE=/Users/fb_local/Programs/GENIE/Generator`
  - `ROOTSYS=/Users/fb_local/Programs/root_install_6.30.08_pythia6`
  - `G4_INSTALL=/Users/fb_local/Programs/geant4_install_10.7.4`
  - `QPIX_RTD_DIR=/Users/fb_local/Programs/qpixrtd`
- `geant4.sh` is sourced from inside its own directory because the Mac install cannot self-locate when sourced elsewhere.
- Matplotlib/fontconfig caches are redirected into this workflow folder.
- The CMake build on this Mac required explicit include/link flags:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="/Users/fb_local/Programs/geant4_install_10.7.4;/Users/fb_local/Programs/root_install_6.30.08_pythia6" \
  -DGeant4_DIR=/Users/fb_local/Programs/geant4_install_10.7.4/lib/Geant4-10.7.4 \
  -DROOT_DIR=/Users/fb_local/Programs/root_install_6.30.08_pythia6/cmake \
  -DCMAKE_CXX_FLAGS="-I/Users/fb_local/Programs/geant4_install_10.7.4/include/Geant4 -I/Users/fb_local/Programs/marley/include" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/Users/fb_local/Programs/marley/build" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/Users/fb_local/Programs/marley/build"
cmake --build build -j2
```

## Flux and Spline Record

The flux download was run from this folder:

```bash
./G4_QPIX_analysis/atmospheric_nu/01_genie/prepare_flux.py
```

It downloaded `dune-ally-20-12-solmin.raw.d`, retained the 0.1 GeV and above rows, validated 240 angular blocks, and wrote:

```text
G4_QPIX_analysis/atmospheric_nu/01_genie/flux/dune-ally-20-12-solmin-genie.d
sha256 9d8b3ea89723a8019e418f5fde4cb76005100136b2d306e6f6824d1898ebb3b2
```

An attempted fresh `gmkspl` run did not advance beyond initial GENIE configuration after several minutes. To avoid leaving a partial XML, it was stopped and the complete four-flavor spline from the attached Section 6 bundle was installed:

```text
G4_QPIX_analysis/atmospheric_nu/01_genie/xsec/argon40_G18_02a_00_000_4flavor.xml
sha256 b53ae3893a3ee6764f092238ac534a7a879e52033ca67f21e90d9416da09a18f
```

`prepare_xsec.sh` validates that PDGs `12`, `-12`, `14`, and `-14` are all present.

## Running Section 6

From the repository root:

```bash
./G4_QPIX_analysis/atmospheric_nu/01_genie/prepare_flux.py
./G4_QPIX_analysis/atmospheric_nu/01_genie/prepare_xsec.sh

./G4_QPIX_analysis/atmospheric_nu/run_chain.sh --events 2 --seed 24680 --run 2001 --force
./G4_QPIX_analysis/atmospheric_nu/run_chain.sh --events 10000 --seed 24680 --run 2002 --force
./G4_QPIX_analysis/atmospheric_nu/run_chain.sh --plots-only
```

## Completed Checks

Smoke test completed successfully for 2 events:

- GENIE events: 2
- Geant4 events: 2
- RTD events: 2
- RTD resets: 3638
- Event counts matched.
- Vertices were inside the HD active volume.
- Deposited energy and resets were nonzero.

Smoke analysis output:

```text
G4_QPIX_analysis/atmospheric_nu/04_analysis/output/summary.json
G4_QPIX_analysis/atmospheric_nu/04_analysis/output/plots/
```

Production was completed through GENIE and Geant4 for 10,000 events:

```text
GENIE atmospheric.2002.ghep.root sha256 f215e42920facc2387f0437f0e28c6b23dbaa9faf782e1a4894e76e4eb304e50
GENIE atmospheric.2002.gtrac.root sha256 6941228bc5b4c9a67a96d82dd2daa4c059bfb084f2ce7a2a9b7fd5ba03ac1d13
Geant4 atmospheric_g4.root sha256 4007345de73cf535e3b68c8ad7188f2f7a9b40aa20b4c2edf7859470fb18a43f
Geant4 validated_events=10000 total_geant4_hits=33356815
```

The full 10,000-event RTD production was started but not completed in this foreground task. It reached about event 71 after several minutes and was stopped because the extrapolated runtime was hours. The partial RTD file is not a completed production artifact and should be regenerated before using production-level RTD or analysis results.

## Analysis Script

`04_analysis/analyze.py` checks:

- GENIE flavor composition for `nu_e`, `anti-nu_e`, `nu_mu`, and `anti-nu_mu`.
- GENIE energy and zenith spectra.
- Stable final-state multiplicity.
- Geant4 randomized vertex distributions and containment in 230 x 600 x 360 cm.
- Total deposited energy.
- K+ occurrence.
- Geant4 hit-level dE/dx when hit energy and length branches are available.
- RTD reset multiplicity.
- GENIE, Geant4, and RTD event-count agreement.

For 2-event smoke tests, the flavor check verifies accounting but does not require all four flavors to appear, because that is statistically impossible for only two events.

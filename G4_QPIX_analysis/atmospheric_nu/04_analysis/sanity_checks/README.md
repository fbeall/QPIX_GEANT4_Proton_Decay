# Atmospheric-Neutrino Particle Sanity Checks

This directory contains two truth-level checks for the Q-Pix Geant4 atmospheric-neutrino sample. They use `event_tree` from the Geant4 ROOT output and do not depend on qpixRTD.

## Running the scripts

From the repository root, use the atmospheric-neutrino Python environment:

```bash
./uproot_env/bin/python G4_QPIX_analysis/atmospheric_nu/04_analysis/sanity_checks/muon_distribution.py
./uproot_env/bin/python G4_QPIX_analysis/atmospheric_nu/04_analysis/sanity_checks/kaon_distribution.py
```

The default input is:

```text
G4_QPIX_analysis/atmospheric_nu/02_qpix_geant4/output/atmospheric_g4.root
```

The scripts create:

```text
G4_QPIX_analysis/atmospheric_nu/04_analysis/sanity_checks/outputs/muon_distribution_checks.pdf
G4_QPIX_analysis/atmospheric_nu/04_analysis/sanity_checks/outputs/kaon_distribution_checks.pdf
```

Every plotted PDF page is also saved as a high-resolution standalone PNG under:

```text
G4_QPIX_analysis/atmospheric_nu/04_analysis/sanity_checks/outputs/muon_pngs/
G4_QPIX_analysis/atmospheric_nu/04_analysis/sanity_checks/outputs/kaon_pngs/
```

The last two pages of each PDF form a data-driven report. They define the truth populations, quote statistics measured
from the selected input file, assess whether the distributions are physically sensible, and describe limitations on
the conclusions.

Analyze another Geant4 file or write another PDF with `--input` and `--output`:

```bash
./uproot_env/bin/python G4_QPIX_analysis/atmospheric_nu/04_analysis/sanity_checks/muon_distribution.py \
  --input /path/to/atmospheric_g4.root \
  --output /path/to/muon_checks.pdf
```

Use `--help` on either script for its command-line summary.

## Truth definitions and units

- "Generator initial state" means the incoming atmospheric neutrino and argon target before the modeled interaction.
- "Generator final state" means particles stored in `generator_final_particle_*`. GENIE has modeled the
  neutrino-argon interaction and intranuclear final-state interactions, and this is the resulting particle list handed
  to Geant4. These entries are interaction products, not detector hits and not later Geant4 secondaries.
- "G4 primary track" means a transported particle with `particle_parent_track_id == 0`.
- "G4 secondary track" means a particle subsequently created by a Geant4 decay or interaction and therefore having a
  nonzero parent track ID.
- The atmospheric Geant4 ROOT branches used here store energy, mass, and momentum in MeV-based units. The plots therefore use MeV and MeV/c.
- `cos(theta_z)` is defined as `pz / |p|` in the simulation coordinate system. It is not automatically the conventional atmospheric zenith angle unless the detector axes and flux convention identify `z` as zenith.
- The plots contain unweighted counts and fractions for the generated sample. They are not absolute rates until flux weights, exposure, fiducial mass, oscillation weights, and any generator oversampling are applied.

## `muon_distribution.py`

The script selects generator final-state `mu-` (PDG 13) and `mu+` (PDG -13). Its PDF contains four pages:

1. **Truth overview:** muon charge counts, event-level muon multiplicity, and the fraction of generated interactions containing a muon.
2. **Muon kinematics:** kinetic-energy and momentum spectra separated by charge.
3. **Angular distributions:** `cos(theta_z)` and the opening angle between each final-state muon and its event's incoming neutrino.
4. **Neutrino-muon correlations:** incoming neutrino energy versus final-state muon energy, and neutrino-muon opening angle versus incoming neutrino energy.
5. **Two-page report:** sample-specific accounting, generator-final-state definition, charge-sign consistency,
   kinematic summaries, angular interpretation, and validation limitations.

Useful checks include:

- Most ordinary charged-current muon-neutrino events should have one leading muon.
- The muon energy must not exceed the incoming neutrino energy apart from tiny numerical effects or a misunderstood unit/schema convention.
- The neutrino-muon direction should become more forward-correlated as neutrino energy increases. Low-energy atmospheric events can have broad opening angles.
- `mu-` should primarily accompany `nu_mu` charged-current interactions and `mu+` should primarily accompany anti-`nu_mu` interactions. Secondary muons from hadron decay can add exceptions and multiplicities above one.

## `kaon_distribution.py`

The script selects PDGs 321, -321, 311, -311, 310, and 130, corresponding to charged kaons, generator neutral flavor states, and physical short/long neutral states. Its PDF contains five pages:

1. **Truth overview:** generator final-state species counts, kaon multiplicity, and the kaon-event fraction.
2. **Kaon kinematics:** generator final-state kinetic-energy and momentum spectra by species.
3. **Direction and production energy:** kaon `cos(theta_z)` and the incoming-neutrino energy associated with each produced kaon.
4. **Production and handoff:** kaon-event fraction versus incoming neutrino energy, plus generator final-state versus Geant4-primary species counts.
5. **Geant4 transport:** initial kinetic energy and stored decay-flag counts for transported primary kaons.
6. **Three-page report:** sample-specific accounting, generator-final-state definition, species interpretation,
   production-energy behavior, transport handoff, decay outcomes, and validation limitations.

Neutral-kaon bookkeeping needs care. A generator `K0` or anti-`K0` may be handed to Geant4 as the physical `K0S` or `K0L` state, so exact per-PDG agreement between the generator and transported-primary bars is not required. Compare the neutral-kaon totals as well as individual labels.

Kaon production is rare in this 10,000-event sample, so statistical fluctuations are expected. A physically sensible sample generally shows kaons concentrated at higher neutrino energies, low event multiplicity, and more `K+` than `K-` in neutrino-dominated argon interactions. These are qualitative checks; exact spectra and rates depend strongly on GENIE's interaction, hadronization, nuclear-remnant, and final-state-interaction models.

## Comparison strategy

These PDFs are strongest as regression and bookkeeping tests. For physics validation, compare distributions at three levels:

1. Compare incoming neutrino energy, flavor, and direction against the atmospheric flux model used to generate the sample.
2. Compare generator final-state muon and kaon distributions against GENIE predictions made with the same flux, target, tune, and event weights. Repeat with GENIE systematic variations or another generator such as NuWro or NEUT to estimate model dependence.
3. Compare relevant differential cross sections and particle-production rates with external neutrino-argon or nearby-target measurements. Beam data do not reproduce the atmospheric flux directly, but they test the interaction model that is folded with that flux.

The expected event distribution is a convolution of atmospheric flux, oscillation probability, neutrino-nucleus cross section, nuclear effects/final-state interactions, target composition, and acceptance. There is therefore no single closed-form theoretical muon or kaon spectrum to overlay on these unweighted detector-level plots. A controlled generator-level prediction with the same configuration is the appropriate quantitative reference.

## `vertex_distribution.py`

This script checks whether atmospheric-neutrino interaction vertices are uniformly distributed through the configured
`230 x 600 x 360 cm` active volume. It reads the common event vertex from `generator_initial_particle_x/y/z` and
verifies that all generator initial-state particles in each event share that vertex.

Run it from the repository root:

```bash
./uproot_env/bin/python G4_QPIX_analysis/atmospheric_nu/04_analysis/sanity_checks/vertex_distribution.py
```

It writes three PNG figures and a text summary under `outputs/vertex_distribution/`:

1. **3D vertex display:** a transparent point cloud inside a wireframe active-volume boundary. Vertex `z` supplies the
   color only to improve depth perception.
2. **Orthogonal projections:** binned `x-y`, `x-z`, and `y-z` density maps. These are generally more sensitive than the
   3D view to holes, planes, edge effects, or localized clustering.
3. **Uniformity checks:** one-dimensional equal-width occupancy histograms and standardized residuals for each axis.
   A reduced chi-square near one and pulls fluctuating around zero are consistent with uniform random sampling.

The script also checks containment, reports coordinate ranges and means, and accepts alternate dimensions through
`--x-length`, `--y-length`, and `--z-length`.

## `atmospheric_neutrino_truth.py`

This script studies the incoming atmospheric neutrino and its generator-level interaction outcome. Run it with:

```bash
./uproot_env/bin/python G4_QPIX_analysis/atmospheric_nu/04_analysis/sanity_checks/atmospheric_neutrino_truth.py
```

It creates `outputs/atmospheric_neutrino_truth_checks.pdf`, a separate
`outputs/atmospheric_neutrino_truth_report.pdf`, and five standalone figures under `outputs/neutrino_pngs/`:

1. Incoming flavor counts and neutrino-versus-antineutrino interaction counts.
2. Incoming energy spectra and energy-dependent flavor fractions.
3. Detector-coordinate `cos(theta_z)` and azimuth distributions.
4. Energy-direction density and median-energy directional profile.
5. Charged-current-like final-state classification and final-state multiplicity versus incoming energy.

The raw distributions describe generated interactions, not the atmospheric flux alone. They include the effects of
flux sampling, neutrino/antineutrino cross sections, the argon target, and the generator configuration. Oscillation and
event-weight treatment must be matched before comparing these counts with an underground detector prediction.

### Statistical annotations

Across the muon, kaon, and atmospheric-neutrino studies, continuous distributions show the sample size, median,
mean, and central 90% interval where space permits. The median is the primary typical-event summary because the
energy and momentum spectra are right-skewed; the mean is retained to expose sensitivity to the high-energy tail.
Multiplicity plots instead emphasize exact event counts and fractions because they are discrete. Ordinary arithmetic
means are intentionally omitted for azimuth, which is a circular variable and requires circular statistics.

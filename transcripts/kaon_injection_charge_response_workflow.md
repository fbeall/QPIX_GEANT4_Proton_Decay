# Kaon Injection and QPix Charge Response Workflow

Generated on May 20, 2026.

This transcript explains how to inject a controlled charged kaon into the QPix Geant4 detector, propagate it through the detector simulation, convert the Geant4 energy deposits into QPix reset/charge response with Q_PIX_RTD, and visualize the result with qpixar.

The purpose of this workflow is to test whether the detector and reconstruction chain can see a kaon-like topology before moving to a full GENIE-generated proton decay event.

## Goal

Before using GENIE to generate a full proton decay event, it is useful to run a simpler controlled test:

```text
single injected K+
  -> Geant4 detector simulation
  -> Q_PIX_GEANT4 ROOT output
  -> Q_PIX_RTD charge/reset response
  -> qpixar visualization
```

This isolates the detector-response question:

```text
If I put a kaon into the detector, can I see its track, decay daughters,
energy deposits, and QPix reset response?
```

## Which Tool Does What?

### Geant4 / Q_PIX_GEANT4

Use Geant4 for the first controlled kaon injection.

Q_PIX_GEANT4 uses Geant4's General Particle Source, abbreviated GPS, through macro commands such as:

```text
/gps/particle kaon+
/gps/pos/centre 115 475 150 cm
/gps/ene/mono 105 MeV
```

This is the right tool when the goal is:

- inject one known particle type,
- choose its starting position,
- choose its kinetic energy,
- choose its direction,
- let Geant4 propagate it and its decay/interaction products.

### GENIE

GENIE is not the first tool to use for this specific diagnostic.

GENIE is useful later when the goal is to generate a realistic physics event, such as a proton decay final state or neutrino interaction. For the first kaon response test, GENIE adds unnecessary complexity because it controls the event generation rather than giving a simple hand-injected particle.

Use GENIE later after verifying that the detector and QPix response chain behaves sensibly for a known kaon.

### ROOT

ROOT is the file format and inspection/analysis tool here.

ROOT is not used to generate the kaon. Instead:

- Q_PIX_GEANT4 writes a ROOT file containing Geant4 truth and hit information.
- Q_PIX_RTD reads that ROOT file and writes a new ROOT file with QPix pixel/reset branches.
- ROOT, uproot, or qpixar can inspect those output files.

### Q_PIX_RTD

Q_PIX_RTD converts Geant4 hit information into QPix detector response.

It models the detector readout effects such as:

- pixelization,
- charge drift,
- diffusion,
- recombination,
- reset threshold,
- reset time behavior.

This is the step that turns Geant4 energy deposits into the charge/reset pattern you actually want to reconstruct.

### qpixar

qpixar is used for visualization and quick analysis.

For this workflow, use:

```sh
python qpix_ed.py path/to/file.root --3d
```

or, for a combined view of all events:

```sh
python qpix_ed.py path/to/file.root --3d --all-events
```

## Files Created

The Geant4 macro created for this test is:

```text
/Users/fb_local/Programs/Q_PIX_GEANT4/macros/single_kaon.mac
```

The Geant4 output file produced by running this macro is:

```text
/Users/fb_local/Programs/Q_PIX_GEANT4/output/single_kaon.root
```

The Q_PIX_RTD output file produced from the Geant4 output is:

```text
/Users/fb_local/Programs/qpixrtd/single_kaon_RTD.root
```

## The Kaon Macro

The full macro is:

```text
# set verbosity
/control/verbose 1
/run/verbose 1
/tracking/verbose 0

# The detector configuration (HD, VD, or TS)
/geometry/detector_configuration HD

# output path
/inputs/output_file ./output/single_kaon.root

# initialize run
/run/initialize
/random/setSeeds 0 31

# limit radioactive decays
#/grdm/nucleusLimits 1 35 1 17  # aMin aMax zMin zMax

# K+ from p -> K+ nu has p ~= 339 MeV/c, so T_K ~= 105 MeV.
/gps/particle kaon+

# point source near the HD detector center
/gps/pos/type Point
/gps/pos/centre 115 475 150 cm

# fixed direction along -y
/gps/ang/type iso
/gps/ang/rot1 -1 0 0
/gps/ang/rot2 0 0 1
/gps/ang/mintheta 0 deg
/gps/ang/maxtheta 0 deg

# kinetic energy
/gps/ene/type Mono
/gps/ene/mono 105 MeV

# run
/run/beamOn 100
```

## Macro Syntax Explained

### Verbosity

```text
/control/verbose 1
/run/verbose 1
/tracking/verbose 0
```

These control how much Geant4 prints to the terminal.

- `/control/verbose 1`: moderate command-level verbosity.
- `/run/verbose 1`: moderate run-level verbosity.
- `/tracking/verbose 0`: do not print every tracking step. This keeps output manageable.

### Detector Configuration

```text
/geometry/detector_configuration HD
```

This selects the horizontal-drift detector configuration used by Q_PIX_GEANT4.

The existing examples also support comments for `VD` and `TS`, but this test used `HD`.

### Output File

```text
/inputs/output_file ./output/single_kaon.root
```

This tells Q_PIX_GEANT4 where to write the Geant4 simulation output.

The path is relative to the directory where the executable is run. In this workflow, the command is run from:

```text
/Users/fb_local/Programs/Q_PIX_GEANT4
```

so the output becomes:

```text
/Users/fb_local/Programs/Q_PIX_GEANT4/output/single_kaon.root
```

### Run Initialization

```text
/run/initialize
/random/setSeeds 0 31
```

`/run/initialize` initializes the detector, physics list, and run manager.

`/random/setSeeds 0 31` sets reproducible random seeds. Keeping the seeds fixed is helpful while debugging because the same macro gives repeatable events.

### Radioactive Decay Limit

```text
#/grdm/nucleusLimits 1 35 1 17  # aMin aMax zMin zMax
```

This line is intentionally commented out.

The original QPix installation instructions note that this line can cause issues with newer Geant4 behavior. It is not needed for this kaon particle-gun test.

### Particle Type

```text
/gps/particle kaon+
```

This is the most important line for the injection test.

`/gps/particle` is a Geant4 General Particle Source command. It tells Geant4 what primary particle to inject.

Useful particle names include:

```text
kaon+
kaon-
pi+
pi-
mu+
mu-
proton
neutron
gamma
e-
e+
```

For a proton decay channel like:

```text
p -> K+ + nu
```

the relevant visible strange particle is usually `kaon+`.

### Initial Position

```text
/gps/pos/type Point
/gps/pos/centre 115 475 150 cm
```

This injects the kaon from a single point.

The position is in centimeters and follows the coordinate convention used by Q_PIX_GEANT4. This point was copied from the existing single-muon macro and is near the center of the HD detector geometry used in these tests.

For early detector-response tests, a point source is cleaner than a distributed source because it makes the topology easier to interpret.

### Direction

```text
/gps/ang/type iso
/gps/ang/rot1 -1 0 0
/gps/ang/rot2 0 0 1
/gps/ang/mintheta 0 deg
/gps/ang/maxtheta 0 deg
```

This uses Geant4 GPS angular commands to make a fixed direction.

An attempted command:

```text
/gps/ang/type beam
```

was not accepted by this Geant4 build. The working syntax follows the existing Q_PIX_GEANT4 muon macro: use `iso` with zero angular opening.

The important part is:

```text
/gps/ang/mintheta 0 deg
/gps/ang/maxtheta 0 deg
```

That collapses the distribution to a single direction rather than a cone.

The chosen rotation basis points the particle along the detector's `-y` direction, matching the style of the existing Q_PIX_GEANT4 single-muon macro.

### Energy

```text
/gps/ene/type Mono
/gps/ene/mono 105 MeV
```

This injects a monoenergetic charged kaon with 105 MeV kinetic energy.

The value is motivated by two-body proton decay at rest:

```text
p -> K+ + nu
```

For this decay, the kaon momentum is about:

```text
p_K ~= 339 MeV/c
```

Using the charged kaon mass:

```text
m_K ~= 493.7 MeV/c^2
```

the total energy is:

```text
E_K = sqrt(p_K^2 + m_K^2)
```

and the kinetic energy is:

```text
T_K = E_K - m_K ~= 105 MeV
```

This is not yet a full GENIE proton-decay event; it is a controlled detector-response approximation.

### Number of Events

```text
/run/beamOn 100
```

This generates 100 kaon events.

For fast debugging, it is reasonable to temporarily reduce this to:

```text
/run/beamOn 5
```

or:

```text
/run/beamOn 10
```

The 100-event kaon RTD step took several minutes because the kaon events generated many more Geant4 hits than the earlier muon test.

## Step 1: Run The Geant4 Kaon Simulation

Source the environment:

```sh
source /Users/fb_local/Programs/qpix-root630-pythia6-env.sh
```

Go to the Q_PIX_GEANT4 directory:

```sh
cd /Users/fb_local/Programs/Q_PIX_GEANT4
```

Run the macro:

```sh
./build/app/G4_QPIX macros/single_kaon.mac
```

This produced:

```text
/Users/fb_local/Programs/Q_PIX_GEANT4/output/single_kaon.root
```

The file size after the test run was about:

```text
14 MB
```

Verify:

```sh
ls -lh /Users/fb_local/Programs/Q_PIX_GEANT4/output/single_kaon.root
```

## Step 2: Convert To QPix Charge/Reset Response

Use Q_PIX_RTD after the Geant4 output exists.

Source the same environment:

```sh
source /Users/fb_local/Programs/qpix-root630-pythia6-env.sh
```

Go to the RTD directory:

```sh
cd /Users/fb_local/Programs/qpixrtd
```

Run the RTD example executable:

```sh
./EXAMPLE/build/EXAMPLE \
  /Users/fb_local/Programs/Q_PIX_GEANT4/output/single_kaon.root \
  single_kaon_RTD.root
```

This produced:

```text
/Users/fb_local/Programs/qpixrtd/single_kaon_RTD.root
```

The file size after the test run was about:

```text
16 MB
```

Verify:

```sh
ls -lh /Users/fb_local/Programs/qpixrtd/single_kaon_RTD.root
```

The RTD run printed the liquid argon and QPix response parameters used:

```text
W value                       = 23.6 [eV]
Drift velocity                = 164800 [cm/s]
Longitidunal diffusion        = 6.8223 [cm^2/s]
Transverse diffusion          = 13.1586 [cm^2/s]
Electron life time            = 0.1 [s]
Readout dimensions            = 100 [cm]
Pixel size                    = 0.4 [cm]
Reset threshold               = 6250 [electrons]
Sample time                   = 1e-08 [s]
Buffer window                 = 0.01 [s]
Dead time                     = 0 [s]
Downsampling (1 = full stats) = 1
Charge loss                   = NO [yes/no]
Recombination                 = YES [yes/no]
Noise                         = YES [yes/no]
TimeWindow                    = YES [yes/no]
```

These are the detector-response settings used to turn Geant4 energy deposits into QPix resets.

## Step 3: Visualize With qpixar

For event-by-event visualization:

```sh
source /Users/fb_local/Programs/qpix-root630-pythia6-env.sh
cd /Users/fb_local/Programs/qpixar/examples
python qpix_ed.py /Users/fb_local/Programs/qpixrtd/single_kaon_RTD.root --3d
```

For one combined view of all events:

```sh
python qpix_ed.py /Users/fb_local/Programs/qpixrtd/single_kaon_RTD.root --3d --all-events
```

The local `--all-events` mode was added to avoid opening one popup per event. It accumulates all event content and displays it in a single window.

## What To Inspect

### In The Geant4 ROOT File

Use the Geant4 output:

```text
/Users/fb_local/Programs/Q_PIX_GEANT4/output/single_kaon.root
```

Useful branches include:

```text
generator_initial_particle_pdg_code
generator_initial_particle_energy
generator_initial_particle_px
generator_initial_particle_py
generator_initial_particle_pz
particle_track_id
particle_parent_track_id
particle_pdg_code
particle_initial_energy
particle_initial_px
particle_initial_py
particle_initial_pz
hit_energy_deposit
hit_start_x
hit_start_y
hit_start_z
hit_end_x
hit_end_y
hit_end_z
```

For a kaon injection, look for:

- initial `kaon+`, PDG code `321`,
- daughter particles from kaon decay or interaction,
- charged daughters such as `mu+` or `pi+`,
- parent-child relationships through `particle_parent_track_id`,
- hit clusters associated with the kaon and daughter tracks.

### In The RTD ROOT File

Use the RTD output:

```text
/Users/fb_local/Programs/qpixrtd/single_kaon_RTD.root
```

Useful branches include:

```text
pixel_x
pixel_y
pixel_reset
pixel_tslr
pixel_reset_truth_track_id
pixel_reset_truth_weight
```

These branches connect QPix resets back to the Geant4 truth tracks.

This is where the charge-response question lives:

```text
Did the kaon and/or daughters make reset patterns that can be separated,
tracked, and matched back to truth?
```

## Recommended Debugging Strategy

Start with a small number of events:

```text
/run/beamOn 5
```

Then run:

```sh
./build/app/G4_QPIX macros/single_kaon.mac
```

and:

```sh
./EXAMPLE/build/EXAMPLE \
  /Users/fb_local/Programs/Q_PIX_GEANT4/output/single_kaon.root \
  single_kaon_RTD.root
```

Inspect a few events with:

```sh
python qpix_ed.py /Users/fb_local/Programs/qpixrtd/single_kaon_RTD.root --3d
```

After the event topology looks reasonable, increase to:

```text
/run/beamOn 100
```

or more.

## When To Move To GENIE

Move to GENIE after the following are understood:

1. A single `kaon+` can be injected and tracked in Geant4.
2. Q_PIX_RTD can convert the kaon event into reset response.
3. qpixar can visualize the kaon and its daughters.
4. The truth branches can be used to identify kaon/daughter kinematics.
5. The charge response can be connected back to the correct truth tracks.

At that point, GENIE can be used to generate a realistic proton-decay event rather than a hand-injected kaon.

## Summary Of Commands Used

Run Geant4:

```sh
source /Users/fb_local/Programs/qpix-root630-pythia6-env.sh
cd /Users/fb_local/Programs/Q_PIX_GEANT4
./build/app/G4_QPIX macros/single_kaon.mac
```

Run RTD:

```sh
source /Users/fb_local/Programs/qpix-root630-pythia6-env.sh
cd /Users/fb_local/Programs/qpixrtd
./EXAMPLE/build/EXAMPLE \
  /Users/fb_local/Programs/Q_PIX_GEANT4/output/single_kaon.root \
  single_kaon_RTD.root
```

Visualize:

```sh
source /Users/fb_local/Programs/qpix-root630-pythia6-env.sh
cd /Users/fb_local/Programs/qpixar/examples
python qpix_ed.py /Users/fb_local/Programs/qpixrtd/single_kaon_RTD.root --3d
```

Visualize all events in one display:

```sh
python qpix_ed.py /Users/fb_local/Programs/qpixrtd/single_kaon_RTD.root --3d --all-events
```


# Reset-based dE/dx versus residual range

`dEdx_vs_residual_range_resets.py` makes a first, truth-assisted reconstruction
of particle dE/dx from QPix reset signals. It deliberately works with individual
reset points and does not voxelize the reset cloud.

## Inputs

Pass the generated event directory containing:

- `resets_output.txt`
- `particles_output.txt`

The script also reads detector-response metadata from the matching RTD ROOT file
beside the event directory. Use `--metadata-root` if it cannot be found
automatically.

```bash
G4_QPIX_analysis/analysis_venv/bin/python \
  G4_QPIX_analysis/dEdx_vs_residual_range_resets.py \
  qpixrtd_output/proton_decay_argon_1k_events_RTD_2026-09-03_153336
```

The default path-length bin width is 1 cm. It can be changed with
`--bin-width-cm`.

## Reconstruction algorithm

### 1. Convert each reset to a 3D point

The center of the readout pixel supplies x and y. Reset arrival time supplies
the drift coordinate:

```text
x = (pixel_x + 0.5) * pixel_size
y = (pixel_y + 0.5) * pixel_size
z = reset_time * drift_velocity
```

This assumes the simulation event time is the drift-time reference and uses the
same coordinate convention as the generated sample.

### 2. Assign reset charge to simulated tracks

Each reset contains parallel `MC_TrackIDs` and `MC_Weights` lists. A reset is
expanded into one contribution per listed track, and the weight is interpreted
as the number of electrons contributed by that track. Mixed resets are therefore
split fractionally rather than assigned entirely to one particle.

Rows whose track-ID lists were shortened with `...` when the text file was
written are omitted because their IDs can no longer be aligned safely with the
unshortened weight list. There are 63 such rows in the current 1000-event sample.

`particles_output.txt` maps `(event, track ID)` to PDG code and parent track ID.
A primary kaon is selected with:

```text
abs(PDG) == 321 and parent_track_id == 0
```

This is truth-assisted particle identification. The charge and coordinates come
from resets, but this first version does not attempt detector-only particle or
track separation.

### 3. Reconstruct a simple track coordinate

For each particle track, the script fits a charge-weighted principal axis to its
individual 3D reset points. Every point is projected onto that axis. The axis is
oriented from the particle's initial position toward its final position using
`particles_output.txt`.

The projected coordinate is used as path length. Residual range is measured from
the final end:

```text
residual range = fitted track length - projected bin midpoint
```

This PCA model is intentionally simple. It works best for approximately straight
tracks. Curved tracks, hard scatters, branches, and showers can have their path
length underestimated. A later reconstruction can replace this step with a
neighbor graph, minimum-spanning-tree backbone, or spline without changing the
charge calibration.

### 4. Convert reset charge to dE/dx

Electron weights are summed in fixed bins along the fitted track. The collected
charge-equivalent energy density is:

```text
dQ/dx = electrons_in_bin * 23.6e-6 MeV / bin_path_length_cm
```

Edge bins are normalized to the requested bin width (or the full fitted length
for tracks shorter than one bin). This prevents an endpoint reset from being
divided by an arbitrarily tiny final sliver of the fitted axis.

When the RTD metadata says recombination was enabled, the script inverts the
same modified-box model used by QPix RTD:

```text
R = ln(A + B*dE/dx) / (B*dE/dx)
A = 0.930
B = 0.212 / 0.5

dE/dx = (exp(B*dQ/dx) - A) / B
```

If simulated charge loss is enabled, each contribution is first corrected using
the configured electron lifetime. This sample has charge loss disabled.

### 5. Produce the plots

The script writes:

- a primary-kaon-only reset reconstruction with the kaon Bethe-Bloch curve;
- a primary-kaon view capped at 10 cm residual range and 40 MeV/cm;
- an all-particle reset reconstruction, colored by the most common PDG species;
- a CSV containing every reconstructed 1 cm segment;
- a text summary of the input and reconstruction settings.

Outputs are placed under:

```text
G4_QPIX_analysis/dEdx_resets/kaon_decay_<number_events>_events/
```

## Interpretation and current limitations

This is a closure-test reconstruction, not a fully blind detector
reconstruction. The largest limitations are:

- track identity and mixed-reset charge fractions come from MC truth;
- the PCA path assumes each track is approximately straight;
- absolute drift position requires a known event time;
- completed resets do not include charge left below threshold at readout end;
- reset timing, diffusion, threshold quantization, and overlapping tracks broaden
  the reconstructed curve;
- short tracks with less than half a path-length bin of projected extent are
  omitted.

The most useful validation is to compare the generated segment CSV with the G4
primary-kaon segment CSV event by event. That separates charge-calibration bias
from path-reconstruction bias.

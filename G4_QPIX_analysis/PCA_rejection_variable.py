#!/usr/bin/env python3

"""Evaluate an energy-weighted PCA electron-shower rejection variable."""

# Import argument parsing so all detector and robustness assumptions are configurable.
import argparse
# Import operating-system helpers so matplotlib can use a writable cache directory.
import os
# Import filesystem paths so input and output locations are explicit and portable.
from pathlib import Path

# Import NumPy for covariance matrices, eigenvalues, random variations, and statistics.
import numpy as np
# Import pandas for chunked truth-table input and tabular output files.
import pandas as pd


# Resolve the directory containing this analysis script.
SCRIPT_DIR = Path(__file__).resolve().parent
# Resolve the repository root from the script location.
REPO_ROOT = SCRIPT_DIR.parent
# Keep matplotlib cache files inside the analysis area when the home cache is unavailable.
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

# Import matplotlib only after setting its writable cache directory.
import matplotlib

# Select a noninteractive backend so plots can be produced in batch jobs.
matplotlib.use("Agg")
# Import pyplot for all histogram, ROC, scatter, and robustness figures.
import matplotlib.pyplot as plt

# Reuse the tested truth-table and ancestry readers from the other-particle PCA analysis.
from PCA_other_particles_track import (
    build_electron_shower_membership,
    read_g4_steps,
    read_particle_table,
)
# Reuse the tested G4 midpoint and path-length construction from the primary-kaon analysis.
from PCA_kaon_track import add_step_geometry


# Point to the requested 1000-event proton-decay RTD truth directory by default.
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "qpixrtd_output"
    / "proton_decay_argon_1k_events_RTD_2026-09-03_153336"
)
# Save every artifact from this study under one new analysis hierarchy.
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "PCA_rejection_variable"
# Record the primary K+ PDG code once so all selections use the same identifier.
KAON_PLUS_PDG = 321
# Record the electron PDG code used to identify shower roots.
ELECTRON_PDG = 11
# Define the requested deposited-energy intervals in MeV.
ENERGY_BINS_MEV = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0), (100.0, 200.0)]
# Define the signal efficiencies at which working points will be reported.
TARGET_KAON_EFFICIENCIES = (0.90, 0.95, 0.99)
# Convert the prior f_T = 1-PC1 threshold into R_T = f_T/(1-f_T).
PRELIMINARY_FT_THRESHOLD = 0.00242
# Store the converted preliminary threshold for direct comparison in this analysis.
PRELIMINARY_RT_THRESHOLD = PRELIMINARY_FT_THRESHOLD / (1.0 - PRELIMINARY_FT_THRESHOLD)


def object_seed(global_seed, event_id, object_id, scenario_index):
    """Build a reproducible random seed for one object and one detector scenario."""
    # Combine integer identifiers with large coprime multipliers for stable object-level randomness.
    return int(
        (
            int(global_seed)
            + 1_000_003 * int(event_id)
            + 9_176 * int(object_id)
            + 104_729 * int(scenario_index)
        )
        % (2**32 - 1)
    )


def energy_weighted_eigenvalues(points, energies):
    """Return descending eigenvalues of the deposited-energy-weighted 3D covariance."""
    # Convert coordinates to a dense floating-point array.
    points = np.asarray(points, dtype=float)
    # Convert deposited energies to a dense floating-point array.
    energies = np.asarray(energies, dtype=float)
    # Reject objects that do not have enough spatial samples for a stable covariance estimate.
    if len(points) < 3:
        return None
    # Sum deposited energy so the PCA weights can be normalized.
    total_energy = float(np.sum(energies))
    # Reject empty or nonphysical charge objects.
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        return None
    # Normalize each step energy into a PCA weight.
    weights = energies / total_energy
    # Calculate the deposited-energy-weighted center of the object.
    center = np.average(points, axis=0, weights=weights)
    # Shift all coordinates to that weighted center.
    shifted = points - center
    # Construct the symmetric energy-weighted covariance matrix.
    covariance = (shifted * weights[:, np.newaxis]).T @ shifted
    # Diagonalize the covariance matrix using the symmetric-matrix eigensolver.
    eigenvalues = np.linalg.eigvalsh(covariance)
    # Sort eigenvalues from longitudinal to smallest transverse spread.
    eigenvalues = np.sort(np.clip(eigenvalues, 0.0, None))[::-1]
    # Reject zero-size objects whose longitudinal eigenvalue cannot define R_T.
    if eigenvalues[0] <= np.finfo(float).eps:
        return None
    # Return lambda_1, lambda_2, and lambda_3 in descending order.
    return eigenvalues


def calculate_score(points, energies):
    """Calculate eigenvalues, R_T, and log10(R_T) for one detector object."""
    # Calculate the ordered energy-weighted covariance eigenvalues.
    eigenvalues = energy_weighted_eigenvalues(points, energies)
    # Return no score when the covariance is undefined.
    if eigenvalues is None:
        return None
    # Unpack the longitudinal and two transverse eigenvalues.
    lambda_1, lambda_2, lambda_3 = eigenvalues
    # Form the transverse-to-longitudinal shower score requested by the user.
    transverse_ratio = float((lambda_2 + lambda_3) / lambda_1)
    # Protect the logarithm from exact numerical zeros for perfectly straight tracks.
    safe_ratio = max(transverse_ratio, np.finfo(float).tiny)
    # Transform the ratio to log10 space for readable distributions over many decades.
    log10_ratio = float(np.log10(safe_ratio))
    # Return all values needed for output tables and diagnostic plots.
    return {
        "lambda_1_cm2": float(lambda_1),
        "lambda_2_cm2": float(lambda_2),
        "lambda_3_cm2": float(lambda_3),
        "R_T": transverse_ratio,
        "log10_R_T": log10_ratio,
    }


def prepare_object_arrays(steps):
    """Construct immutable midpoint, energy, and track-ID arrays once per truth object."""
    # Construct midpoint and path-length columns once before testing any detector scenario.
    prepared = add_step_geometry(steps)
    # Copy midpoint coordinates into a writable, scenario-independent array.
    points = prepared[["x_cm", "y_cm", "z_cm"]].to_numpy(dtype=float, copy=True)
    # Copy deposited energies into a writable, scenario-independent array.
    energies = prepared["E"].to_numpy(dtype=float, copy=True)
    # Copy particle track IDs so clustering-loss masks can operate without pandas overhead.
    track_ids = prepared["ParticleID"].to_numpy(dtype=int, copy=True)
    # Return all immutable base arrays for reuse by every scenario.
    return points, energies, track_ids


def apply_detector_scenario(prepared, scenario, args, event_id, object_id, root_track_id):
    """Apply one deterministic detector or clustering variation to cached object arrays."""
    # Unpack the geometry calculated once for this truth object.
    base_points, base_energies, base_track_ids = prepared
    # Stop when geometry filtering leaves fewer points than required by the analysis.
    if len(base_points) < args.min_points:
        return None
    # Create a deterministic random generator unique to this object and scenario.
    rng = np.random.default_rng(
        object_seed(args.random_seed, event_id, object_id, scenario["index"])
    )

    # Start with every prepared point retained.
    keep = np.ones(len(base_points), dtype=bool)

    # Remove a configurable fraction of steps to emulate missing collected charge.
    if scenario["missing_charge_fraction"] > 0.0:
        # Draw one keep decision for each G4 step.
        keep &= rng.random(len(base_points)) >= scenario["missing_charge_fraction"]

    # Remove descendant branches to emulate incomplete electromagnetic-shower clustering.
    if scenario["branch_loss_fraction"] > 0.0:
        # Identify all daughter tracks other than the initiating electron track.
        daughter_ids = np.array(
            [track for track in np.unique(base_track_ids) if int(track) != int(root_track_id)]
        )
        # Randomly choose which daughter tracks remain attached to the shower cluster.
        keep_daughters = daughter_ids[
            rng.random(len(daughter_ids)) >= scenario["branch_loss_fraction"]
        ]
        # Always retain the initiating electron and retain only selected daughter branches.
        branch_keep = (base_track_ids == int(root_track_id)) | np.isin(
            base_track_ids,
            keep_daughters,
        )
        # Combine the imperfect-clustering mask with any prior charge-loss mask.
        keep &= branch_keep

    # Stop when detector losses leave too few spatial samples for the requested PCA quality cut.
    if int(np.count_nonzero(keep)) < args.min_points:
        return None

    # Copy retained midpoint coordinates into a mutable scenario-specific array.
    points = base_points[keep].copy()
    # Copy retained deposited energies into a mutable scenario-specific array.
    energies = base_energies[keep].copy()

    # Pixelize x and y at the documented Q-Pix pixel pitch while retaining truth drift z.
    if scenario["pixelize_xy"]:
        # Snap x coordinates to the center of the nearest pitch interval.
        points[:, 0] = np.round(points[:, 0] / args.pixel_pitch_cm) * args.pixel_pitch_cm
        # Snap y coordinates to the center of the nearest pitch interval.
        points[:, 1] = np.round(points[:, 1] / args.pixel_pitch_cm) * args.pixel_pitch_cm

    # Add configurable Gaussian position smearing to all three coordinates.
    if scenario["position_sigma_cm"] > 0.0:
        # Generate and add independent coordinate-resolution fluctuations.
        points += rng.normal(0.0, scenario["position_sigma_cm"], size=points.shape)

    # Add configurable fractional Gaussian charge noise to every deposited-energy weight.
    if scenario["energy_noise_fraction"] > 0.0:
        # Generate multiplicative charge-response fluctuations around unity.
        scale = rng.normal(1.0, scenario["energy_noise_fraction"], size=len(energies))
        # Clip fluctuated energies at zero because negative deposited charge is unphysical.
        energies = np.clip(energies * scale, 0.0, None)

    # Calculate the PCA score after applying this detector scenario.
    score = calculate_score(points, energies)
    # Stop when the varied object no longer produces a valid covariance.
    if score is None:
        return None
    # Store the number of points that survived this detector scenario.
    score["n_points"] = int(len(points))
    # Store the varied total deposited energy used by the weighted PCA.
    score["deposited_energy_MeV"] = float(np.sum(energies))
    # Return the complete scenario result.
    return score


def detector_scenarios(args):
    """Define nominal and robustness scenarios in one auditable table."""
    # Return scenario dictionaries so each detector effect is independently reproducible.
    return [
        {
            "name": "nominal_truth",
            "index": 0,
            "pixelize_xy": False,
            "position_sigma_cm": 0.0,
            "missing_charge_fraction": 0.0,
            "energy_noise_fraction": 0.0,
            "branch_loss_fraction": 0.0,
        },
        {
            "name": "qpix_xy_pixelized",
            "index": 1,
            "pixelize_xy": True,
            "position_sigma_cm": 0.0,
            "missing_charge_fraction": 0.0,
            "energy_noise_fraction": 0.0,
            "branch_loss_fraction": 0.0,
        },
        {
            "name": "missing_charge_10pct",
            "index": 2,
            "pixelize_xy": False,
            "position_sigma_cm": 0.0,
            "missing_charge_fraction": args.missing_charge_fraction,
            "energy_noise_fraction": 0.0,
            "branch_loss_fraction": 0.0,
        },
        {
            "name": "charge_noise_10pct",
            "index": 3,
            "pixelize_xy": False,
            "position_sigma_cm": 0.0,
            "missing_charge_fraction": 0.0,
            "energy_noise_fraction": args.energy_noise_fraction,
            "branch_loss_fraction": 0.0,
        },
        {
            "name": "position_smearing",
            "index": 4,
            "pixelize_xy": False,
            "position_sigma_cm": args.position_sigma_cm,
            "missing_charge_fraction": 0.0,
            "energy_noise_fraction": 0.0,
            "branch_loss_fraction": 0.0,
        },
        {
            "name": "electron_branch_loss_10pct",
            "index": 5,
            "pixelize_xy": False,
            "position_sigma_cm": 0.0,
            "missing_charge_fraction": 0.0,
            "energy_noise_fraction": 0.0,
            "branch_loss_fraction": args.branch_loss_fraction,
        },
        {
            "name": "combined_detector_effects",
            "index": 6,
            "pixelize_xy": True,
            "position_sigma_cm": args.position_sigma_cm,
            "missing_charge_fraction": args.missing_charge_fraction,
            "energy_noise_fraction": args.energy_noise_fraction,
            "branch_loss_fraction": args.branch_loss_fraction,
        },
    ]


def build_primary_kaon_keys(particles):
    """Return event/track keys for only proton-decay primary K+ particles."""
    # Select positively charged kaons whose Geant4 parent track is zero.
    primary_kaons = particles[
        particles["particle_pdg_code"].eq(KAON_PLUS_PDG)
        & particles["particle_parent_track_id"].eq(0)
    ].copy()
    # Return exact event/track tuples because Geant4 track IDs restart in every event.
    return set(
        zip(
            primary_kaons["event"].astype(int),
            primary_kaons["particle_track_id"].astype(int),
        )
    )


def analyze_objects(particles, g4_steps, args):
    """Calculate R_T for primary K+ tracks and complete electron showers."""
    # Build the complete list of detector-response scenarios.
    scenarios = detector_scenarios(args)
    # Build event/track keys for the primary proton-decay K+ signal definition.
    primary_kaon_keys = build_primary_kaon_keys(particles)
    # Prepare one output row per object and detector scenario.
    rows = []
    # Prepare counters for objects rejected by point-count or covariance requirements.
    skipped = {scenario["name"]: {"primary_k_plus": 0, "electron_shower": 0} for scenario in scenarios}

    # Select K+ G4 steps before grouping by event and track ID.
    kaon_steps = g4_steps[g4_steps["PDG"].eq(KAON_PLUS_PDG)].copy()
    # Group K+ deposits into individual truth tracks.
    kaon_groups = kaon_steps.groupby(["event", "ParticleID"], sort=True)
    # Report the number of K+ groups before the primary-parent selection.
    print(f"Scanning {kaon_groups.ngroups:,} K+ G4 track group(s) ...")
    # Loop over every K+ truth track.
    for (event_id, track_id), steps in kaon_groups:
        # Skip secondary K+ tracks because the signal definition is the primary proton-decay K+.
        if (int(event_id), int(track_id)) not in primary_kaon_keys:
            continue
        # Construct midpoint, energy, and track-ID arrays once for all detector scenarios.
        prepared = prepare_object_arrays(steps)
        # Evaluate every detector scenario on exactly the same truth object.
        for scenario in scenarios:
            # Apply detector effects and calculate the PCA shower score.
            result = apply_detector_scenario(
                prepared,
                scenario,
                args,
                int(event_id),
                int(track_id),
                int(track_id),
            )
            # Count and skip objects that fail this scenario's PCA quality requirements.
            if result is None:
                skipped[scenario["name"]]["primary_k_plus"] += 1
                continue
            # Add identifiers and the scenario result to the long-format output table.
            rows.append(
                {
                    "event": int(event_id),
                    "object_id": int(track_id),
                    "object_type": "primary_k_plus",
                    "scenario": scenario["name"],
                    **result,
                }
            )

    # Select all deposits assigned to an earliest-electron shower root.
    shower_steps = g4_steps[g4_steps["electron_shower_root_track_id"].notna()].copy()
    # Convert the root identifier from nullable float storage back to an exact integer.
    shower_steps["electron_shower_root_track_id"] = shower_steps[
        "electron_shower_root_track_id"
    ].astype(int)
    # Group all recursive descendants into one non-overlapping object per initiating electron.
    shower_groups = shower_steps.groupby(
        ["event", "electron_shower_root_track_id"],
        sort=True,
    )
    # Report the number of complete electron-shower candidates.
    print(f"Scanning {shower_groups.ngroups:,} complete electron shower group(s) ...")
    # Loop over every complete truth-defined electron shower.
    for shower_number, ((event_id, root_track_id), steps) in enumerate(
        shower_groups,
        start=1,
    ):
        # Print occasional progress for the large electron-shower population.
        if shower_number % 50_000 == 0:
            print(f"  Processed {shower_number:,} / {shower_groups.ngroups:,} showers")
        # Construct midpoint, energy, and track-ID arrays once for all detector scenarios.
        prepared = prepare_object_arrays(steps)
        # Evaluate every detector scenario on exactly the same complete shower.
        for scenario in scenarios:
            # Apply detector and clustering effects before calculating the PCA score.
            result = apply_detector_scenario(
                prepared,
                scenario,
                args,
                int(event_id),
                int(root_track_id),
                int(root_track_id),
            )
            # Count and skip showers that fail this scenario's PCA quality requirements.
            if result is None:
                skipped[scenario["name"]]["electron_shower"] += 1
                continue
            # Add identifiers and the scenario result to the long-format output table.
            rows.append(
                {
                    "event": int(event_id),
                    "object_id": int(root_track_id),
                    "object_type": "electron_shower",
                    "scenario": scenario["name"],
                    **result,
                }
            )

    # Convert all object/scenario dictionaries into one analysis table.
    scores = pd.DataFrame(rows)
    # Sort deterministically for reproducible CSV output and plotting.
    scores = scores.sort_values(
        ["scenario", "object_type", "event", "object_id"]
    ).reset_index(drop=True)
    # Return scores, skipped-object counts, and the exact scenario definitions.
    return scores, skipped, scenarios


def split_events(scores, validation_fraction, random_seed):
    """Assign complete events to train or validation without object leakage."""
    # Extract sorted unique event IDs from the nominal object table.
    events = np.sort(scores["event"].unique())
    # Create a deterministic generator for the event-level split.
    rng = np.random.default_rng(random_seed)
    # Shuffle a copy so event assignment is random but reproducible.
    shuffled = events.copy()
    # Apply the deterministic event shuffle.
    rng.shuffle(shuffled)
    # Calculate the number of events reserved for validation.
    n_validation = max(1, int(round(validation_fraction * len(shuffled))))
    # Store validation event IDs in a set for fast membership testing.
    validation_events = set(shuffled[:n_validation])
    # Label every score row according to its event-level partition.
    scores = scores.copy()
    # Use the same event assignment for every scenario and every object in that event.
    scores["split"] = np.where(
        scores["event"].isin(validation_events),
        "validation",
        "train",
    )
    # Return the labeled table and the exact held-out event IDs.
    return scores, sorted(validation_events)


def auc_from_scores(kaon_scores, electron_scores):
    """Compute the rank AUC for lower R_T being more kaon-like."""
    # Convert R_T into a classifier score where larger means more signal-like.
    signal_like_kaons = -np.asarray(kaon_scores, dtype=float)
    # Apply the same sign convention to electron showers.
    signal_like_electrons = -np.asarray(electron_scores, dtype=float)
    # Concatenate the two classes for a tie-aware rank calculation.
    combined = np.concatenate([signal_like_kaons, signal_like_electrons])
    # Use average ranks so identical scores receive correct half-credit.
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=float)
    # Count signal and background objects.
    n_signal = len(signal_like_kaons)
    # Count electron-shower background objects.
    n_background = len(signal_like_electrons)
    # Return NaN when either class is absent.
    if n_signal == 0 or n_background == 0:
        return np.nan
    # Sum ranks assigned to K+ objects.
    signal_rank_sum = float(np.sum(ranks[:n_signal]))
    # Convert the rank sum into the Mann-Whitney probability interpretation of AUC.
    return (
        signal_rank_sum - n_signal * (n_signal + 1) / 2.0
    ) / (n_signal * n_background)


def roc_curve(kaon_scores, electron_scores, n_thresholds=600):
    """Scan log-spaced R_T thresholds and return efficiency and rejection arrays."""
    # Convert both classes to finite NumPy arrays.
    kaon_scores = np.asarray(kaon_scores, dtype=float)
    # Convert electron-shower scores to a finite NumPy array.
    electron_scores = np.asarray(electron_scores, dtype=float)
    # Determine the full finite score range across both classes.
    all_scores = np.concatenate([kaon_scores, electron_scores])
    # Replace exact zeros from perfectly collinear objects before taking logarithms.
    all_scores = np.clip(all_scores, np.finfo(float).tiny, None)
    # Build evenly spaced thresholds in log10(R_T), including endpoint padding.
    log_thresholds = np.linspace(
        np.min(np.log10(all_scores)) - 0.05,
        np.max(np.log10(all_scores)) + 0.05,
        n_thresholds,
    )
    # Convert logarithmic thresholds back to R_T.
    thresholds = 10.0**log_thresholds
    # Accept K+ candidates below each track-like threshold.
    kaon_efficiency = np.array([np.mean(kaon_scores < threshold) for threshold in thresholds])
    # Reject electron showers at or above each threshold.
    electron_rejection = np.array(
        [np.mean(electron_scores >= threshold) for threshold in thresholds]
    )
    # Return thresholds and both operating characteristics.
    return thresholds, kaon_efficiency, electron_rejection


def derive_working_points(scores):
    """Learn thresholds on train kaons and evaluate them on held-out events."""
    # Prepare one row per scenario and requested signal-efficiency target.
    rows = []
    # Process each detector scenario independently.
    for scenario, scenario_data in scores.groupby("scenario", sort=True):
        # Select training K+ scores used to choose thresholds.
        train_kaons = scenario_data[
            scenario_data["split"].eq("train")
            & scenario_data["object_type"].eq("primary_k_plus")
        ]["R_T"].to_numpy(dtype=float)
        # Select validation K+ scores used only for held-out performance reporting.
        validation_kaons = scenario_data[
            scenario_data["split"].eq("validation")
            & scenario_data["object_type"].eq("primary_k_plus")
        ]["R_T"].to_numpy(dtype=float)
        # Select validation electron showers used only for held-out rejection reporting.
        validation_electrons = scenario_data[
            scenario_data["split"].eq("validation")
            & scenario_data["object_type"].eq("electron_shower")
        ]["R_T"].to_numpy(dtype=float)
        # Calculate a held-out rank AUC for this detector scenario.
        validation_auc = auc_from_scores(validation_kaons, validation_electrons)

        # Derive one cut for every requested training K+ efficiency.
        for target_efficiency in TARGET_KAON_EFFICIENCIES:
            # Choose the training-K+ quantile that accepts the requested fraction below threshold.
            threshold = float(np.quantile(train_kaons, target_efficiency))
            # Measure the actual held-out K+ efficiency at the learned threshold.
            validation_efficiency = float(np.mean(validation_kaons < threshold))
            # Measure the held-out electron-shower rejection at the learned threshold.
            validation_rejection = float(np.mean(validation_electrons >= threshold))
            # Save the complete working-point definition and validation result.
            rows.append(
                {
                    "scenario": scenario,
                    "target_train_kaon_efficiency": target_efficiency,
                    "R_T_threshold": threshold,
                    "log10_R_T_threshold": float(np.log10(threshold)),
                    "validation_kaon_efficiency": validation_efficiency,
                    "validation_electron_rejection": validation_rejection,
                    "validation_auc": validation_auc,
                    "n_train_kaons": len(train_kaons),
                    "n_validation_kaons": len(validation_kaons),
                    "n_validation_electron_showers": len(validation_electrons),
                }
            )
    # Return all scenario working points as a tidy table.
    return pd.DataFrame(rows)


def plot_nominal_score_histogram(scores, output_path):
    """Plot normalized nominal log10(R_T) distributions for K+ and electron showers."""
    # Select held-out nominal scores so the main comparison is validation-only.
    data = scores[
        scores["scenario"].eq("nominal_truth") & scores["split"].eq("validation")
    ]
    # Define fixed-width 0.1-decade bins over the displayed score range.
    bins = np.arange(-8.0, 2.0001, 0.1)
    # Create the normalized comparison figure.
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    # Plot the primary-K+ probability density.
    ax.hist(
        data.loc[data["object_type"].eq("primary_k_plus"), "log10_R_T"],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        label="Primary K+",
    )
    # Plot the complete electron-shower probability density.
    ax.hist(
        data.loc[data["object_type"].eq("electron_shower"), "log10_R_T"],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        label="Electron showers",
    )
    # Label the horizontal shower-score axis.
    ax.set_xlabel(r"$\log_{10}(R_T)$, $R_T=(\lambda_2+\lambda_3)/\lambda_1$")
    # Label the normalized vertical axis.
    ax.set_ylabel("Probability density")
    # State that this figure uses held-out nominal objects.
    ax.set_title("Validation PCA shower-score distributions")
    # Add a light grid for comparing distribution shapes.
    ax.grid(True, axis="y", alpha=0.25)
    # Add the particle-type legend.
    ax.legend()
    # Display the histogram bin width directly in the figure.
    ax.text(
        0.02,
        0.96,
        "Bin width: 0.1 in log10(R_T)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.6"},
    )
    # Prevent labels from clipping outside the canvas.
    fig.tight_layout()
    # Save a high-resolution PNG.
    fig.savefig(output_path, dpi=220)
    # Release figure memory.
    plt.close(fig)


def plot_energy_binned_histograms(scores, output_path):
    """Compare nominal score shapes in the requested deposited-energy intervals."""
    # Select held-out nominal objects for an unbiased energy-binned comparison.
    data = scores[
        scores["scenario"].eq("nominal_truth") & scores["split"].eq("validation")
    ]
    # Define shared fixed-width bins in log10(R_T).
    bins = np.arange(-8.0, 2.0001, 0.1)
    # Create one panel per requested energy interval.
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), sharex=True, sharey=True)
    # Loop over axes and deposited-energy intervals together.
    for ax, (energy_low, energy_high) in zip(axes.flat, ENERGY_BINS_MEV):
        # Select objects whose varied deposited energy falls inside this interval.
        energy_slice = data[
            data["deposited_energy_MeV"].ge(energy_low)
            & data["deposited_energy_MeV"].lt(energy_high)
        ]
        # Select primary K+ scores in this energy interval.
        kaons = energy_slice.loc[
            energy_slice["object_type"].eq("primary_k_plus"), "log10_R_T"
        ]
        # Select complete electron-shower scores in this energy interval.
        electrons = energy_slice.loc[
            energy_slice["object_type"].eq("electron_shower"), "log10_R_T"
        ]
        # Plot K+ only when the interval contains at least one signal object.
        if len(kaons):
            ax.hist(
                kaons,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=f"Primary K+ (n={len(kaons)})",
            )
        # Plot electron showers only when the interval contains at least one background object.
        if len(electrons):
            ax.hist(
                electrons,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=f"Electron showers (n={len(electrons)})",
            )
        # Label the deposited-energy interval in the panel title.
        ax.set_title(f"{energy_low:g} <= Edep < {energy_high:g} MeV")
        # Add a light vertical-density grid.
        ax.grid(True, axis="y", alpha=0.25)
        # Add a compact panel legend.
        ax.legend(fontsize=8)
    # Label the shared horizontal axis on the bottom panels.
    axes[1, 0].set_xlabel(r"$\log_{10}(R_T)$")
    # Label the shared horizontal axis on the second bottom panel.
    axes[1, 1].set_xlabel(r"$\log_{10}(R_T)$")
    # Label the shared probability-density axis on the left panels.
    axes[0, 0].set_ylabel("Probability density")
    # Label the shared probability-density axis on the second left panel.
    axes[1, 0].set_ylabel("Probability density")
    # Add a figure-level title explaining the energy control.
    fig.suptitle("Validation PCA shower score in deposited-energy bins")
    # State the shared histogram bin width.
    fig.text(0.5, 0.01, "Histogram bin width: 0.1 in log10(R_T)", ha="center")
    # Leave room for the title and footer while preventing clipping.
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.96])
    # Save the four-panel figure.
    fig.savefig(output_path, dpi=220)
    # Release figure memory.
    plt.close(fig)


def plot_nominal_roc(scores, working_points, output_path):
    """Plot held-out electron rejection versus K+ efficiency for nominal truth."""
    # Select held-out nominal objects only.
    data = scores[
        scores["scenario"].eq("nominal_truth") & scores["split"].eq("validation")
    ]
    # Extract primary-K+ R_T scores.
    kaons = data.loc[data["object_type"].eq("primary_k_plus"), "R_T"].to_numpy()
    # Extract complete electron-shower R_T scores.
    electrons = data.loc[data["object_type"].eq("electron_shower"), "R_T"].to_numpy()
    # Scan the full nominal threshold range.
    _, efficiency, rejection = roc_curve(kaons, electrons)
    # Calculate the conventional rank AUC.
    auc = auc_from_scores(kaons, electrons)
    # Create the ROC-style rejection figure.
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    # Draw electron rejection as a function of retained K+ efficiency.
    ax.plot(efficiency, rejection, linewidth=2.0, label=f"Validation AUC = {auc:.4f}")
    # Select nominal learned working points.
    nominal_points = working_points[working_points["scenario"].eq("nominal_truth")]
    # Draw each requested operating point with its performance in the legend.
    for point in nominal_points.itertuples(index=False):
        # Mark this held-out operating point without overlapping plot annotations.
        ax.scatter(
            [point.validation_kaon_efficiency],
            [point.validation_electron_rejection],
            s=55,
            zorder=3,
            label=(
                f"{100.0 * point.target_train_kaon_efficiency:.0f}% target: "
                f"K+ {100.0 * point.validation_kaon_efficiency:.1f}%, "
                f"e- rejection {100.0 * point.validation_electron_rejection:.1f}%"
            ),
        )
    # Label the signal-efficiency axis.
    ax.set_xlabel("Primary K+ efficiency")
    # Label the background-rejection axis.
    ax.set_ylabel("Electron-shower rejection")
    # State that thresholds are evaluated on held-out events.
    ax.set_title("PCA electron-shower rejection on validation events")
    # Limit both probability axes to their physical range.
    ax.set_xlim(0.0, 1.0)
    # Limit the rejection axis to its physical range.
    ax.set_ylim(0.0, 1.0)
    # Add a readability grid.
    ax.grid(True, alpha=0.25)
    # Add the AUC and working-point legend.
    ax.legend(loc="lower left")
    # Prevent labels from clipping.
    fig.tight_layout()
    # Save the ROC figure.
    fig.savefig(output_path, dpi=220)
    # Release figure memory.
    plt.close(fig)


def plot_robustness_roc(scores, output_path):
    """Overlay held-out ROC curves for every detector and clustering scenario."""
    # Create one comparison figure for all scenarios.
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    # Loop over detector scenarios in sorted name order.
    for scenario, scenario_data in scores[scores["split"].eq("validation")].groupby(
        "scenario",
        sort=True,
    ):
        # Extract held-out K+ scores for this scenario.
        kaons = scenario_data.loc[
            scenario_data["object_type"].eq("primary_k_plus"), "R_T"
        ].to_numpy()
        # Extract held-out electron-shower scores for this scenario.
        electrons = scenario_data.loc[
            scenario_data["object_type"].eq("electron_shower"), "R_T"
        ].to_numpy()
        # Skip incomplete scenarios that contain no objects from one class.
        if len(kaons) == 0 or len(electrons) == 0:
            continue
        # Scan thresholds for this scenario.
        _, efficiency, rejection = roc_curve(kaons, electrons)
        # Calculate this scenario's rank AUC.
        auc = auc_from_scores(kaons, electrons)
        # Draw the scenario curve and include AUC in the legend.
        ax.plot(efficiency, rejection, linewidth=1.5, label=f"{scenario} (AUC={auc:.3f})")
    # Label the signal-efficiency axis.
    ax.set_xlabel("Primary K+ efficiency")
    # Label the background-rejection axis.
    ax.set_ylabel("Electron-shower rejection")
    # Describe the robustness comparison.
    ax.set_title("PCA rejection robustness on validation events")
    # Restrict efficiencies to physical values.
    ax.set_xlim(0.0, 1.0)
    # Restrict rejections to physical values.
    ax.set_ylim(0.0, 1.0)
    # Add a light comparison grid.
    ax.grid(True, alpha=0.25)
    # Add a compact scenario legend.
    ax.legend(fontsize=8, loc="lower left")
    # Prevent labels from clipping.
    fig.tight_layout()
    # Save the robustness figure.
    fig.savefig(output_path, dpi=220)
    # Release figure memory.
    plt.close(fig)


def plot_score_vs_energy(scores, output_path):
    """Show how nominal R_T depends on deposited energy for both object classes."""
    # Select held-out nominal objects only.
    data = scores[
        scores["scenario"].eq("nominal_truth") & scores["split"].eq("validation")
    ]
    # Create one scatter panel per object class.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), sharex=True, sharey=True)
    # Loop over object classes and display labels.
    for ax, object_type, label in zip(
        axes,
        ["primary_k_plus", "electron_shower"],
        ["Primary K+", "Electron showers"],
    ):
        # Select this class from held-out nominal data.
        class_data = data[data["object_type"].eq(object_type)]
        # Plot every held-out object with transparent rasterized markers.
        ax.scatter(
            class_data["deposited_energy_MeV"],
            class_data["log10_R_T"],
            s=7,
            alpha=0.22,
            linewidths=0,
            rasterized=True,
        )
        # Use logarithmic deposited energy because the truth population spans many decades.
        ax.set_xscale("log")
        # Label the class panel.
        ax.set_title(label)
        # Add a light grid.
        ax.grid(True, alpha=0.25)
        # Label deposited energy on each panel.
        ax.set_xlabel("Deposited energy [MeV]")
    # Label the shared shower-score axis.
    axes[0].set_ylabel(r"$\log_{10}(R_T)$")
    # Add a figure-level interpretation title.
    fig.suptitle("Energy dependence of the PCA shower score")
    # Leave room for the title and prevent clipping.
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    # Save the energy-dependence figure.
    fig.savefig(output_path, dpi=220)
    # Release figure memory.
    plt.close(fig)


def write_method_summary(output_path, args, scores, validation_events, scenarios):
    """Write a human-readable definition of objects, score, split, and variations."""
    # Select nominal rows for concise object counts.
    nominal = scores[scores["scenario"].eq("nominal_truth")]
    # Count nominal primary K+ objects.
    n_kaons = int(nominal["object_type"].eq("primary_k_plus").sum())
    # Count nominal complete electron showers.
    n_showers = int(nominal["object_type"].eq("electron_shower").sum())
    # Build an auditable plain-text method description.
    lines = [
        "PCA electron-shower rejection analysis",
        "======================================",
        "",
        f"Input directory: {Path(args.input_dir).resolve()}",
        f"Minimum nonzero-length spatial points: {args.min_points}",
        f"Q-Pix x/y pitch used for pixelization: {args.pixel_pitch_cm:g} cm",
        "Q-Pix caveat: this scenario snaps truth midpoints to x/y pixels; it does not simulate resets or merge charge into reconstructed hits.",
        f"Position-smearing sigma: {args.position_sigma_cm:g} cm",
        f"Missing-charge fraction: {args.missing_charge_fraction:.3g}",
        f"Fractional charge-noise sigma: {args.energy_noise_fraction:.3g}",
        f"Electron descendant-branch loss fraction: {args.branch_loss_fraction:.3g}",
        f"Random seed: {args.random_seed}",
        f"Validation-event fraction: {args.validation_fraction:.3g}",
        f"Validation events: {len(validation_events)}",
        "",
        "Detector objects:",
        "  Primary K+: only deposits from the parent_track_id == 0 K+ before its endpoint.",
        "  Electron shower: initiating electron plus every recursive descendant.",
        "  A secondary electron with an electron ancestor remains in that ancestor's shower.",
        "",
        "PCA definition:",
        "  G4 step midpoints are weighted by deposited energy.",
        "  Eigenvalues are ordered lambda_1 >= lambda_2 >= lambda_3.",
        "  R_T = (lambda_2 + lambda_3) / lambda_1.",
        "  Small R_T is track-like; large R_T is shower-like.",
        "",
        f"Nominal primary K+ objects analyzed: {n_kaons}",
        f"Nominal electron showers analyzed: {n_showers}",
        "",
        "Scenarios:",
    ]
    # Document every robustness scenario by name.
    lines.extend(f"  {scenario['name']}" for scenario in scenarios)
    # Save the complete method summary.
    output_path.write_text("\n".join(lines) + "\n")


def write_working_point_summary(output_path, working_points, scores):
    """Write nominal learned thresholds and the preliminary reference cut."""
    # Select nominal train-derived working points.
    nominal = working_points[working_points["scenario"].eq("nominal_truth")]
    # Select held-out nominal object scores for evaluating the preliminary reference threshold.
    validation = scores[
        scores["scenario"].eq("nominal_truth") & scores["split"].eq("validation")
    ]
    # Extract held-out K+ R_T scores.
    validation_kaons = validation.loc[
        validation["object_type"].eq("primary_k_plus"), "R_T"
    ]
    # Extract held-out electron-shower R_T scores.
    validation_electrons = validation.loc[
        validation["object_type"].eq("electron_shower"), "R_T"
    ]
    # Evaluate the earlier f_T-based threshold after converting it to R_T.
    preliminary_kaon_efficiency = float(
        np.mean(validation_kaons < PRELIMINARY_RT_THRESHOLD)
    )
    # Evaluate electron-shower rejection at that same preliminary threshold.
    preliminary_electron_rejection = float(
        np.mean(validation_electrons >= PRELIMINARY_RT_THRESHOLD)
    )
    # Begin the human-readable operating-point report.
    lines = [
        "PCA rejection working points",
        "============================",
        "",
        "Selection rule: accept a track-like candidate when R_T < threshold.",
        "Thresholds below are learned from training-event primary K+ quantiles.",
        "Performance is evaluated only on held-out validation events.",
        "",
    ]
    # Add each nominal train-derived operating point.
    for point in nominal.itertuples(index=False):
        # Format one compact block per target K+ efficiency.
        lines.extend(
            [
                f"Target training K+ efficiency: {100.0 * point.target_train_kaon_efficiency:.1f}%",
                f"  R_T threshold: {point.R_T_threshold:.8g}",
                f"  log10(R_T) threshold: {point.log10_R_T_threshold:.6g}",
                f"  Validation K+ efficiency: {100.0 * point.validation_kaon_efficiency:.4g}%",
                f"  Validation electron-shower rejection: {100.0 * point.validation_electron_rejection:.4g}%",
                f"  Validation AUC: {point.validation_auc:.6g}",
                "",
            ]
        )
    # Add the explicitly requested preliminary reference working point.
    lines.extend(
        [
            "Preliminary reference from 1-PC1 < 0.00242:",
            f"  Equivalent R_T threshold: {PRELIMINARY_RT_THRESHOLD:.8g}",
            f"  Validation K+ efficiency: {100.0 * preliminary_kaon_efficiency:.4g}%",
            f"  Validation electron-shower rejection: {100.0 * preliminary_electron_rejection:.4g}%",
            "  This is a reference only, not a final detector-level cut.",
        ]
    )
    # Save the working-point report.
    output_path.write_text("\n".join(lines) + "\n")


def write_outputs(scores, skipped, scenarios, validation_events, args):
    """Save tables, figures, method notes, and held-out performance summaries."""
    # Resolve the requested output directory.
    output_dir = Path(args.output_dir).expanduser().resolve()
    # Create the output hierarchy when it does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save every object and scenario score for reproducibility.
    scores.to_csv(output_dir / "pca_rejection_scores.csv", index=False)
    # Derive train-only thresholds and held-out performance measurements.
    working_points = derive_working_points(scores)
    # Save all scenario working points as a machine-readable table.
    working_points.to_csv(output_dir / "pca_rejection_working_points.csv", index=False)
    # Convert skipped counters into a tidy list of rows.
    skipped_rows = [
        {"scenario": scenario, "object_type": object_type, "skipped_objects": count}
        for scenario, object_counts in skipped.items()
        for object_type, count in object_counts.items()
    ]
    # Save PCA quality-cut losses by scenario and object class.
    pd.DataFrame(skipped_rows).to_csv(output_dir / "pca_rejection_skipped_objects.csv", index=False)
    # Save the exact held-out event list for reproducible train/validation separation.
    (output_dir / "validation_event_ids.txt").write_text(
        "\n".join(str(event_id) for event_id in validation_events) + "\n"
    )

    # Save the main normalized nominal score comparison.
    plot_nominal_score_histogram(
        scores,
        output_dir / "log10_RT_normalized_histogram.png",
    )
    # Save score distributions in the four requested deposited-energy intervals.
    plot_energy_binned_histograms(
        scores,
        output_dir / "log10_RT_energy_binned_normalized_histograms.png",
    )
    # Save the main held-out ROC and marked operating points.
    plot_nominal_roc(
        scores,
        working_points,
        output_dir / "electron_shower_rejection_ROC.png",
    )
    # Save overlaid ROC curves for every detector and clustering variation.
    plot_robustness_roc(
        scores,
        output_dir / "electron_shower_rejection_robustness_ROCs.png",
    )
    # Save the explicit score-versus-energy diagnostic.
    plot_score_vs_energy(
        scores,
        output_dir / "log10_RT_vs_deposited_energy.png",
    )

    # Save the complete object, PCA, split, and robustness method description.
    write_method_summary(
        output_dir / "pca_rejection_method_summary.txt",
        args,
        scores,
        validation_events,
        scenarios,
    )
    # Save nominal operating points and the previous preliminary reference cut.
    write_working_point_summary(
        output_dir / "pca_rejection_working_point_summary.txt",
        working_points,
        scores,
    )
    # Return the table so the caller can print concise completion statistics.
    return working_points


def run(args):
    """Execute object construction, PCA scoring, validation, and plotting."""
    # Resolve the aggregate truth-data directory.
    input_dir = Path(args.input_dir).expanduser().resolve()
    # Construct the particle-table path.
    particle_path = input_dir / "particles_output.txt"
    # Construct the G4-step-table path.
    g4_path = input_dir / "g4_output.txt"
    # Fail early when either required aggregate input is absent.
    if not particle_path.is_file() or not g4_path.is_file():
        raise FileNotFoundError("Expected particles_output.txt and g4_output.txt in --input-dir")

    # Report the first input stage.
    print(f"Reading particle metadata from {particle_path.name} ...")
    # Read selected particle metadata, full genealogy, and the particle inventory.
    particles, genealogy, _ = read_particle_table(particle_path)
    # Report ancestry construction.
    print("Building non-overlapping recursive electron-shower ancestry groups ...")
    # Assign every electron descendant to its earliest electron ancestor.
    shower_membership = build_electron_shower_membership(genealogy)
    # Report G4 truth-step loading.
    print(f"Reading primary-K+ and electron-shower deposits from {g4_path.name} ...")
    # Read charged tracks and all electron-descended G4 deposits.
    g4_steps = read_g4_steps(g4_path, shower_membership)
    # Calculate nominal and varied R_T scores for every detector object.
    scores, skipped, scenarios = analyze_objects(particles, g4_steps, args)
    # Stop explicitly if no object survived the requested PCA quality cut.
    if scores.empty:
        raise RuntimeError("No K+ tracks or electron showers produced valid PCA scores.")
    # Assign whole events to train or validation once for every scenario.
    scores, validation_events = split_events(
        scores,
        args.validation_fraction,
        args.random_seed,
    )
    # Save every requested table, text report, and PNG.
    working_points = write_outputs(
        scores,
        skipped,
        scenarios,
        validation_events,
        args,
    )
    # Extract the nominal 95%-target working point for the terminal summary.
    nominal_95 = working_points[
        working_points["scenario"].eq("nominal_truth")
        & np.isclose(working_points["target_train_kaon_efficiency"], 0.95)
    ].iloc[0]
    # Report the total long-format score-row count.
    print(f"Saved {len(scores):,} object/scenario score rows.")
    # Report the most relevant held-out operating point.
    print(
        "Nominal 95%-target result: "
        f"validation K+ efficiency={100.0 * nominal_95.validation_kaon_efficiency:.2f}%, "
        f"electron-shower rejection={100.0 * nominal_95.validation_electron_rejection:.2f}%."
    )
    # Report the final output directory.
    print(f"Saved PCA rejection outputs under: {Path(args.output_dir).resolve()}")


def build_parser():
    """Define all input, PCA-quality, detector-effect, and split controls."""
    # Create the command-line parser with a concise analysis description.
    parser = argparse.ArgumentParser(
        description="Evaluate R_T=(lambda2+lambda3)/lambda1 for K+/electron-shower rejection."
    )
    # Allow another aggregate truth directory to be analyzed without editing code.
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    # Allow another output hierarchy while preserving the requested default.
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    # Require enough spatial samples for a more stable detector-level PCA estimate.
    parser.add_argument("--min-points", type=int, default=10)
    # Use the documented Q-Pix 0.4-cm x/y pixel pitch by default.
    parser.add_argument("--pixel-pitch-cm", type=float, default=0.4)
    # Make the illustrative position-resolution smearing explicit and configurable.
    parser.add_argument("--position-sigma-cm", type=float, default=0.1)
    # Configure the missing-charge robustness fraction.
    parser.add_argument("--missing-charge-fraction", type=float, default=0.10)
    # Configure the fractional Gaussian charge-noise width.
    parser.add_argument("--energy-noise-fraction", type=float, default=0.10)
    # Configure loss of electron-shower daughter branches from imperfect clustering.
    parser.add_argument("--branch-loss-fraction", type=float, default=0.10)
    # Reserve 30% of event IDs for held-out validation by default.
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    # Fix all random variations and event splits for reproducibility.
    parser.add_argument("--random-seed", type=int, default=12345)
    # Return the configured command-line parser.
    return parser


def validate_arguments(args):
    """Reject invalid analysis controls before reading large truth tables."""
    # Require at least three points mathematically, while retaining the safer default of ten.
    if args.min_points < 3:
        raise ValueError("--min-points must be at least 3.")
    # Require a positive pixel pitch.
    if args.pixel_pitch_cm <= 0.0:
        raise ValueError("--pixel-pitch-cm must be positive.")
    # Require a nonnegative position-smearing width.
    if args.position_sigma_cm < 0.0:
        raise ValueError("--position-sigma-cm must be nonnegative.")
    # Validate every fractional detector-loss argument.
    for name in [
        "missing_charge_fraction",
        "energy_noise_fraction",
        "branch_loss_fraction",
    ]:
        # Read the named fraction from parsed arguments.
        value = getattr(args, name)
        # Require a conventional probability or fractional width range.
        if not 0.0 <= value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1).")
    # Require both train and validation events to exist.
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1).")


def main():
    """Parse controls, validate them, and run the complete rejection study."""
    # Build and parse the command-line interface.
    args = build_parser().parse_args()
    # Validate controls before performing any large file reads.
    validate_arguments(args)
    # Run the requested analysis end to end.
    run(args)


# Execute the command-line entry point only when this file is run as a script.
if __name__ == "__main__":
    # Call the main analysis function.
    main()

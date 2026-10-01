#!/usr/bin/env python3

"""Run a PCA-based containment study for primary K+ tracks in G4 truth output."""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Keep matplotlib cache files local to the analysis area when the script runs on
# systems where the home-directory cache is not writable.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# The analysis is intentionally restricted to the primary K+ from the proton
# decay event.
KAON_PLUS_PDG = 321

# This is the requested RTD event set.
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "qpixrtd_output"
    / "proton_decay_argon_1k_events_RTD_2026-09-03_153336"
)

# This is the requested output directory for the plots and summary files.
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "PCA" / "kaon_decay_1000_events"


def event_number_from_path(path):
    """Read the integer event number from a name such as G4_E123.txt."""
    match = re.search(r"_E(\d+)\.txt$", path.name)
    if match is None:
        raise ValueError(f"Could not read event number from {path.name}")
    return int(match.group(1))


def sorted_event_files(input_dir, prefix):
    """Return event shard files sorted by their numeric event number."""
    paths = sorted(Path(input_dir).glob(f"{prefix}_E*.txt"), key=event_number_from_path)
    if not paths:
        raise FileNotFoundError(f"No {prefix}_E*.txt files were found in {input_dir}")
    return paths


def weighted_quantile(values, weights, quantile):
    """Compute an energy-weighted quantile for one PCA coordinate."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Drop non-finite entries before sorting so the quantile is well defined.
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        raise ValueError("No valid weighted-quantile entries were supplied.")

    # Sort the coordinate values and carry the weights through the same order.
    order = np.argsort(values[valid])
    sorted_values = values[valid][order]
    sorted_weights = weights[valid][order]

    # Build the cumulative deposited-energy fraction along this PCA axis.
    cumulative_weight = np.cumsum(sorted_weights)
    cumulative_fraction = cumulative_weight / cumulative_weight[-1]

    # Interpolate to the requested fraction of contained deposited energy.
    return float(np.interp(quantile, cumulative_fraction, sorted_values))


def read_particles(particles_path):
    """Load the truth particle table for one event shard."""
    columns = [
        "event",
        "particle_track_id",
        "particle_parent_track_id",
        "particle_pdg_code",
        "particle_initial_energy",
        "particle_initial_x",
        "particle_initial_y",
        "particle_initial_z",
        "particle_final_x",
        "particle_final_y",
        "particle_final_z",
        "particle_final_kinetic_energy",
        "particle_decay_flag",
    ]

    # Only read the columns used in this analysis to keep per-event I/O light.
    particles = pd.read_csv(particles_path, usecols=columns)

    # Convert text fields to numeric values and drop malformed rows.
    for column in columns:
        particles[column] = pd.to_numeric(particles[column], errors="coerce")
    particles = particles.dropna(subset=columns).copy()

    # Track IDs and PDG codes are identifiers, so store them as integers.
    integer_columns = [
        "event",
        "particle_track_id",
        "particle_parent_track_id",
        "particle_pdg_code",
        "particle_decay_flag",
    ]
    for column in integer_columns:
        particles[column] = particles[column].astype(int)

    # One row per simulated particle is enough for selecting the primary kaon.
    return particles.drop_duplicates(["event", "particle_track_id"], keep="first")


def read_g4_steps(g4_path):
    """Load the G4 truth energy-deposit steps for one event shard."""
    columns = ["event", "xi", "xf", "yi", "yf", "zi", "zf", "ti", "tf", "E", "ParticleID", "PDG"]

    # Read only the truth-step fields needed for PCA and energy accounting.
    steps = pd.read_csv(g4_path, usecols=columns)

    # Convert all step fields to numeric values and remove malformed rows.
    for column in columns:
        steps[column] = pd.to_numeric(steps[column], errors="coerce")
    steps = steps.dropna(subset=columns).copy()

    # Store IDs as integers so joins and comparisons are exact.
    for column in ["event", "ParticleID", "PDG"]:
        steps[column] = steps[column].astype(int)

    # Keep only physical positive-energy steps.
    return steps[steps["E"] > 0.0].copy()


def read_primary_kaons_from_combined_file(particles_path, max_tracks=None, chunksize=200_000):
    """Stream the combined particle table and retain only primary K+ rows."""
    columns = [
        "event",
        "particle_track_id",
        "particle_parent_track_id",
        "particle_pdg_code",
        "particle_initial_energy",
        "particle_initial_x",
        "particle_initial_y",
        "particle_initial_z",
        "particle_final_x",
        "particle_final_y",
        "particle_final_z",
        "particle_final_kinetic_energy",
        "particle_decay_flag",
    ]
    selected_chunks = []

    # The aggregate particle file is large, so process it in bounded chunks.
    for chunk in pd.read_csv(particles_path, usecols=columns, chunksize=chunksize):
        # Reuse the same cleaning and primary-particle selection as shard mode.
        cleaned = read_particles_from_frame(chunk)
        selected = select_primary_kaons(cleaned)
        if not selected.empty:
            selected_chunks.append(selected)

    if not selected_chunks:
        return pd.DataFrame()

    primary_kaons = (
        pd.concat(selected_chunks, ignore_index=True)
        .drop_duplicates(["event", "particle_track_id"], keep="first")
        .sort_values(["event", "particle_track_id"])
        .reset_index(drop=True)
    )

    # Apply the development limit before reading G4 data so only needed tracks
    # are retained from the much larger combined step table.
    if max_tracks is not None:
        primary_kaons = primary_kaons.head(max_tracks).copy()
    return primary_kaons


def read_particles_from_frame(particles):
    """Clean particle columns already loaded from an aggregate-file chunk."""
    columns = [
        "event",
        "particle_track_id",
        "particle_parent_track_id",
        "particle_pdg_code",
        "particle_initial_energy",
        "particle_initial_x",
        "particle_initial_y",
        "particle_initial_z",
        "particle_final_x",
        "particle_final_y",
        "particle_final_z",
        "particle_final_kinetic_energy",
        "particle_decay_flag",
    ]
    particles = particles.loc[:, columns].copy()
    for column in columns:
        particles[column] = pd.to_numeric(particles[column], errors="coerce")
    particles = particles.dropna(subset=columns)
    for column in [
        "event",
        "particle_track_id",
        "particle_parent_track_id",
        "particle_pdg_code",
        "particle_decay_flag",
    ]:
        particles[column] = particles[column].astype(int)
    return particles


def read_matching_g4_steps_from_combined_file(g4_path, primary_kaons, chunksize=300_000):
    """Stream combined G4 output and retain steps belonging to selected K+ tracks."""
    columns = ["event", "xi", "xf", "yi", "yf", "zi", "zf", "ti", "tf", "E", "ParticleID", "PDG"]
    track_keys = set(
        zip(
            primary_kaons["event"].astype(int),
            primary_kaons["particle_track_id"].astype(int),
        )
    )
    selected_chunks = []

    for chunk in pd.read_csv(g4_path, usecols=columns, chunksize=chunksize):
        for column in columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.dropna(subset=columns)
        for column in ["event", "ParticleID", "PDG"]:
            chunk[column] = chunk[column].astype(int)

        # Check both event and track ID because Geant4 track IDs restart in each event.
        keys = zip(chunk["event"], chunk["ParticleID"])
        keep = np.fromiter((key in track_keys for key in keys), dtype=bool, count=len(chunk))
        selected = chunk[keep & chunk["PDG"].eq(KAON_PLUS_PDG) & chunk["E"].gt(0.0)]
        if not selected.empty:
            selected_chunks.append(selected.copy())

    if not selected_chunks:
        return pd.DataFrame(columns=columns)
    return pd.concat(selected_chunks, ignore_index=True)


def select_primary_kaons(particles):
    """Select only K+ tracks produced directly by the proton decay primary."""
    return particles[
        particles["particle_pdg_code"].eq(KAON_PLUS_PDG)
        & particles["particle_parent_track_id"].eq(0)
    ].copy()


def add_step_geometry(steps):
    """Add midpoint coordinates and step lengths to a G4 step table."""
    steps = steps.copy()

    # The step midpoint is the spatial point used in the PCA.
    steps["x_cm"] = 0.5 * (steps["xi"] + steps["xf"])
    steps["y_cm"] = 0.5 * (steps["yi"] + steps["yf"])
    steps["z_cm"] = 0.5 * (steps["zi"] + steps["zf"])

    # The step length is a useful diagnostic for track length.
    steps["ds_cm"] = np.sqrt(
        (steps["xf"] - steps["xi"]) ** 2
        + (steps["yf"] - steps["yi"]) ** 2
        + (steps["zf"] - steps["zi"]) ** 2
    )

    # Keep only steps with a positive spatial extent.
    return steps[steps["ds_cm"] > 0.0].copy()


def energy_weighted_pca(points, energies):
    """Find PCA axes for the K+ step midpoints, weighted by deposited energy."""
    points = np.asarray(points, dtype=float)
    energies = np.asarray(energies, dtype=float)

    # Require enough spatial points to define at least two principal axes.
    if points.shape[0] < 3:
        raise ValueError("At least three G4 steps are required for a 3D PCA.")

    # Normalize deposited energies into PCA weights.
    total_energy = float(np.sum(energies))
    if total_energy <= 0.0:
        raise ValueError("Total deposited energy must be positive.")
    weights = energies / total_energy

    # Compute the deposited-energy-weighted centroid of the kaon steps.
    center = np.average(points, axis=0, weights=weights)

    # Shift the point cloud to the weighted centroid before building covariance.
    shifted = points - center

    # Compute the weighted covariance matrix in x/y/z truth coordinates.
    covariance = (shifted * weights[:, np.newaxis]).T @ shifted

    # Diagonalize the symmetric covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort principal axes from largest to smallest variance.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order].T

    # Make axis directions deterministic so repeated runs draw the same box.
    for axis_index in range(components.shape[0]):
        largest_component = np.argmax(np.abs(components[axis_index]))
        if components[axis_index, largest_component] < 0:
            components[axis_index] *= -1.0

    # Project each point into the PCA coordinate system.
    coordinates = shifted @ components.T

    # Record the variance fraction explained by each axis.
    total_variance = float(np.sum(np.clip(eigenvalues, 0.0, None)))
    if total_variance > 0.0:
        explained = np.clip(eigenvalues, 0.0, None) / total_variance
    else:
        explained = np.zeros_like(eigenvalues)

    return center, components, coordinates, eigenvalues, explained


def analyze_primary_kaon_track(steps, containment_quantile):
    """Run PCA and measure deposited energy outside the robust PC1-PC2 box."""
    if not 0.0 < containment_quantile <= 1.0:
        raise ValueError("--containment-quantile must be in the interval (0, 1].")

    # Add midpoint and path-length columns before the PCA.
    steps = add_step_geometry(steps)

    # Prepare the coordinate and deposited-energy arrays.
    points = steps[["x_cm", "y_cm", "z_cm"]].to_numpy(dtype=float)
    energies = steps["E"].to_numpy(dtype=float)

    # Find the deposited-energy-weighted PCA axes of the primary K+ track.
    center, components, coordinates, eigenvalues, explained = energy_weighted_pca(points, energies)

    # Use a symmetric central energy interval so the box is not dominated by
    # rare outlying steps.
    tail_fraction = 0.5 * (1.0 - containment_quantile)
    pc1_low = weighted_quantile(coordinates[:, 0], energies, tail_fraction)
    pc1_high = weighted_quantile(coordinates[:, 0], energies, 1.0 - tail_fraction)
    pc2_low = weighted_quantile(coordinates[:, 1], energies, tail_fraction)
    pc2_high = weighted_quantile(coordinates[:, 1], energies, 1.0 - tail_fraction)

    # Mark steps outside the PC1-PC2 rectangular region.
    inside_box = (
        (coordinates[:, 0] >= pc1_low)
        & (coordinates[:, 0] <= pc1_high)
        & (coordinates[:, 1] >= pc2_low)
        & (coordinates[:, 1] <= pc2_high)
    )

    # Sum energy inside and outside the PCA-defined box.
    total_energy = float(np.sum(energies))
    outside_energy = float(np.sum(energies[~inside_box]))
    inside_energy = total_energy - outside_energy
    outside_percent = 100.0 * outside_energy / total_energy if total_energy > 0.0 else np.nan

    # Store PCA coordinates and containment tags for optional plotting/debugging.
    steps["pc1_cm"] = coordinates[:, 0]
    steps["pc2_cm"] = coordinates[:, 1]
    steps["pc3_cm"] = coordinates[:, 2]
    steps["inside_pca_box"] = inside_box

    return {
        "steps": steps,
        "center": center,
        "components": components,
        "coordinates": coordinates,
        "eigenvalues": eigenvalues,
        "explained": explained,
        "box": {
            "pc1_low": pc1_low,
            "pc1_high": pc1_high,
            "pc2_low": pc2_low,
            "pc2_high": pc2_high,
        },
        "total_energy_MeV": total_energy,
        "inside_energy_MeV": inside_energy,
        "outside_energy_MeV": outside_energy,
        "outside_percent": outside_percent,
        "track_length_cm": float(steps["ds_cm"].sum()),
    }


def pca_box_corners(center, components, box):
    """Build the four 3D corners of the PC1-PC2 rectangle."""
    corners_pc = np.array(
        [
            [box["pc1_low"], box["pc2_low"], 0.0],
            [box["pc1_high"], box["pc2_low"], 0.0],
            [box["pc1_high"], box["pc2_high"], 0.0],
            [box["pc1_low"], box["pc2_high"], 0.0],
            [box["pc1_low"], box["pc2_low"], 0.0],
        ],
        dtype=float,
    )

    # Convert PCA coordinates back into detector x/y/z coordinates.
    return center + corners_pc @ components


def set_axes_equal(ax):
    """Use equal x/y/z scaling so the event display geometry is not distorted."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)


def classify_kaon_decay(particle):
    """Classify a decay-flagged kaon using its endpoint kinetic energy."""
    decay_flag = int(particle["particle_decay_flag"])
    final_kinetic_energy = float(particle["particle_final_kinetic_energy"])

    # Follow the convention used by kaon_decay_energy_hist.py in this analysis:
    # zero endpoint kinetic energy means decay at rest; positive energy means
    # the kaon was still moving when it decayed.
    if decay_flag != 1:
        return "No decay recorded"
    if np.isclose(final_kinetic_energy, 0.0, atol=1.0e-9):
        return "Decay at rest"
    return "Decay in flight"


def save_event_display(event_id, track_id, result, particle, output_path, containment_quantile):
    """Save a 3D event display for one primary K+ track and its PCA box."""
    steps = result["steps"].sort_values("ti", kind="mergesort")

    # Build the thin red truth path from the ordered G4 step midpoints.
    path_points = steps[["x_cm", "y_cm", "z_cm"]].to_numpy(dtype=float)

    # Build the PCA rectangle in detector coordinates.
    rectangle = pca_box_corners(result["center"], result["components"], result["box"])
    rectangle_segments = np.stack([rectangle[:-1], rectangle[1:]], axis=1)

    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    # Draw K+ deposits; outside-box steps are open black circles.
    inside = steps["inside_pca_box"].to_numpy(dtype=bool)
    colors = steps["E"].to_numpy(dtype=float)
    scatter = ax.scatter(
        steps.loc[inside, "x_cm"],
        steps.loc[inside, "y_cm"],
        steps.loc[inside, "z_cm"],
        c=colors[inside],
        cmap="viridis",
        s=22,
        alpha=0.82,
        label="K+ deposits inside PCA box",
    )
    if np.any(~inside):
        ax.scatter(
            steps.loc[~inside, "x_cm"],
            steps.loc[~inside, "y_cm"],
            steps.loc[~inside, "z_cm"],
            facecolors="none",
            edgecolors="black",
            s=34,
            linewidths=0.8,
            label="K+ deposits outside PCA box",
        )

    # Draw the primary K+ path as requested.
    ax.plot(
        path_points[:, 0],
        path_points[:, 1],
        path_points[:, 2],
        color="red",
        linewidth=1.0,
        label="Primary K+ G4 path",
    )

    # Draw the PCA box outline in the PC1-PC2 plane.
    collection = Line3DCollection(rectangle_segments, colors="black", linewidths=0.5)
    ax.add_collection3d(collection)
    ax.plot([], [], [], color="black", linewidth=0.5, label="PCA PC1-PC2 box")

    # Mark the particle-table start and end points for orientation.
    ax.scatter(
        [particle["particle_initial_x"]],
        [particle["particle_initial_y"]],
        [particle["particle_initial_z"]],
        marker="*",
        s=120,
        color="tab:orange",
        edgecolors="black",
        linewidths=0.5,
        label="Particle-table start",
    )
    ax.scatter(
        [particle["particle_final_x"]],
        [particle["particle_final_y"]],
        [particle["particle_final_z"]],
        marker="X",
        s=80,
        color="tab:purple",
        edgecolors="black",
        linewidths=0.5,
        label="Particle-table end",
    )

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_zlabel("z [cm]")
    ax.set_title(f"Event {event_id}: primary K+ track {track_id} PCA containment")
    set_axes_equal(ax)

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.74, pad=0.08)
    colorbar.set_label("G4 step deposited energy [MeV]")

    decay_mode = classify_kaon_decay(particle)
    final_kinetic_energy = float(particle["particle_final_kinetic_energy"])
    legend_text = (
        f"Outside energy: {result['outside_percent']:.2f}%\n"
        f"Outside / total: {result['outside_energy_MeV']:.3g} / {result['total_energy_MeV']:.3g} MeV\n"
        f"Decay mode: {decay_mode}\n"
        f"Final K+ kinetic energy: {final_kinetic_energy:.4g} MeV\n"
        f"Track length: {result['track_length_cm']:.2f} cm\n"
        f"PC1 + PC2 variance: {100.0 * np.sum(result['explained'][:2]):.2f}%\n"
        f"Box containment target: {100.0 * containment_quantile:.1f}%"
    )
    ax.text2D(
        0.02,
        0.98,
        legend_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.65"},
    )
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_outside_energy_histogram(summary, output_path):
    """Save the requested 1%-bin histogram of energy outside the PCA box."""
    values = summary["outside_percent"].dropna().to_numpy(dtype=float)
    bins = np.arange(0.0, 101.0, 1.0)

    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.hist(values, bins=bins, color="tab:blue", edgecolor="black", linewidth=0.45)
    ax.set_xlabel("Percent energy outside the PCA PC1-PC2 box [%]")
    ax.set_ylabel("Counts")
    ax.set_title("Primary K+ PCA containment")
    ax.set_xlim(0.0, 100.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.98,
        0.96,
        "\n".join(
            [
                f"Tracks: {len(values)}",
                f"Mean: {np.mean(values):.2f}%",
                f"Median: {np.median(values):.2f}%",
                f"90th percentile: {np.percentile(values, 90):.2f}%",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.6"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_diagnostic_scatters(summary, output_dir):
    """Save diagnostic plots that help interpret the containment histogram."""
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(
        summary["track_length_cm"],
        summary["outside_percent"],
        s=18,
        alpha=0.65,
        linewidths=0,
        color="tab:green",
    )
    ax.set_xlabel("Primary K+ G4 path length [cm]")
    ax.set_ylabel("Energy outside PCA box [%]")
    ax.set_title("PCA leakage vs primary K+ path length")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "outside_energy_percent_vs_track_length.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(
        100.0 * summary["pc1_pc2_explained_variance"],
        summary["outside_percent"],
        s=18,
        alpha=0.65,
        linewidths=0,
        color="tab:purple",
    )
    ax.set_xlabel("Variance explained by PC1 + PC2 [%]")
    ax.set_ylabel("Energy outside PCA box [%]")
    ax.set_title("PCA leakage vs two-component explained variance")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "outside_energy_percent_vs_pc12_variance.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6.2))
    bins = np.linspace(0.0, 100.0, 51)
    ax.hist(
        100.0 * summary["pc1_explained_variance"],
        bins=bins,
        histtype="step",
        linewidth=2.0,
        label="PC1",
    )
    ax.hist(
        100.0 * summary["pc1_pc2_explained_variance"],
        bins=bins,
        histtype="step",
        linewidth=2.0,
        label="PC1 + PC2",
    )
    ax.set_xlabel("Explained variance [%]")
    ax.set_ylabel("Counts")
    ax.set_title("Primary K+ PCA explained variance")
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_explained_variance_hist.png", dpi=220)
    plt.close(fig)


def write_text_summary(summary, output_path, input_dir, containment_quantile):
    """Write a small plain-text summary of the PCA containment run."""
    outside = summary["outside_percent"].to_numpy(dtype=float)
    lines = [
        "Primary K+ PCA containment summary",
        "===================================",
        "",
        f"Input directory: {input_dir}",
        f"Tracks analyzed: {len(summary)}",
        f"PCA box energy-quantile target: {100.0 * containment_quantile:.3g}%",
        "",
        "Energy outside PC1-PC2 box:",
        f"  Mean: {np.mean(outside):.6g}%",
        f"  Median: {np.median(outside):.6g}%",
        f"  Standard deviation: {np.std(outside):.6g}%",
        f"  Minimum: {np.min(outside):.6g}%",
        f"  Maximum: {np.max(outside):.6g}%",
        f"  90th percentile: {np.percentile(outside, 90):.6g}%",
        "",
        "Method:",
        "  Only particle-table K+ rows with parent_track_id == 0 are selected.",
        "  G4 truth steps are matched by event number and ParticleID because track IDs restart each event.",
        "  PCA is run on G4 step midpoints in detector x/y/z coordinates.",
        "  Each midpoint is weighted by the G4 step deposited energy.",
        "  The PC1-PC2 box is the central weighted quantile interval along PC1 and PC2.",
        "  Energy outside the box is the sum of K+ step energy whose PC1 or PC2 coordinate lies outside that interval.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def iter_primary_kaon_tracks(input_dir, max_tracks=None):
    """Yield particle rows and matching G4 steps from aggregate files or shards."""
    combined_g4_path = input_dir / "g4_output.txt"
    combined_particles_path = input_dir / "particles_output.txt"

    if combined_g4_path.is_file() and combined_particles_path.is_file():
        print(f"Reading primary K+ particles from {combined_particles_path.name} ...")
        primary_kaons = read_primary_kaons_from_combined_file(
            combined_particles_path,
            max_tracks=max_tracks,
        )
        if primary_kaons.empty:
            return

        print(f"Reading matching primary K+ steps from {combined_g4_path.name} ...")
        matching_steps = read_matching_g4_steps_from_combined_file(
            combined_g4_path,
            primary_kaons,
        )

        for _, particle in primary_kaons.iterrows():
            event_id = int(particle["event"])
            track_id = int(particle["particle_track_id"])
            steps = matching_steps[
                matching_steps["event"].eq(event_id)
                & matching_steps["ParticleID"].eq(track_id)
            ].copy()
            yield event_id, track_id, particle, steps
        return

    # Retain shard support for older output directories that do not contain the
    # two combined tables.
    g4_files = sorted_event_files(input_dir, "G4")
    particle_files = {
        event_number_from_path(path): path
        for path in sorted_event_files(input_dir, "particles")
    }
    yielded = 0
    for g4_path in g4_files:
        event_id = event_number_from_path(g4_path)
        particles_path = particle_files.get(event_id)
        if particles_path is None:
            continue
        primary_kaons = select_primary_kaons(read_particles(particles_path))
        if primary_kaons.empty:
            continue
        g4_steps = read_g4_steps(g4_path)
        for _, particle in primary_kaons.sort_values("particle_track_id").iterrows():
            if max_tracks is not None and yielded >= max_tracks:
                return
            track_id = int(particle["particle_track_id"])
            steps = g4_steps[
                g4_steps["ParticleID"].eq(track_id)
                & g4_steps["PDG"].eq(KAON_PLUS_PDG)
            ].copy()
            yield event_id, track_id, particle, steps
            yielded += 1


def run_analysis(args):
    """Loop over events, analyze the primary K+, and save plots/tables."""
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    event_display_dir = output_dir / "event_displays"
    high_outside_display_dir = output_dir / "highest_outside_energy_event_displays"

    # Create the requested output directories if they do not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)
    event_display_dir.mkdir(parents=True, exist_ok=True)
    high_outside_display_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    display_records = []
    saved_event_displays = 0
    skipped_events = []

    for event_id, track_id, particle, kaon_steps in iter_primary_kaon_tracks(
        input_dir,
        max_tracks=args.max_events,
    ):
        try:
            if kaon_steps.empty:
                skipped_events.append((event_id, f"no G4 steps for primary K+ track {track_id}"))
                continue

            result = analyze_primary_kaon_track(kaon_steps, args.containment_quantile)
            decay_mode = classify_kaon_decay(particle)

            rows.append(
                {
                    "event": event_id,
                    "track_id": track_id,
                    "n_g4_steps": int(len(result["steps"])),
                    "total_energy_MeV": result["total_energy_MeV"],
                    "inside_energy_MeV": result["inside_energy_MeV"],
                    "outside_energy_MeV": result["outside_energy_MeV"],
                    "outside_percent": result["outside_percent"],
                    "track_length_cm": result["track_length_cm"],
                    "pc1_explained_variance": float(result["explained"][0]),
                    "pc2_explained_variance": float(result["explained"][1]),
                    "pc3_explained_variance": float(result["explained"][2]),
                    "pc1_pc2_explained_variance": float(np.sum(result["explained"][:2])),
                    "pc1_low_cm": result["box"]["pc1_low"],
                    "pc1_high_cm": result["box"]["pc1_high"],
                    "pc2_low_cm": result["box"]["pc2_low"],
                    "pc2_high_cm": result["box"]["pc2_high"],
                    "particle_initial_energy_MeV": float(particle["particle_initial_energy"]),
                    "particle_final_kinetic_energy_MeV": float(particle["particle_final_kinetic_energy"]),
                    "particle_decay_flag": int(particle["particle_decay_flag"]),
                    "decay_mode": decay_mode,
                }
            )

            # Retain the compact primary-K+ result so the largest-leakage tracks
            # can be ranked only after every event has been analyzed.
            display_records.append(
                {
                    "event_id": event_id,
                    "track_id": track_id,
                    "result": result,
                    "particle": particle.copy(),
                }
            )

            if saved_event_displays < args.event_displays:
                display_path = event_display_dir / f"event_{event_id:04d}_track_{track_id}_pca_box.png"
                save_event_display(
                    event_id=event_id,
                    track_id=track_id,
                    result=result,
                    particle=particle,
                    output_path=display_path,
                    containment_quantile=args.containment_quantile,
                )
                saved_event_displays += 1

        except Exception as exc:
            skipped_events.append((event_id, str(exc)))

    if not rows:
        raise RuntimeError("No primary K+ tracks were successfully analyzed.")

    summary = pd.DataFrame(rows).sort_values(["event", "track_id"]).reset_index(drop=True)
    summary_path = output_dir / "kaon_pca_track_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Sort all successful tracks by outside-energy percentage and save a second
    # display set focused on the most extreme PCA-box leakage behavior.
    ranked_records = sorted(
        display_records,
        key=lambda record: record["result"]["outside_percent"],
        reverse=True,
    )
    for rank, record in enumerate(
        ranked_records[: args.highest_outside_event_displays],
        start=1,
    ):
        display_path = high_outside_display_dir / (
            f"rank_{rank:02d}_event_{record['event_id']:04d}_"
            f"track_{record['track_id']}_pca_box.png"
        )
        save_event_display(
            event_id=record["event_id"],
            track_id=record["track_id"],
            result=record["result"],
            particle=record["particle"],
            output_path=display_path,
            containment_quantile=args.containment_quantile,
        )

    if skipped_events:
        skipped_path = output_dir / "skipped_events.txt"
        skipped_path.write_text(
            "\n".join(f"event {event_id}: {reason}" for event_id, reason in skipped_events) + "\n"
        )

    plot_outside_energy_histogram(summary, output_dir / "outside_energy_percent_hist_1pct_bins.png")
    plot_diagnostic_scatters(summary, output_dir)
    write_text_summary(
        summary=summary,
        output_path=output_dir / "pca_containment_summary.txt",
        input_dir=input_dir,
        containment_quantile=args.containment_quantile,
    )

    print(f"Analyzed {len(summary)} primary K+ track(s).")
    print(f"Saved summary table: {summary_path}")
    print(f"Saved plots under: {output_dir}")
    if skipped_events:
        print(f"Skipped {len(skipped_events)} event/track item(s); see skipped_events.txt")


def build_parser():
    """Create the command-line interface for the PCA analysis."""
    parser = argparse.ArgumentParser(
        description="PCA containment study for the primary K+ in proton-decay G4 truth."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing g4_output.txt and particles_output.txt (or event shards).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PCA plots and summaries will be saved.",
    )
    parser.add_argument(
        "--containment-quantile",
        type=float,
        default=0.98,
        help="Central deposited-energy quantile used for the PC1-PC2 box.",
    )
    parser.add_argument(
        "--event-displays",
        type=int,
        default=10,
        help="Number of first successfully analyzed primary-K+ tracks to draw.",
    )
    parser.add_argument(
        "--highest-outside-event-displays",
        type=int,
        default=10,
        help="Number of largest outside-energy-percent primary-K+ tracks to draw.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional development/debug limit on the number of analyzed primary-K+ tracks.",
    )
    return parser


def main():
    """Parse arguments and run the PCA containment analysis."""
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()

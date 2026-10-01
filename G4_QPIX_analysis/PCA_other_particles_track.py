#!/usr/bin/env python3

"""Run the primary-K+ PCA containment analysis for other detector particles."""

import argparse
import heapq
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Reuse the tested PCA mathematics from the primary-kaon analysis.  This keeps
# the box definition and energy accounting identical in both studies.
from PCA_kaon_track import analyze_primary_kaon_track, pca_box_corners, set_axes_equal


DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "qpixrtd_output"
    / "proton_decay_argon_1k_events_RTD_2026-09-03_153336"
)
DEFAULT_EVENT_SET_DIR = SCRIPT_DIR / "PCA" / "kaon_decay_1000_events"

# The categories are intentionally separated by charge/sign.  Primary and
# secondary K+ tracks share PDG 321 and are separated using parent_track_id.
CATEGORY_INFO = {
    "primary_k_plus": {"label": "Primary K+", "pdg": 321, "other": False},
    "secondary_k_plus": {"label": "Secondary K+", "pdg": 321, "other": True},
    "muon_plus": {"label": "Muon+", "pdg": -13, "other": True},
    "muon_minus": {"label": "Muon-", "pdg": 13, "other": True},
    "electron": {"label": "Electron", "pdg": 11, "other": True},
    "electron_showers": {"label": "Electron shower", "pdg": 11, "other": True},
    "positron": {"label": "Positron", "pdg": -11, "other": True},
    "pion_plus": {"label": "Pion+", "pdg": 211, "other": True},
    "pion_minus": {"label": "Pion-", "pdg": -211, "other": True},
    "proton": {"label": "Proton", "pdg": 2212, "other": True},
}

PDG_TO_CATEGORY = {
    -211: "pion_minus",
    -13: "muon_plus",
    -11: "positron",
    11: "electron",
    13: "muon_minus",
    211: "pion_plus",
    2212: "proton",
}
ANALYZED_PDGS = set(PDG_TO_CATEGORY) | {321}

# These species occur in the particle table but have no direct positive-energy
# G4 step rows in this event set.  Their charged daughters/secondaries are still
# analyzed in the charged categories above.
NO_DIRECT_PCA_SPECIES = {
    22: "Photon",
    111: "Pion0",
    130: "K0-long",
    310: "K0-short",
    -321: "K-",
    2112: "Neutron",
}


def category_for_particle(pdg, parent_track_id):
    """Map a particle PDG and parent ID to one analysis category."""
    pdg = int(pdg)
    if pdg == 321:
        return "primary_k_plus" if int(parent_track_id) == 0 else "secondary_k_plus"
    return PDG_TO_CATEGORY.get(pdg)


def endpoint_status(particle):
    """Describe whether the particle decayed at rest, in flight, or not at all."""
    decay_flag = int(particle["particle_decay_flag"])
    final_ke = float(particle["particle_final_kinetic_energy"])
    if decay_flag != 1:
        return "No decay recorded"
    if np.isclose(final_ke, 0.0, atol=1.0e-9):
        return "Decay at rest"
    return "Decay in flight"


def read_particle_table(path):
    """Stream particle metadata and retain the species used by this study."""
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
    inventory = Counter()
    selected_chunks = []
    genealogy_chunks = []

    for chunk in pd.read_csv(path, usecols=columns, chunksize=200_000):
        for column in columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.dropna(subset=columns)
        chunk["particle_pdg_code"] = chunk["particle_pdg_code"].astype(int)
        inventory.update(chunk["particle_pdg_code"])
        genealogy_chunks.append(
            chunk[
                [
                    "event",
                    "particle_track_id",
                    "particle_parent_track_id",
                    "particle_pdg_code",
                ]
            ].copy()
        )
        selected = chunk[chunk["particle_pdg_code"].isin(ANALYZED_PDGS)].copy()
        if not selected.empty:
            selected_chunks.append(selected)

    particles = pd.concat(selected_chunks, ignore_index=True)
    for column in [
        "event",
        "particle_track_id",
        "particle_parent_track_id",
        "particle_pdg_code",
        "particle_decay_flag",
    ]:
        particles[column] = particles[column].astype(int)
    particles = particles.drop_duplicates(["event", "particle_track_id"], keep="first")
    genealogy = pd.concat(genealogy_chunks, ignore_index=True)
    for column in genealogy.columns:
        genealogy[column] = genealogy[column].astype(int)
    genealogy = genealogy.drop_duplicates(["event", "particle_track_id"], keep="first")
    return particles, genealogy, inventory


def build_electron_shower_membership(genealogy):
    """Assign each electron-descended particle to its earliest electron ancestor."""
    membership_rows = []

    # Geant4 creates a parent before its daughters, so sorting by track ID lets
    # each child inherit an already-known electron-shower root from its parent.
    for event_id, event_particles in genealogy.groupby("event", sort=True):
        root_by_track = {}
        event_particles = event_particles.sort_values("particle_track_id")
        for particle in event_particles.itertuples(index=False):
            track_id = int(particle.particle_track_id)
            parent_id = int(particle.particle_parent_track_id)
            pdg = int(particle.particle_pdg_code)

            inherited_root = root_by_track.get(parent_id)
            if inherited_root is not None:
                # Every descendant stays attached to the first electron in its
                # ancestry, even when the daughter is another electron.
                root_by_track[track_id] = inherited_root
            elif pdg == 11:
                # An electron with no earlier electron ancestor starts a new,
                # non-overlapping truth shower.
                root_by_track[track_id] = track_id

        membership_rows.extend(
            (int(event_id), track_id, root_track_id)
            for track_id, root_track_id in root_by_track.items()
        )

    return pd.DataFrame(
        membership_rows,
        columns=["event", "ParticleID", "electron_shower_root_track_id"],
    )


def read_g4_steps(path, shower_membership):
    """Stream charged tracks plus all deposits descended from electron roots."""
    columns = ["event", "xi", "xf", "yi", "yf", "zi", "zf", "ti", "tf", "E", "ParticleID", "PDG"]
    selected_chunks = []
    shower_root_lookup = shower_membership.set_index(
        ["event", "ParticleID"]
    )["electron_shower_root_track_id"]

    for chunk in pd.read_csv(path, usecols=columns, chunksize=300_000):
        for column in columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.dropna(subset=columns)
        for column in ["event", "ParticleID", "PDG"]:
            chunk[column] = chunk[column].astype(int)
        chunk = chunk[chunk["E"].gt(0.0)].copy()

        # Zero-length deposits cannot define a spatial PCA direction.
        positive_length = (
            (chunk["xf"] - chunk["xi"]) ** 2
            + (chunk["yf"] - chunk["yi"]) ** 2
            + (chunk["zf"] - chunk["zi"]) ** 2
        ) > 0.0
        chunk = chunk[positive_length]
        if chunk.empty:
            continue

        # Attach the earliest electron ancestor to every shower-descended step.
        keys = pd.MultiIndex.from_arrays([chunk["event"], chunk["ParticleID"]])
        chunk["electron_shower_root_track_id"] = shower_root_lookup.reindex(keys).to_numpy()

        # Retain the union of ordinary charged tracks and complete electron
        # shower descendants.  A descendant may have a neutral or ion PDG code.
        keep = chunk["PDG"].isin(ANALYZED_PDGS) | chunk[
            "electron_shower_root_track_id"
        ].notna()
        selected_chunks.append(chunk[keep].copy())

    return pd.concat(selected_chunks, ignore_index=True)


def particle_lookup_table(particles):
    """Create an event/track keyed particle table for fast G4 group lookup."""
    return particles.set_index(["event", "particle_track_id"], drop=False).sort_index()


def save_event_display(record, output_path, containment_quantile):
    """Draw one particle track, its deposits, and its thin PC1-PC2 box."""
    event_id = record["event"]
    track_id = record["track_id"]
    category = record["particle_type"]
    label = CATEGORY_INFO[category]["label"]
    particle = record["particle"]
    result = record["result"]
    steps = result["steps"].sort_values("ti", kind="mergesort")
    path_points = steps[["x_cm", "y_cm", "z_cm"]].to_numpy(dtype=float)

    rectangle = pca_box_corners(result["center"], result["components"], result["box"])
    rectangle_segments = np.stack([rectangle[:-1], rectangle[1:]], axis=1)

    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    inside = steps["inside_pca_box"].to_numpy(dtype=bool)
    energies = steps["E"].to_numpy(dtype=float)
    scatter = ax.scatter(
        steps.loc[inside, "x_cm"],
        steps.loc[inside, "y_cm"],
        steps.loc[inside, "z_cm"],
        c=energies[inside],
        cmap="viridis",
        s=22,
        alpha=0.82,
        label=f"{label} deposits inside PCA box",
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
            label=f"{label} deposits outside PCA box",
        )

    if category == "electron_showers":
        # Draw each descendant trajectory separately.  Connecting all shower
        # steps in global time order would create lines between unrelated
        # branches and misrepresent the electromagnetic shower topology.
        for _, branch in steps.groupby("ParticleID", sort=False):
            branch = branch.sort_values("ti", kind="mergesort")
            branch_points = branch[["x_cm", "y_cm", "z_cm"]].to_numpy(dtype=float)
            ax.plot(
                branch_points[:, 0],
                branch_points[:, 1],
                branch_points[:, 2],
                color="red",
                linewidth=0.65,
                alpha=0.55,
            )
        ax.plot([], [], [], color="red", linewidth=0.65, label="Electron-shower G4 paths")
    else:
        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            path_points[:, 2],
            color="red",
            linewidth=1.0,
            label=f"{label} G4 path",
        )
    box_lines = Line3DCollection(rectangle_segments, colors="black", linewidths=0.5)
    ax.add_collection3d(box_lines)
    ax.plot([], [], [], color="black", linewidth=0.5, label="PCA PC1-PC2 box")
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
    ax.set_title(f"Event {event_id}: {label} track {track_id} PCA containment")
    set_axes_equal(ax)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.74, pad=0.08)
    colorbar.set_label("G4 step deposited energy [MeV]")

    final_ke = float(particle["particle_final_kinetic_energy"])
    info = (
        f"Particle: {label} (PDG {int(particle['particle_pdg_code'])})\n"
        f"Outside energy: {result['outside_percent']:.2f}%\n"
        f"Outside / total: {result['outside_energy_MeV']:.3g} / {result['total_energy_MeV']:.3g} MeV\n"
        f"Endpoint status: {endpoint_status(particle)}\n"
        f"Final kinetic energy: {final_ke:.4g} MeV\n"
        f"Track length: {result['track_length_cm']:.2f} cm\n"
        f"PC1 + PC2 variance: {100.0 * np.sum(result['explained'][:2]):.2f}%\n"
        f"Box containment target: {100.0 * containment_quantile:.1f}%"
    )
    ax.text2D(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.65"},
    )
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_category(summary, category, output_dir):
    """Save the same four containment diagnostics for one particle category."""
    label = CATEGORY_INFO[category]["label"]
    values = summary["outside_percent"].to_numpy(dtype=float)
    bins = np.arange(0.0, 101.0, 1.0)

    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.hist(values, bins=bins, color="tab:blue", edgecolor="black", linewidth=0.45)
    ax.set(xlabel="Percent energy outside the PCA PC1-PC2 box [%]", ylabel="Counts")
    ax.set_title(f"{label} PCA containment")
    ax.set_xlim(0.0, 100.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.98,
        0.96,
        f"Bin width: 1%\nTracks: {len(values)}\nMean: {np.mean(values):.2f}%\nMedian: {np.median(values):.2f}%\n90th percentile: {np.percentile(values, 90):.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.6"},
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"outside_energy_percent_hist_1pct_bins_{category}.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(summary["track_length_cm"], values, s=10, alpha=0.35, linewidths=0, rasterized=True)
    ax.set(xlabel=f"{label} G4 path length [cm]", ylabel="Energy outside PCA box [%]")
    ax.set_title(f"PCA leakage vs {label} path length")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"outside_energy_percent_vs_track_length_{category}.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(
        100.0 * summary["pc1_pc2_explained_variance"],
        values,
        s=10,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    ax.set(xlabel="Variance explained by PC1 + PC2 [%]", ylabel="Energy outside PCA box [%]")
    ax.set_title(f"{label} PCA leakage vs two-component variance")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"outside_energy_percent_vs_pc12_variance_{category}.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6.2))
    variance_bins = np.linspace(0.0, 100.0, 51)
    ax.hist(100.0 * summary["pc1_explained_variance"], bins=variance_bins, histtype="step", linewidth=2, label="PC1")
    ax.hist(100.0 * summary["pc1_pc2_explained_variance"], bins=variance_bins, histtype="step", linewidth=2, label="PC1 + PC2")
    ax.set(xlabel="Explained variance [%]", ylabel="Counts")
    ax.set_title(f"{label} PCA explained variance")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.02,
        0.96,
        "Bin width: 2%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.6"},
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"pca_explained_variance_hist_{category}.png", dpi=220)
    plt.close(fig)


def write_category_summary(summary, category, output_path, containment_quantile):
    """Write human-readable statistics and method notes for one category."""
    label = CATEGORY_INFO[category]["label"]
    outside = summary["outside_percent"].to_numpy(dtype=float)
    lines = [
        f"{label} PCA containment summary",
        "=" * (len(label) + 24),
        "",
        f"Tracks analyzed: {len(summary)}",
        f"PCA box energy-quantile target: {100.0 * containment_quantile:.3g}%",
        f"Mean outside energy: {np.mean(outside):.6g}%",
        f"Median outside energy: {np.median(outside):.6g}%",
        f"Standard deviation: {np.std(outside):.6g}%",
        f"Minimum: {np.min(outside):.6g}%",
        f"Maximum: {np.max(outside):.6g}%",
        f"90th percentile: {np.percentile(outside, 90):.6g}%",
        "",
        "Method:",
        "  PCA uses positive-energy G4 step midpoints weighted by deposited energy.",
        "  The box is the central weighted interval along PC1 and PC2.",
        "  Outside energy is summed where either PC1 or PC2 lies outside its interval.",
        "  At least three nonzero-length G4 steps are required per track.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def plot_combined(summary, output_dir):
    """Save comparison plots containing primary K+ and every other category."""
    categories = [category for category in CATEGORY_INFO if category in set(summary["particle_type"])]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(categories)))
    bins = np.arange(0.0, 101.0, 1.0)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    for color, category in zip(colors, categories):
        data = summary.loc[summary["particle_type"].eq(category), "outside_percent"]
        ax.hist(data, bins=bins, histtype="step", linewidth=1.5, color=color, label=f"{CATEGORY_INFO[category]['label']} (n={len(data)})")
    ax.set(xlabel="Percent energy outside the PCA PC1-PC2 box [%]", ylabel="Counts")
    ax.set_title("PCA containment by particle type")
    ax.set_xlim(0.0, 100.0)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.text(
        0.02,
        0.96,
        "Bin width: 1%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.6"},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "outside_energy_percent_hist_1pct_bins_combined.png", dpi=220)
    plt.close(fig)

    # Limit only the rendered scatter density; all tracks remain in CSV files
    # and in every histogram/statistic.  Sampling is deterministic per category.
    sampled = []
    for category in categories:
        category_data = summary[summary["particle_type"].eq(category)]
        sampled.append(category_data.sample(min(len(category_data), 20_000), random_state=12345))
    scatter_data = pd.concat(sampled, ignore_index=True)

    for x_column, x_label, filename, title in [
        ("track_length_cm", "G4 path length [cm]", "outside_energy_percent_vs_track_length_combined.png", "PCA leakage vs path length"),
        ("pc1_pc2_explained_variance_percent", "Variance explained by PC1 + PC2 [%]", "outside_energy_percent_vs_pc12_variance_combined.png", "PCA leakage vs two-component variance"),
    ]:
        fig, ax = plt.subplots(figsize=(10.0, 7.0))
        for color, category in zip(colors, categories):
            data = scatter_data[scatter_data["particle_type"].eq(category)]
            ax.scatter(data[x_column], data["outside_percent"], s=8, alpha=0.22, linewidths=0, color=color, label=CATEGORY_INFO[category]["label"], rasterized=True)
        ax.set(xlabel=x_label, ylabel="Energy outside PCA box [%]")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    variance_bins = np.linspace(0.0, 100.0, 51)
    for color, category in zip(colors, categories):
        data = summary[summary["particle_type"].eq(category)]
        ax.hist(100.0 * data["pc1_explained_variance"], bins=variance_bins, histtype="step", linewidth=1.5, color=color, label=CATEGORY_INFO[category]["label"])
    ax.set(xlabel="PC1 explained variance [%]", ylabel="Counts")
    ax.set_title("PC1 explained variance by particle type")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.text(
        0.02,
        0.96,
        "Bin width: 2%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.6"},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "pca_explained_variance_hist_combined.png", dpi=220)
    plt.close(fig)


def analyze_tracks(particles, g4_steps, containment_quantile, displays_per_particle):
    """Analyze every eligible track and retain only records needed for displays."""
    particle_index = particle_lookup_table(particles)
    rows = []
    skipped = Counter()
    first_records = defaultdict(list)
    highest_heaps = defaultdict(list)
    heap_sequence = 0

    charged_steps = g4_steps[g4_steps["PDG"].isin(ANALYZED_PDGS)]
    grouped = charged_steps.groupby(["event", "ParticleID", "PDG"], sort=True)
    total_groups = grouped.ngroups
    print(f"Analyzing {total_groups:,} charged G4 track group(s) ...")

    for group_number, ((event_id, track_id, pdg), steps) in enumerate(grouped, start=1):
        if group_number % 50_000 == 0:
            print(f"  Processed {group_number:,} / {total_groups:,} G4 track groups")
        if len(steps) < 3:
            skipped["fewer than three nonzero-length G4 steps"] += 1
            continue

        key = (int(event_id), int(track_id))
        if key not in particle_index.index:
            skipped["missing particle-table metadata"] += 1
            continue
        particle = particle_index.loc[key]
        if isinstance(particle, pd.DataFrame):
            particle = particle.iloc[0]
        category = category_for_particle(pdg, particle["particle_parent_track_id"])
        if category is None:
            skipped["uncategorized PDG"] += 1
            continue

        try:
            result = analyze_primary_kaon_track(steps, containment_quantile)
        except Exception as exc:
            skipped[str(exc)] += 1
            continue

        row = {
            "event": int(event_id),
            "track_id": int(track_id),
            "particle_type": category,
            "particle_label": CATEGORY_INFO[category]["label"],
            "pdg": int(pdg),
            "parent_track_id": int(particle["particle_parent_track_id"]),
            "n_g4_steps": len(result["steps"]),
            "total_energy_MeV": result["total_energy_MeV"],
            "inside_energy_MeV": result["inside_energy_MeV"],
            "outside_energy_MeV": result["outside_energy_MeV"],
            "outside_percent": result["outside_percent"],
            "track_length_cm": result["track_length_cm"],
            "pc1_explained_variance": float(result["explained"][0]),
            "pc2_explained_variance": float(result["explained"][1]),
            "pc3_explained_variance": float(result["explained"][2]),
            "pc1_pc2_explained_variance": float(np.sum(result["explained"][:2])),
            "pc1_pc2_explained_variance_percent": float(100.0 * np.sum(result["explained"][:2])),
            "pc1_low_cm": result["box"]["pc1_low"],
            "pc1_high_cm": result["box"]["pc1_high"],
            "pc2_low_cm": result["box"]["pc2_low"],
            "pc2_high_cm": result["box"]["pc2_high"],
            "particle_initial_energy_MeV": float(particle["particle_initial_energy"]),
            "particle_final_kinetic_energy_MeV": float(particle["particle_final_kinetic_energy"]),
            "particle_decay_flag": int(particle["particle_decay_flag"]),
            "endpoint_status": endpoint_status(particle),
        }
        rows.append(row)

        record = {
            "event": int(event_id),
            "track_id": int(track_id),
            "particle_type": category,
            "particle": particle.copy(),
            "result": result,
        }
        if CATEGORY_INFO[category]["other"] and len(first_records[category]) < displays_per_particle:
            first_records[category].append(record)

        if CATEGORY_INFO[category]["other"] and displays_per_particle > 0:
            heap = highest_heaps[category]
            heap_item = (result["outside_percent"], heap_sequence, record)
            heap_sequence += 1
            if len(heap) < displays_per_particle:
                heapq.heappush(heap, heap_item)
            elif heap_item[0] > heap[0][0]:
                heapq.heapreplace(heap, heap_item)

    # Analyze each complete electron shower as one object.  All G4 deposits
    # carrying the same earliest-electron root are included, irrespective of
    # the descendant particle's own PDG code.
    shower_steps = g4_steps[g4_steps["electron_shower_root_track_id"].notna()].copy()
    shower_steps["electron_shower_root_track_id"] = shower_steps[
        "electron_shower_root_track_id"
    ].astype(int)
    shower_groups = shower_steps.groupby(
        ["event", "electron_shower_root_track_id"],
        sort=True,
    )
    total_showers = shower_groups.ngroups
    print(f"Analyzing {total_showers:,} complete electron shower group(s) ...")

    for shower_number, ((event_id, root_track_id), steps) in enumerate(
        shower_groups,
        start=1,
    ):
        if shower_number % 50_000 == 0:
            print(f"  Processed {shower_number:,} / {total_showers:,} electron showers")
        if len(steps) < 3:
            skipped["electron shower: fewer than three nonzero-length G4 steps"] += 1
            continue

        key = (int(event_id), int(root_track_id))
        if key not in particle_index.index:
            skipped["electron shower: missing root-electron metadata"] += 1
            continue
        particle = particle_index.loc[key]
        if isinstance(particle, pd.DataFrame):
            particle = particle.iloc[0]

        try:
            result = analyze_primary_kaon_track(steps, containment_quantile)
        except Exception as exc:
            skipped[f"electron shower: {exc}"] += 1
            continue

        category = "electron_showers"
        rows.append(
            {
                "event": int(event_id),
                "track_id": int(root_track_id),
                "particle_type": category,
                "particle_label": CATEGORY_INFO[category]["label"],
                "pdg": 11,
                "parent_track_id": int(particle["particle_parent_track_id"]),
                "n_g4_steps": len(result["steps"]),
                "n_descendant_tracks": int(steps["ParticleID"].nunique()),
                "total_energy_MeV": result["total_energy_MeV"],
                "inside_energy_MeV": result["inside_energy_MeV"],
                "outside_energy_MeV": result["outside_energy_MeV"],
                "outside_percent": result["outside_percent"],
                "track_length_cm": result["track_length_cm"],
                "pc1_explained_variance": float(result["explained"][0]),
                "pc2_explained_variance": float(result["explained"][1]),
                "pc3_explained_variance": float(result["explained"][2]),
                "pc1_pc2_explained_variance": float(np.sum(result["explained"][:2])),
                "pc1_pc2_explained_variance_percent": float(
                    100.0 * np.sum(result["explained"][:2])
                ),
                "pc1_low_cm": result["box"]["pc1_low"],
                "pc1_high_cm": result["box"]["pc1_high"],
                "pc2_low_cm": result["box"]["pc2_low"],
                "pc2_high_cm": result["box"]["pc2_high"],
                "particle_initial_energy_MeV": float(particle["particle_initial_energy"]),
                "particle_final_kinetic_energy_MeV": float(
                    particle["particle_final_kinetic_energy"]
                ),
                "particle_decay_flag": int(particle["particle_decay_flag"]),
                "endpoint_status": endpoint_status(particle),
            }
        )

        record = {
            "event": int(event_id),
            "track_id": int(root_track_id),
            "particle_type": category,
            "particle": particle.copy(),
            "result": result,
        }
        if len(first_records[category]) < displays_per_particle:
            first_records[category].append(record)
        if displays_per_particle > 0:
            heap = highest_heaps[category]
            heap_item = (result["outside_percent"], heap_sequence, record)
            heap_sequence += 1
            if len(heap) < displays_per_particle:
                heapq.heappush(heap, heap_item)
            elif heap_item[0] > heap[0][0]:
                heapq.heapreplace(heap, heap_item)

    summary = pd.DataFrame(rows).sort_values(["particle_type", "event", "track_id"]).reset_index(drop=True)
    return summary, skipped, first_records, highest_heaps


def write_outputs(summary, skipped, first_records, highest_heaps, inventory, args):
    """Write per-particle, display, no-data, and combined output hierarchies."""
    event_set_dir = Path(args.output_event_set_dir).expanduser().resolve()
    other_root = event_set_dir / "other_particles"
    display_root = other_root / "event_displays"
    combined_root = event_set_dir / "combined"
    kaon_electron_shower_root = event_set_dir / "kaon_e-_showers"
    other_root.mkdir(parents=True, exist_ok=True)
    display_root.mkdir(parents=True, exist_ok=True)
    combined_root.mkdir(parents=True, exist_ok=True)
    kaon_electron_shower_root.mkdir(parents=True, exist_ok=True)

    for category, info in CATEGORY_INFO.items():
        if not info["other"]:
            continue
        category_summary = summary[summary["particle_type"].eq(category)].copy()
        category_dir = other_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        if category_summary.empty:
            (category_dir / "no_valid_pca_tracks.txt").write_text(
                f"No {info['label']} tracks had at least three positive-energy, nonzero-length G4 steps.\n"
            )
            continue
        category_summary.to_csv(category_dir / f"pca_track_summary_{category}.csv", index=False)
        write_category_summary(
            category_summary,
            category,
            category_dir / f"pca_containment_summary_{category}.txt",
            args.containment_quantile,
        )
        plot_category(category_summary, category, category_dir)

        first_dir = display_root / category / "first_events"
        highest_dir = display_root / category / "highest_outside_energy"
        first_dir.mkdir(parents=True, exist_ok=True)
        highest_dir.mkdir(parents=True, exist_ok=True)
        for rank, record in enumerate(first_records[category], start=1):
            path = first_dir / f"{category}_event_{record['event']:04d}_track_{record['track_id']}_pca_box.png"
            save_event_display(record, path, args.containment_quantile)
        ranked = sorted(highest_heaps[category], key=lambda item: item[0], reverse=True)
        for rank, (_, _, record) in enumerate(ranked, start=1):
            path = highest_dir / f"rank_{rank:02d}_{category}_event_{record['event']:04d}_track_{record['track_id']}_pca_box.png"
            save_event_display(record, path, args.containment_quantile)

    no_pca_lines = [
        "Particle-table species without direct PCA tracks",
        "===============================================",
        "",
        "Neutral particles generally leave no direct positive-energy G4 steps in this output.",
        "Their charged daughters and secondary tracks are included in the charged-particle categories.",
        "Nuclear-ion PDG codes and neutrinos are intentionally excluded.",
        "",
        "PDG, particle, particle-table rows",
    ]
    for pdg, label in NO_DIRECT_PCA_SPECIES.items():
        no_pca_lines.append(f"{pdg}, {label}, {inventory.get(pdg, 0)}")
    (other_root / "species_without_direct_pca.txt").write_text("\n".join(no_pca_lines) + "\n")
    (other_root / "skipped_track_summary.txt").write_text(
        "\n".join(f"{reason}: {count}" for reason, count in skipped.most_common()) + "\n"
    )

    summary.to_csv(combined_root / "pca_track_summary_combined.csv", index=False)
    plot_combined(summary, combined_root)
    counts = summary.groupby(["particle_type", "particle_label"]).size()
    combined_lines = [
        "Combined particle PCA containment summary",
        "=========================================",
        "",
        f"Total tracks analyzed: {len(summary)}",
        f"PCA box energy-quantile target: {100.0 * args.containment_quantile:.3g}%",
        "",
        "Analyzed tracks by category:",
    ]
    for (category, label), count in counts.items():
        data = summary[summary["particle_type"].eq(category)]["outside_percent"]
        combined_lines.append(
            f"  {label}: {count} tracks, mean outside {data.mean():.6g}%, median {data.median():.6g}%"
        )
    (combined_root / "pca_containment_summary_combined.txt").write_text(
        "\n".join(combined_lines) + "\n"
    )

    # Save a focused primary-K+ versus complete-electron-shower comparison.
    kaon_electron_summary = summary[
        summary["particle_type"].isin(["primary_k_plus", "electron_showers"])
    ].copy()
    kaon_electron_summary.to_csv(
        kaon_electron_shower_root / "pca_track_summary_kaon_electron_showers.csv",
        index=False,
    )
    plot_combined(kaon_electron_summary, kaon_electron_shower_root)
    focused_lines = [
        "Primary K+ and electron-shower PCA summary",
        "==========================================",
        "",
        f"PCA box energy-quantile target: {100.0 * args.containment_quantile:.3g}%",
        "",
    ]
    for category in ["primary_k_plus", "electron_showers"]:
        data = kaon_electron_summary[
            kaon_electron_summary["particle_type"].eq(category)
        ]["outside_percent"]
        focused_lines.append(
            f"{CATEGORY_INFO[category]['label']}: {len(data)} objects, "
            f"mean outside {data.mean():.6g}%, median {data.median():.6g}%"
        )
    (kaon_electron_shower_root / "pca_containment_summary_kaon_electron_showers.txt").write_text(
        "\n".join(focused_lines) + "\n"
    )


def run(args):
    """Load the aggregate truth files, run every track PCA, and save outputs."""
    input_dir = Path(args.input_dir).expanduser().resolve()
    particle_path = input_dir / "particles_output.txt"
    g4_path = input_dir / "g4_output.txt"
    if not particle_path.is_file() or not g4_path.is_file():
        raise FileNotFoundError("Expected particles_output.txt and g4_output.txt in --input-dir")

    print(f"Reading particle metadata from {particle_path.name} ...")
    particles, genealogy, inventory = read_particle_table(particle_path)
    print("Building non-overlapping electron-shower ancestry groups ...")
    shower_membership = build_electron_shower_membership(genealogy)
    print(f"Reading charged-particle and electron-shower G4 steps from {g4_path.name} ...")
    g4_steps = read_g4_steps(g4_path, shower_membership)
    summary, skipped, first_records, highest_heaps = analyze_tracks(
        particles,
        g4_steps,
        args.containment_quantile,
        args.event_displays_per_particle,
    )
    if summary.empty:
        raise RuntimeError("No tracks produced a valid PCA result.")
    write_outputs(summary, skipped, first_records, highest_heaps, inventory, args)
    print(f"Analyzed {len(summary):,} tracks across {summary['particle_type'].nunique()} categories.")
    print(f"Saved other-particle outputs under: {Path(args.output_event_set_dir).resolve() / 'other_particles'}")
    print(f"Saved combined outputs under: {Path(args.output_event_set_dir).resolve() / 'combined'}")
    print(
        "Saved primary-K+ versus electron-shower outputs under: "
        f"{Path(args.output_event_set_dir).resolve() / 'kaon_e-_showers'}"
    )


def build_parser():
    """Define command-line controls for input, output, PCA box, and displays."""
    parser = argparse.ArgumentParser(description="PCA containment study for charged detector particles.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-event-set-dir", default=DEFAULT_EVENT_SET_DIR)
    parser.add_argument("--containment-quantile", type=float, default=0.98)
    parser.add_argument("--event-displays-per-particle", type=int, default=5)
    return parser


def main():
    """Parse arguments and run the other-particle PCA analysis."""
    args = build_parser().parse_args()
    if not 0.0 < args.containment_quantile <= 1.0:
        raise ValueError("--containment-quantile must be in (0, 1].")
    if args.event_displays_per_particle < 0:
        raise ValueError("--event-displays-per-particle must be nonnegative.")
    run(args)


if __name__ == "__main__":
    main()

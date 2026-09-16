#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import uproot


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_BIN_WIDTH_CM = 1.0
W_ION_MEV = 23.6e-6
KAON_PDG = 321
MODIFIED_BOX_A = 0.930
MODIFIED_BOX_B_OVER_E = 0.212 / 0.5

PDG_LABELS = {
    -13: "mu+",
    13: "mu-",
    -11: "e+",
    11: "e-",
    22: "gamma",
    211: "pi+",
    -211: "pi-",
    321: "K+",
    -321: "K-",
    2112: "n",
    2212: "p",
    1000180380: "Ar38",
    1000180390: "Ar39",
    1000180400: "Ar40",
}


def find_required_file(input_path, file_name):
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_file():
        if input_path.name != file_name:
            raise FileNotFoundError(f"Input file is not named {file_name}: {input_path}")
        return input_path

    direct = input_path / file_name
    if direct.exists():
        return direct

    matches = sorted(input_path.rglob(file_name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {file_name} found under {input_path}")
    raise FileExistsError(
        f"Found multiple {file_name} files; pass the event folder directly:\n"
        + "\n".join(str(match) for match in matches)
    )


def find_metadata_root(event_dir, explicit_path=None):
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Metadata ROOT file does not exist: {path}")
        return path

    candidates = []
    for path in event_dir.parent.glob("*.root"):
        if event_dir.name.startswith(path.stem):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "Could not find the matching RTD ROOT file beside the event folder. "
            "Pass it with --metadata-root."
        )
    return max(candidates, key=lambda path: len(path.stem))


def read_metadata(root_path):
    names = [
        "drift_velocity",
        "pixel_size",
        "reset_threshold",
        "electron_lifetime",
        "charge_loss",
        "recombination",
    ]
    with uproot.open(root_path) as root_file:
        metadata = root_file["metadata"]
        values = {}
        for name in names:
            if name not in metadata:
                raise KeyError(f"Missing metadata branch '{name}' in {root_path}")
            value = metadata[name].array(library="np")[0]
            values[name] = value.item() if hasattr(value, "item") else value
    return values


def read_particles(path):
    columns = [
        "event",
        "particle_track_id",
        "particle_parent_track_id",
        "particle_pdg_code",
        "particle_initial_x",
        "particle_initial_y",
        "particle_initial_z",
        "particle_final_x",
        "particle_final_y",
        "particle_final_z",
    ]
    particles = pd.read_csv(path, usecols=columns)
    for column in columns:
        particles[column] = pd.to_numeric(particles[column], errors="coerce")
    particles = particles.dropna(subset=columns).copy()
    for column in columns[:4]:
        particles[column] = particles[column].astype(int)
    particles = particles.drop_duplicates(["event", "particle_track_id"], keep="first")
    particles["is_primary_kaon"] = (
        particles["particle_pdg_code"].abs().eq(KAON_PDG)
        & particles["particle_parent_track_id"].eq(0)
    )
    return particles


def parse_integer_list(value):
    text = str(value).strip().strip("[]")
    # Pandas shortened a small number of very long truth-ID lists with "..."
    # when the per-event text files were originally written. Their weights can
    # no longer be aligned safely, so omit those mixed resets.
    if not text or "..." in text:
        return np.empty(0, dtype=np.int64)
    return np.fromstring(text, sep=",", dtype=np.int64)


def read_reset_contributions(path, particles, metadata, chunk_size=100_000):
    particle_columns = [
        "event",
        "particle_track_id",
        "particle_parent_track_id",
        "particle_pdg_code",
        "particle_initial_x",
        "particle_initial_y",
        "particle_initial_z",
        "particle_final_x",
        "particle_final_y",
        "particle_final_z",
        "is_primary_kaon",
    ]
    particle_lookup = particles[particle_columns]
    output_chunks = []
    reset_rows = 0

    usecols = [
        "event",
        "pixel_x",
        "pixel_y",
        "reset_time",
        "MC_TrackIDs",
        "MC_Weights",
    ]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        reset_rows += len(chunk)
        id_lists = [parse_integer_list(value) for value in chunk["MC_TrackIDs"]]
        weight_lists = [parse_integer_list(value) for value in chunk["MC_Weights"]]
        lengths = np.fromiter(
            (min(len(ids), len(weights)) for ids, weights in zip(id_lists, weight_lists)),
            dtype=np.int64,
            count=len(chunk),
        )
        keep_rows = lengths > 0
        if not np.any(keep_rows):
            continue

        selected = chunk.loc[keep_rows].reset_index(drop=True)
        selected_lengths = lengths[keep_rows]
        selected_ids = [ids[:length] for ids, length in zip(np.asarray(id_lists, dtype=object)[keep_rows], selected_lengths)]
        selected_weights = [weights[:length] for weights, length in zip(np.asarray(weight_lists, dtype=object)[keep_rows], selected_lengths)]

        expanded = pd.DataFrame(
            {
                "event": np.repeat(selected["event"].to_numpy(dtype=np.int64), selected_lengths),
                "particle_track_id": np.concatenate(selected_ids),
                "pixel_x": np.repeat(selected["pixel_x"].to_numpy(dtype=float), selected_lengths),
                "pixel_y": np.repeat(selected["pixel_y"].to_numpy(dtype=float), selected_lengths),
                "reset_time": np.repeat(selected["reset_time"].to_numpy(dtype=float), selected_lengths),
                "electrons": np.concatenate(selected_weights).astype(float),
            }
        )
        expanded = expanded[expanded["electrons"] > 0.0]
        expanded = expanded.merge(
            particle_lookup,
            how="inner",
            on=["event", "particle_track_id"],
            validate="many_to_one",
        )
        if expanded.empty:
            continue

        expanded["x_cm"] = (expanded["pixel_x"] + 0.5) * float(metadata["pixel_size"])
        expanded["y_cm"] = (expanded["pixel_y"] + 0.5) * float(metadata["pixel_size"])
        expanded["z_cm"] = expanded["reset_time"] * float(metadata["drift_velocity"])

        if bool(metadata["charge_loss"]):
            lifetime = float(metadata["electron_lifetime"])
            if lifetime <= 0.0:
                raise ValueError("electron_lifetime must be positive when charge_loss is enabled")
            expanded["electrons"] *= np.exp(expanded["reset_time"] / lifetime)

        output_chunks.append(expanded)

    if not output_chunks:
        raise ValueError("No reset charge could be matched to particles_output.txt")
    return pd.concat(output_chunks, ignore_index=True), reset_rows


def modified_box_inverse(collected_mev_per_cm):
    values = np.asarray(collected_mev_per_cm, dtype=float)
    exponent = np.clip(MODIFIED_BOX_B_OVER_E * values, None, 700.0)
    return (np.exp(exponent) - MODIFIED_BOX_A) / MODIFIED_BOX_B_OVER_E


def reconstruct_track_segments(contributions, bin_width_cm, recombination_enabled):
    if bin_width_cm <= 0.0:
        raise ValueError("bin_width_cm must be positive")

    rows = []
    group_columns = ["event", "particle_track_id"]
    for (event, track_id), track in contributions.groupby(group_columns, sort=False):
        if len(track) < 2:
            continue

        points = track[["x_cm", "y_cm", "z_cm"]].to_numpy(dtype=float)
        weights = track["electrons"].to_numpy(dtype=float)
        total_weight = weights.sum()
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            continue

        center = np.average(points, axis=0, weights=weights)
        centered = points - center
        covariance = (centered * weights[:, None]).T @ centered / total_weight
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, np.argmax(eigenvalues)]

        first = track.iloc[0]
        truth_direction = np.array(
            [
                first["particle_final_x"] - first["particle_initial_x"],
                first["particle_final_y"] - first["particle_initial_y"],
                first["particle_final_z"] - first["particle_initial_z"],
            ]
        )
        if np.dot(axis, truth_direction) < 0.0:
            axis = -axis

        projection = centered @ axis
        projection -= projection.min()
        track_length = float(projection.max())
        if not np.isfinite(track_length) or track_length < 0.5 * bin_width_cm:
            continue

        bin_index = np.minimum(
            np.floor(projection / bin_width_cm).astype(int),
            max(int(np.ceil(track_length / bin_width_cm)) - 1, 0),
        )
        occupied_bins, inverse = np.unique(bin_index, return_inverse=True)
        electron_sums = np.bincount(inverse, weights=weights)
        contribution_counts = np.bincount(inverse)
        for occupied_bin, electron_sum, contribution_count in zip(
            occupied_bins, electron_sums, contribution_counts
        ):
            low = occupied_bin * bin_width_cm
            high = min((occupied_bin + 1) * bin_width_cm, track_length)
            # Normalize edge bins to the requested sampling length. Otherwise a
            # reset at the fitted endpoint can be divided by a nearly zero tail.
            path_length = min(bin_width_cm, track_length)
            if path_length <= 0.0:
                continue
            collected = electron_sum * W_ION_MEV / path_length
            dedx = (
                float(modified_box_inverse(collected))
                if recombination_enabled
                else collected
            )
            if not np.isfinite(dedx) or dedx <= 0.0:
                continue
            midpoint = 0.5 * (low + high)
            rows.append(
                {
                    "event": int(event),
                    "ParticleID": int(track_id),
                    "PDG": int(first["particle_pdg_code"]),
                    "particle_parent_track_id": int(first["particle_parent_track_id"]),
                    "is_primary_kaon": bool(first["is_primary_kaon"]),
                    "residual_range_cm": track_length - midpoint,
                    "dEdx_MeV_per_cm": dedx,
                    "collected_charge_MeV_per_cm": collected,
                    "electrons": electron_sum,
                    "path_length_cm": path_length,
                    "track_length_cm": track_length,
                    "n_reset_contributions": int(contribution_count),
                }
            )

    if not rows:
        raise ValueError("No tracks had enough reset points to reconstruct dE/dx")
    return pd.DataFrame(rows)


def bethe_bloch_lar(mass_mev, max_range_cm, n_points=2500):
    k_constant = 0.307075
    electron_mass_mev = 0.51099895
    z_over_a_argon = 18.0 / 39.948
    density_lar_g_cm3 = 1.396
    mean_excitation_mev = 188.0e-6

    kinetic_energy = np.geomspace(0.1, 5000.0, n_points)
    gamma = 1.0 + kinetic_energy / mass_mev
    beta2 = np.clip(1.0 - 1.0 / gamma**2, 1e-8, None)
    mass_ratio = electron_mass_mev / mass_mev
    t_max = (
        2.0 * electron_mass_mev * beta2 * gamma**2
        / (1.0 + 2.0 * gamma * mass_ratio + mass_ratio**2)
    )
    log_arg = 2.0 * electron_mass_mev * beta2 * gamma**2 * t_max / mean_excitation_mev**2
    stopping_mass = (
        k_constant
        * z_over_a_argon
        / beta2
        * (0.5 * np.log(np.clip(log_arg, 1.0, None)) - beta2)
    )
    stopping_power = np.clip(stopping_mass * density_lar_g_cm3, 1e-9, None)
    delta_energy = np.diff(kinetic_energy)
    inverse_midpoint = 0.5 * (1.0 / stopping_power[1:] + 1.0 / stopping_power[:-1])
    residual_range = np.concatenate([[0.0], np.cumsum(delta_energy * inverse_midpoint)])
    mask = residual_range <= max_range_cm
    return residual_range[mask], stopping_power[mask]


def pdg_label(pdg):
    return PDG_LABELS.get(int(pdg), str(int(pdg)))


def make_primary_kaon_plot(
    binned,
    output_path,
    number_events,
    bin_width_cm,
    x_max_cm=None,
    y_max_mev_per_cm=None,
):
    primary = binned[binned["is_primary_kaon"]]
    if primary.empty:
        raise ValueError("No primary-kaon reset tracks were reconstructed")

    max_range = max(float(primary["residual_range_cm"].max()), 1.0)
    theory_range, theory_dedx = bethe_bloch_lar(493.677, max_range)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(
        primary["residual_range_cm"],
        primary["dEdx_MeV_per_cm"],
        s=9,
        alpha=0.4,
        linewidths=0,
        color="tab:purple",
        label="primary K+ reset segments",
        rasterized=True,
    )
    ax.plot(
        theory_range,
        theory_dedx,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Bethe-Bloch K in LAr",
    )
    ax.set_xlim(left=0.0, right=x_max_cm)
    ax.set_ylim(bottom=0.0, top=y_max_mev_per_cm)
    ax.set_xlabel("Residual range [cm]")
    ax.set_ylabel("Reconstructed dE/dx [MeV/cm]")
    ax.set_title(f"Primary kaon reset reconstruction ({number_events} events)")
    ax.text(
        0.98,
        0.96,
        "\n".join(
            [
                f"Bin width: {bin_width_cm:g} cm",
                *([f"x max: {x_max_cm:g} cm"] if x_max_cm is not None else []),
                *(
                    [f"y max: {y_max_mev_per_cm:g} MeV/cm"]
                    if y_max_mev_per_cm is not None
                    else []
                ),
                f"Tracks: {primary[['event', 'ParticleID']].drop_duplicates().shape[0]}",
                f"Segments: {len(primary)}",
                "Truth-assisted track identity",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.6"},
    )
    ax.grid(True, alpha=0.25)
    legend = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.76), framealpha=0.9)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_all_particles_plot(binned, output_path, number_events, bin_width_cm):
    counts = binned["PDG"].value_counts()
    top_pdgs = list(counts.head(10).index)
    fig, ax = plt.subplots(figsize=(10, 7))

    other = binned[~binned["PDG"].isin(top_pdgs)]
    if not other.empty:
        ax.scatter(
            other["residual_range_cm"],
            other["dEdx_MeV_per_cm"],
            s=5,
            alpha=0.12,
            color="0.65",
            linewidths=0,
            label=f"other PDGs ({other['PDG'].nunique()})",
            rasterized=True,
        )

    cmap = plt.get_cmap("tab10")
    for index, pdg in enumerate(top_pdgs):
        group = binned[binned["PDG"] == pdg]
        color = "tab:purple" if abs(int(pdg)) == KAON_PDG else cmap(index % 10)
        ax.scatter(
            group["residual_range_cm"],
            group["dEdx_MeV_per_cm"],
            s=7,
            alpha=0.32,
            linewidths=0,
            color=color,
            label=f"{pdg_label(pdg)} ({len(group)})",
            rasterized=True,
        )

    ax.set_yscale("log")
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Residual range [cm]")
    ax.set_ylabel("Reconstructed dE/dx [MeV/cm]")
    ax.set_title(f"All-particle reset reconstruction ({number_events} events)")
    ax.text(
        0.98,
        0.96,
        "\n".join(
            [
                f"Bin width: {bin_width_cm:g} cm",
                f"Tracks: {binned[['event', 'ParticleID']].drop_duplicates().shape[0]}",
                f"Segments: {len(binned)}",
                f"PDG species: {binned['PDG'].nunique()}",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.6"},
    )
    ax.grid(True, alpha=0.25)
    legend = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.78), framealpha=0.9)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct dE/dx versus residual range from QPix reset signals."
    )
    parser.add_argument(
        "input_path",
        help="Generated event folder containing resets_output.txt and particles_output.txt.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        help="Matching RTD ROOT file. By default it is found beside the event folder.",
    )
    parser.add_argument(
        "--bin-width-cm",
        type=float,
        default=DEFAULT_BIN_WIDTH_CM,
        help=f"Track-coordinate bin width. Default: {DEFAULT_BIN_WIDTH_CM:g} cm.",
    )
    args = parser.parse_args()

    resets_path = find_required_file(args.input_path, "resets_output.txt")
    particles_path = find_required_file(args.input_path, "particles_output.txt")
    metadata_root = find_metadata_root(resets_path.parent, args.metadata_root)
    metadata = read_metadata(metadata_root)
    particles = read_particles(particles_path)
    contributions, reset_rows = read_reset_contributions(
        resets_path, particles, metadata
    )
    binned = reconstruct_track_segments(
        contributions,
        args.bin_width_cm,
        recombination_enabled=bool(metadata["recombination"]),
    )

    number_events = int(contributions["event"].max()) + 1
    output_dir = SCRIPT_DIR / "dEdx_resets" / f"kaon_decay_{number_events}_events"
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_plot = output_dir / f"primary_kaons_resets_dEdx_vs_residual_range_{number_events}_events.png"
    primary_zoom_plot = output_dir / f"primary_kaons_resets_dEdx_vs_residual_range_xmax10cm_ymax40MeVcm_{number_events}_events.png"
    all_plot = output_dir / f"all_particles_resets_dEdx_vs_residual_range_{number_events}_events.png"
    binned_csv = output_dir / f"resets_dEdx_binned_segments_{number_events}_events.csv"
    summary_path = output_dir / "reconstruction_summary.txt"

    make_primary_kaon_plot(binned, primary_plot, number_events, args.bin_width_cm)
    make_primary_kaon_plot(
        binned,
        primary_zoom_plot,
        number_events,
        args.bin_width_cm,
        x_max_cm=10.0,
        y_max_mev_per_cm=40.0,
    )
    make_all_particles_plot(binned, all_plot, number_events, args.bin_width_cm)
    binned.to_csv(binned_csv, index=False)
    summary_path.write_text(
        "\n".join(
            [
                f"resets_output: {resets_path}",
                f"particles_output: {particles_path}",
                f"metadata ROOT: {metadata_root}",
                f"events: {number_events}",
                f"reset rows: {reset_rows}",
                f"truth-weighted reset contributions: {len(contributions)}",
                f"reconstructed tracks: {binned[['event', 'ParticleID']].drop_duplicates().shape[0]}",
                f"reconstructed segments: {len(binned)}",
                f"primary-kaon tracks: {binned.loc[binned['is_primary_kaon'], ['event', 'ParticleID']].drop_duplicates().shape[0]}",
                f"pixel size [cm]: {metadata['pixel_size']}",
                f"drift velocity [cm/s]: {metadata['drift_velocity']}",
                f"reset threshold [electrons]: {metadata['reset_threshold']}",
                f"charge loss enabled: {bool(metadata['charge_loss'])}",
                f"recombination enabled: {bool(metadata['recombination'])}",
                "track model: charge-weighted PCA axis through individual reset points",
            ]
        )
        + "\n"
    )

    print(f"Read reset rows: {reset_rows}")
    print(f"Truth-weighted reset contributions: {len(contributions)}")
    print(f"Number of events: {number_events}")
    print(f"Reconstructed dE/dx segments: {len(binned)}")
    print(f"Saved primary-kaon plot: {primary_plot}")
    print(f"Saved primary-kaon zoom plot: {primary_zoom_plot}")
    print(f"Saved all-particle plot: {all_plot}")
    print(f"Saved binned segments: {binned_csv}")
    print(f"Saved reconstruction summary: {summary_path}")


if __name__ == "__main__":
    main()

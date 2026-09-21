#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dedx_midpoint_binning import bin_steps_by_midpoint


DEFAULT_BIN_WIDTH_CM = 1.0
DEFAULT_ENDPOINT_BIN_WIDTH_CM = 0.05
DEFAULT_ENDPOINT_MAX_CM = 1.0
DEFAULT_ZOOM_X_MAX_CM = 10.0
DEFAULT_ZOOM_Y_MAX_MEV_PER_CM = 40.0
KAON_PDG = 321


def find_required_file(input_path, file_name):
    path = Path(input_path).expanduser()
    if path.is_file():
        if path.name != file_name:
            raise FileNotFoundError(f"Input file is not named {file_name}: {path}")
        return path

    direct_path = path / file_name
    if direct_path.exists():
        return direct_path

    matches = sorted(path.rglob(file_name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {file_name} found under {path}")
    raise FileExistsError(
        f"Found multiple {file_name} files; pass the event folder directly:\n"
        + "\n".join(str(match) for match in matches)
    )


def read_g4_hits(g4_path):
    required_columns = [
        "xi",
        "xf",
        "yi",
        "yf",
        "zi",
        "zf",
        "ti",
        "E",
        "ParticleID",
        "PDG",
    ]
    event_columns = ["event", "events"]
    df = pd.read_csv(
        g4_path, usecols=lambda column: column in required_columns + event_columns
    )
    if "event" not in df.columns and "events" in df.columns:
        df = df.rename(columns={"events": "event"})
    required_columns = ["event"] + required_columns
    missing_columns = set(required_columns).difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required column(s) in {g4_path}: {missing}")

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns).copy()
    df["event"] = df["event"].astype(int)
    df["ParticleID"] = df["ParticleID"].astype(int)
    df["PDG"] = df["PDG"].astype(int)
    return df


def read_particles(particles_path):
    required_columns = [
        "event",
        "particle_track_id",
        "particle_parent_track_id",
        "particle_pdg_code",
        "particle_process_key",
        "particle_initial_energy",
        "particle_final_kinetic_energy",
        "particle_decay_flag",
    ]
    df = pd.read_csv(
        particles_path, usecols=lambda column: column in required_columns
    )
    missing_columns = set(required_columns).difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required column(s) in {particles_path}: {missing}")

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns).copy()
    int_columns = [
        "event",
        "particle_track_id",
        "particle_parent_track_id",
        "particle_pdg_code",
        "particle_process_key",
        "particle_decay_flag",
    ]
    for column in int_columns:
        df[column] = df[column].astype(int)
    return df


def event_count_from_last_row(g4_hits):
    return int(g4_hits["event"].iloc[-1]) + 1


def classify_kaons(particles):
    kaons = particles[particles["particle_pdg_code"].abs() == KAON_PDG].copy()
    primary_kaons = kaons[kaons["particle_parent_track_id"] == 0].copy()

    primary_keys = set(
        zip(primary_kaons["event"], primary_kaons["particle_track_id"])
    )
    daughter_mask = kaons.apply(
        lambda row: (row["event"], row["particle_parent_track_id"]) in primary_keys,
        axis=1,
    )
    daughter_kaons = kaons[daughter_mask].copy()

    primary_kaons["kaon_category"] = "primary kaon"
    daughter_kaons["kaon_category"] = "daughter kaon"
    selected = pd.concat([primary_kaons, daughter_kaons], ignore_index=True)
    return primary_kaons, daughter_kaons, selected


def select_hits_for_particles(g4_hits, particles):
    track_table = particles[
        [
            "event",
            "particle_track_id",
            "particle_parent_track_id",
            "particle_process_key",
            "particle_initial_energy",
            "particle_final_kinetic_energy",
            "particle_decay_flag",
            "kaon_category",
        ]
    ].rename(columns={"particle_track_id": "ParticleID"})

    selected_hits = g4_hits.merge(
        track_table,
        how="inner",
        on=["event", "ParticleID"],
        validate="many_to_one",
    )
    return selected_hits


def add_track_geometry(g4_hits):
    hits = g4_hits.copy()
    hits["ds_cm"] = np.sqrt(
        (hits["xf"] - hits["xi"]) ** 2
        + (hits["yf"] - hits["yi"]) ** 2
        + (hits["zf"] - hits["zi"]) ** 2
    )
    hits = hits[(hits["ds_cm"] > 0.0) & (hits["E"] > 0.0)].copy()
    hits = hits.sort_values(["event", "ParticleID", "ti"], kind="mergesort")

    track_keys = ["event", "ParticleID"]
    hits["s_end_cm"] = hits.groupby(track_keys)["ds_cm"].cumsum()
    hits["s_mid_cm"] = hits["s_end_cm"] - 0.5 * hits["ds_cm"]
    hits["track_length_cm"] = hits.groupby(track_keys)["ds_cm"].transform("sum")
    hits["residual_range_step_cm"] = hits["track_length_cm"] - hits["s_mid_cm"]
    return hits


def make_binned_dedx(
    g4_hits,
    bin_width_cm,
    binning="hybrid",
    endpoint_bin_width_cm=DEFAULT_ENDPOINT_BIN_WIDTH_CM,
    endpoint_max_cm=DEFAULT_ENDPOINT_MAX_CM,
):
    if bin_width_cm <= 0:
        raise ValueError("bin_width_cm must be positive")

    hits = add_track_geometry(g4_hits)
    if binning == "hybrid":
        # Assign each complete step by its midpoint and carry all truth fields
        # through the hybrid endpoint/track-start grouping.
        grouped, diagnostics = bin_steps_by_midpoint(
            hits=hits,
            group_columns=[
                "event",
                "ParticleID",
                "PDG",
                "kaon_category",
                "particle_parent_track_id",
                "particle_process_key",
                "particle_final_kinetic_energy",
                "particle_decay_flag",
            ],
            endpoint_bin_width_cm=endpoint_bin_width_cm,
            endpoint_max_cm=endpoint_max_cm,
            track_bin_width_cm=bin_width_cm,
        )
        # Remove undefined numerical results before plotting or CSV export.
        grouped = grouped.replace([np.inf, -np.inf], np.nan)
        grouped = grouped.dropna(subset=["residual_range_cm", "dEdx_MeV_per_cm"])
        return grouped, diagnostics
    elif binning == "fixed":
        hits["s_bin_cm"] = np.floor(hits["s_mid_cm"] / bin_width_cm) * bin_width_cm
        hits["weighted_s_mid_cm"] = hits["s_mid_cm"] * hits["ds_cm"]
        bin_columns = ["s_bin_cm"]
    else:
        raise ValueError("binning must be 'hybrid' or 'fixed'")

    grouped = (
        hits.groupby(
            [
                "event",
                "ParticleID",
                "PDG",
                "kaon_category",
                "particle_parent_track_id",
                "particle_process_key",
                "particle_final_kinetic_energy",
                "particle_decay_flag",
                *bin_columns,
            ],
            as_index=False,
        )
        .agg(
            energy_MeV=("E", "sum"),
            path_length_cm=("ds_cm", "sum"),
            **(
                {"weighted_residual_range_cm": ("weighted_residual_range_cm", "sum")}
                if binning == "hybrid"
                else {"weighted_s_mid_cm": ("weighted_s_mid_cm", "sum")}
            ),
            **(
                {
                    "range_bin_low_cm": ("range_bin_low_cm", "first"),
                    "range_bin_high_cm": ("range_bin_high_cm", "first"),
                }
                if binning == "hybrid"
                else {}
            ),
            track_length_cm=("track_length_cm", "first"),
            n_steps=("E", "size"),
        )
    )
    grouped = grouped[grouped["path_length_cm"] > 0.0].copy()
    grouped["s_mid_cm"] = grouped["weighted_s_mid_cm"] / grouped["path_length_cm"]
    grouped["residual_range_cm"] = grouped["track_length_cm"] - grouped["s_mid_cm"]
    grouped["dEdx_MeV_per_cm"] = grouped["energy_MeV"] / grouped["path_length_cm"]
    grouped = grouped.replace([np.inf, -np.inf], np.nan)
    grouped = grouped.dropna(subset=["residual_range_cm", "dEdx_MeV_per_cm"])
    diagnostics = {
        "assignment_method": "whole_step_track_start_midpoint",
        "input_step_count": int(len(hits)),
        "output_track_bin_count": int(len(grouped)),
        "track_start_bin_width_cm": float(bin_width_cm),
        "input_energy_MeV": float(hits["E"].sum()),
        "grouped_energy_MeV": float(grouped["energy_MeV"].sum()),
        "energy_difference_MeV": float(grouped["energy_MeV"].sum() - hits["E"].sum()),
        "input_path_cm": float(hits["ds_cm"].sum()),
        "grouped_path_cm": float(grouped["path_length_cm"].sum()),
        "path_difference_cm": float(grouped["path_length_cm"].sum() - hits["ds_cm"].sum()),
        "number_of_bins": int(grouped["s_bin_cm"].nunique()),
    }
    return grouped, diagnostics


def bethe_bloch_kaon_lar(max_range_cm, n_points=2500):
    k_constant = 0.307075  # MeV mol^-1 cm^2
    electron_mass_MeV = 0.51099895
    kaon_mass_MeV = 493.677
    z_over_a_argon = 18.0 / 39.948
    density_lar_g_cm3 = 1.396
    mean_excitation_MeV = 188.0e-6

    kinetic_energy = np.geomspace(0.1, 5000.0, n_points)
    gamma = 1.0 + kinetic_energy / kaon_mass_MeV
    beta2 = 1.0 - 1.0 / gamma**2
    beta2 = np.clip(beta2, 1e-8, None)
    mass_ratio = electron_mass_MeV / kaon_mass_MeV
    t_max = (
        2.0
        * electron_mass_MeV
        * beta2
        * gamma**2
        / (1.0 + 2.0 * gamma * mass_ratio + mass_ratio**2)
    )
    log_arg = 2.0 * electron_mass_MeV * beta2 * gamma**2 * t_max / mean_excitation_MeV**2
    stopping_mass = (
        k_constant
        * z_over_a_argon
        / beta2
        * (0.5 * np.log(np.clip(log_arg, 1.0, None)) - beta2)
    )
    stopping_power = np.clip(stopping_mass * density_lar_g_cm3, 1e-9, None)

    dT = np.diff(kinetic_energy)
    inv_stopping_mid = 0.5 * (1.0 / stopping_power[1:] + 1.0 / stopping_power[:-1])
    residual_range = np.concatenate([[0.0], np.cumsum(dT * inv_stopping_mid)])

    mask = residual_range <= max_range_cm
    if mask.sum() < 2:
        return residual_range, stopping_power
    return residual_range[mask], stopping_power[mask]


def plot_primary_kaons(
    binned,
    output_path,
    number_events,
    binning_label,
    x_max_cm=None,
    y_max=None,
):
    primary = binned[binned["kaon_category"] == "primary kaon"].copy()
    if primary.empty:
        raise ValueError("No primary kaon dE/dx segments were found.")

    max_range = max(float(primary["residual_range_cm"].max()), 1.0)
    theory_range, theory_dedx = bethe_bloch_kaon_lar(max_range)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    primary_scatter = ax.scatter(
        primary["residual_range_cm"],
        primary["dEdx_MeV_per_cm"],
        s=9,
        alpha=0.45,
        linewidths=0,
        label="primary K+",
        rasterized=True,
    )
    ax.plot(
        theory_range,
        theory_dedx,
        color="black",
        linewidth=2.0,
        linestyle="--",
        label="_nolegend_",
    )

    ax.set_xlim(left=0.0, right=(x_max_cm if x_max_cm is not None else max_range * 1.03))
    ax.set_ylim(bottom=0.0, top=y_max)
    ax.set_xlabel("Residual range [cm]")
    ax.set_ylabel("dE/dx [MeV/cm]")
    ax.set_title(f"Primary kaon dE/dx vs residual range ({number_events} events)")
    ax.text(
        0.98,
        0.96,
        "\n".join(
            [
                binning_label,
                *([f"x max: {x_max_cm:g} cm"] if x_max_cm is not None else []),
                *([f"y max: {y_max:g} MeV/cm"] if y_max is not None else []),
                f"Primary kaons: {primary[['event', 'ParticleID']].drop_duplicates().shape[0]}",
                f"Segments: {len(primary)}",
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


def plot_primary_and_daughter_kaons(
    binned,
    output_path,
    number_events,
    binning_label,
    x_max_cm=None,
    y_max=None,
):
    if binned.empty:
        raise ValueError("No kaon dE/dx segments were found.")

    max_range = max(float(binned["residual_range_cm"].max()), 1.0)
    theory_range, theory_dedx = bethe_bloch_kaon_lar(max_range)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    styles = {
        "primary kaon": {"color": "tab:blue", "label": "primary K+", "s": 9, "alpha": 0.38},
        "daughter kaon": {"color": "tab:orange", "label": "daughter K+", "s": 16, "alpha": 0.7},
    }
    primary_color = styles["primary kaon"]["color"]
    for category, style in styles.items():
        group = binned[binned["kaon_category"] == category]
        if group.empty:
            continue
        ax.scatter(
            group["residual_range_cm"],
            group["dEdx_MeV_per_cm"],
            s=style["s"],
            alpha=style["alpha"],
            linewidths=0,
            color=style["color"],
            label=style["label"],
            rasterized=True,
        )

    ax.plot(
        theory_range,
        theory_dedx,
        color="black",
        linewidth=2.0,
        linestyle="--",
        label="_nolegend_",
    )

    counts = (
        binned[["event", "ParticleID", "kaon_category"]]
        .drop_duplicates()
        .groupby("kaon_category")
        .size()
    )
    ax.set_xlim(left=0.0, right=(x_max_cm if x_max_cm is not None else max_range * 1.03))
    ax.set_ylim(bottom=0.0, top=y_max)
    ax.set_xlabel("Residual range [cm]")
    ax.set_ylabel("dE/dx [MeV/cm]")
    ax.set_title(
        f"Primary and daughter kaons dE/dx vs residual range ({number_events} events)"
    )
    ax.text(
        0.98,
        0.96,
        "\n".join(
            [
                binning_label,
                *([f"x max: {x_max_cm:g} cm"] if x_max_cm is not None else []),
                *([f"y max: {y_max:g} MeV/cm"] if y_max is not None else []),
                f"Primary tracks: {int(counts.get('primary kaon', 0))}",
                f"Daughter tracks: {int(counts.get('daughter kaon', 0))}",
                f"Segments: {len(binned)}",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.6"},
    )
    ax.grid(True, alpha=0.25)
    legend = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.74), framealpha=0.9)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary(output_dir, primary_kaons, daughter_kaons, binned, diagnostics):
    # Reduce the binned table to one row per selected kaon track.
    selected_tracks = (
        binned[
            [
                "event",
                "ParticleID",
                "kaon_category",
                "particle_parent_track_id",
                "particle_process_key",
                "particle_final_kinetic_energy",
                "particle_decay_flag",
                "track_length_cm",
            ]
        ]
        .drop_duplicates()
        .sort_values(["event", "kaon_category", "ParticleID"])
    )
    # Save a compact truth-selection table alongside the detailed bin table.
    selected_tracks.to_csv(output_dir / "kaon_track_summary.csv", index=False)

    # Fixed comparison runs do not contain hybrid endpoint fields. Document the
    # uniform method separately so the main hybrid summary remains untouched.
    if diagnostics["assignment_method"] == "whole_step_track_start_midpoint":
        width = diagnostics["track_start_bin_width_cm"]
        width_token = f"{width:g}".replace(".", "p")
        summary_lines = [
            "Primary/daughter kaon uniform-bin comparison summary",
            "====================================================",
            "",
            f"Primary kaons in particles_output: {len(primary_kaons)}",
            f"Daughter kaons of primary kaons: {len(daughter_kaons)}",
            f"Kaon tracks with G4 hit segments: {selected_tracks.shape[0]}",
            f"Input positive-energy G4 steps: {diagnostics['input_step_count']}",
            f"Output track-bin measurements: {diagnostics['output_track_bin_count']}",
            "",
            "Midpoint assignment:",
            "  Every positive-energy Geant4 step is kept whole.",
            "  Every step is assigned from its distance-from-track-start midpoint.",
            f"  Uniform track-start bin width: {width:g} cm",
            "  No endpoint transition or special endpoint algorithm is used.",
            "  dE/dx = summed deposited energy / summed G4 step length.",
            "  Residual range is track length minus the path-weighted bin midpoint.",
            "",
            "Accounting checks:",
            f"  Input energy: {diagnostics['input_energy_MeV']:.12g} MeV",
            f"  Grouped energy: {diagnostics['grouped_energy_MeV']:.12g} MeV",
            f"  Energy difference: {diagnostics['energy_difference_MeV']:.12g} MeV",
            f"  Input path length: {diagnostics['input_path_cm']:.12g} cm",
            f"  Grouped path length: {diagnostics['grouped_path_cm']:.12g} cm",
            f"  Path-length difference: {diagnostics['path_difference_cm']:.12g} cm",
            "",
            "Purpose:",
            "  This comparison tests whether uniform fine bins add useful structure",
            "  or primarily expose fluctuations from individual Geant4 steps.",
            "  Bethe-Bloch predictions remain black dashed lines outside legends.",
        ]
        (output_dir / f"kaon_selection_summary_fixed_{width_token}cm.txt").write_text(
            "\n".join(summary_lines) + "\n"
        )
        return

    # Document both the physics selection and the numerical reconstruction.
    summary_lines = [
        "Primary/daughter kaon dE/dx analysis summary",
        "============================================",
        "",
        "Truth selection:",
        f"Primary kaons in particles_output: {len(primary_kaons)}",
        f"Daughter kaons of primary kaons in particles_output: {len(daughter_kaons)}",
        f"Kaon tracks with G4 hit segments: {selected_tracks.shape[0]}",
        f"Input positive-energy G4 steps: {diagnostics['input_step_count']}",
        f"Output track-bin measurements: {diagnostics['output_track_bin_count']}",
        "",
        "Residual-range reconstruction:",
        "  Every positive-energy Geant4 step is kept whole.",
        "  Each step is assigned to exactly one bin from its midpoint.",
        "  No step is split and no energy is redistributed between bins.",
        "  dE/dx = summed deposited energy / summed G4 step length.",
        "  The plotted x coordinate is the path-weighted step-midpoint range.",
        "",
        "Hybrid midpoint bins:",
        f"  Endpoint region: 0 to {diagnostics['endpoint_max_cm']:g} cm residual range",
        f"  Endpoint bin width: {diagnostics['endpoint_bin_width_cm']:g} cm",
        f"  Number of endpoint bins: {diagnostics['endpoint_bin_count']}",
        f"  Above transition: {diagnostics['track_start_bin_width_cm']:g} cm bins from each track start",
        "  A midpoint exactly at the transition belongs to the upper region.",
        "",
        "Conservation checks:",
        f"  Input energy: {diagnostics['input_energy_MeV']:.12g} MeV",
        f"  Grouped energy: {diagnostics['grouped_energy_MeV']:.12g} MeV",
        f"  Energy difference: {diagnostics['energy_difference_MeV']:.12g} MeV",
        f"  Input path length: {diagnostics['input_path_cm']:.12g} cm",
        f"  Grouped path length: {diagnostics['grouped_path_cm']:.12g} cm",
        f"  Path-length difference: {diagnostics['path_difference_cm']:.12g} cm",
        "",
        "Saved tables:",
        "  primary_kaon_dEdx_binned_segments_*.csv contains event, track, PDG,",
        "  kaon category, particles_output metadata, bin boundaries, deposited",
        "  energy, complete-step path, residual range, dE/dx, source-step count,",
        "  nominal width, path fraction, and midpoint-binning region.",
        "  primary_kaon_dEdx_binning_diagnostics_*.csv records configuration",
        "  and energy/path conservation totals for reproducibility.",
        "  kaon_track_summary.csv contains one row per selected kaon track.",
        "",
        "Plot convention:",
        "  Simulated primary and daughter kaons retain their colored markers.",
        "  Bethe-Bloch predictions are black dashed lines and are not in legends.",
        "",
        "Process key reminder:",
        "  14 = hadElastic",
        "  18 = protonInelastic",
        "  19 = dInelastic",
        "  20 = Decay",
        "  -2 = uncategorized process",
    ]
    (output_dir / "kaon_selection_summary.txt").write_text(
        "\n".join(summary_lines) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Make dE/dx vs residual range plots using particles_output.txt to "
            "separate primary kaons from daughter kaons."
        )
    )
    parser.add_argument(
        "input_path",
        help="Generated event folder containing g4_output.txt and particles_output.txt.",
    )
    parser.add_argument(
        "--binning",
        choices=("hybrid", "fixed"),
        default="hybrid",
        help="Use fine endpoint bins plus legacy track bins, or legacy fixed bins everywhere.",
    )
    parser.add_argument(
        "--bin-width-cm",
        type=float,
        default=DEFAULT_BIN_WIDTH_CM,
        help=f"Track-length bin width in cm. Default: {DEFAULT_BIN_WIDTH_CM:g}.",
    )
    parser.add_argument(
        "--endpoint-bin-width-cm",
        type=float,
        default=DEFAULT_ENDPOINT_BIN_WIDTH_CM,
        help=f"Residual-range bin width below the endpoint transition. Default: {DEFAULT_ENDPOINT_BIN_WIDTH_CM:g} cm.",
    )
    parser.add_argument(
        "--endpoint-max-cm",
        type=float,
        default=DEFAULT_ENDPOINT_MAX_CM,
        help=f"Upper residual-range boundary of fine endpoint bins. Default: {DEFAULT_ENDPOINT_MAX_CM:g} cm.",
    )
    args = parser.parse_args()

    g4_path = find_required_file(args.input_path, "g4_output.txt")
    particles_path = find_required_file(args.input_path, "particles_output.txt")

    g4_hits = read_g4_hits(g4_path)
    particles = read_particles(particles_path)
    number_events = event_count_from_last_row(g4_hits)

    primary_kaons, daughter_kaons, selected_kaons = classify_kaons(particles)
    selected_hits = select_hits_for_particles(g4_hits, selected_kaons)
    binned, diagnostics = make_binned_dedx(
        selected_hits,
        args.bin_width_cm,
        binning=args.binning,
        endpoint_bin_width_cm=args.endpoint_bin_width_cm,
        endpoint_max_cm=args.endpoint_max_cm,
    )
    if args.binning == "hybrid":
        binning_label = (
            f"{args.endpoint_bin_width_cm:g} cm midpoint bins below "
            f"{args.endpoint_max_cm:g} cm"
        )
        file_suffix = "_hybrid_midpoint"
    else:
        binning_label = f"Fixed bin width: {args.bin_width_cm:g} cm"
        width_token = f"{args.bin_width_cm:g}".replace(".", "p")
        file_suffix = f"_fixed_{width_token}cm"

    output_dir = (
        SCRIPT_DIR
        / "dEdx_primary_kaons"
        / f"kaon_decay_{number_events}_events"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_plot = (
        output_dir
        / f"primary_kaons_dEdx_vs_residual_range{file_suffix}_{number_events}_events.png"
    )
    primary_daughter_plot = (
        output_dir
        / f"primary_and_daughter_kaons_dEdx_vs_residual_range{file_suffix}_{number_events}_events.png"
    )
    primary_zoom_plot = (
        output_dir
        / f"primary_kaons_dEdx_vs_residual_range_xmax10cm_ymax40MeVcm{file_suffix}_{number_events}_events.png"
    )
    primary_daughter_zoom_plot = (
        output_dir
        / f"primary_and_daughter_kaons_dEdx_vs_residual_range_xmax10cm_ymax40MeVcm{file_suffix}_{number_events}_events.png"
    )
    binned_csv = output_dir / f"primary_kaon_dEdx_binned_segments{file_suffix}_{number_events}_events.csv"
    diagnostics_csv = output_dir / f"primary_kaon_dEdx_binning_diagnostics{file_suffix}_{number_events}_events.csv"

    plot_primary_kaons(binned, primary_plot, number_events, binning_label)
    plot_primary_and_daughter_kaons(
        binned, primary_daughter_plot, number_events, binning_label
    )
    plot_primary_kaons(
        binned,
        primary_zoom_plot,
        number_events,
        binning_label,
        x_max_cm=DEFAULT_ZOOM_X_MAX_CM,
        y_max=DEFAULT_ZOOM_Y_MAX_MEV_PER_CM,
    )
    plot_primary_and_daughter_kaons(
        binned,
        primary_daughter_zoom_plot,
        number_events,
        binning_label,
        x_max_cm=DEFAULT_ZOOM_X_MAX_CM,
        y_max=DEFAULT_ZOOM_Y_MAX_MEV_PER_CM,
    )
    binned.to_csv(binned_csv, index=False)
    pd.DataFrame([diagnostics]).to_csv(diagnostics_csv, index=False)
    write_summary(output_dir, primary_kaons, daughter_kaons, binned, diagnostics)

    print(f"Read G4 hits: {g4_path}")
    print(f"Read particles: {particles_path}")
    print(f"Number of events: {number_events}")
    print(f"Primary kaons in particles_output: {len(primary_kaons)}")
    print(f"Daughter kaons of primary kaons: {len(daughter_kaons)}")
    print(f"Selected G4 hit rows: {len(selected_hits)}")
    print(f"Binned dE/dx segments: {len(binned)}")
    print(f"Saved primary kaon plot: {primary_plot}")
    print(f"Saved primary + daughter kaon plot: {primary_daughter_plot}")
    print(f"Saved primary kaon zoom plot: {primary_zoom_plot}")
    print(f"Saved primary + daughter kaon zoom plot: {primary_daughter_zoom_plot}")
    print(f"Saved binned segment table: {binned_csv}")
    print(f"Saved accounting diagnostics: {diagnostics_csv}")
    print(f"Saved summaries in: {output_dir}")


if __name__ == "__main__":
    main()

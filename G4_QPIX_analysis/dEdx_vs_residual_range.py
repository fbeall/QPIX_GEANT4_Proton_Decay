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
DEFAULT_KAON_ZOOM_X_MAX_CM = 10.0
DEFAULT_ALL_PARTICLES_ZOOM_X_MAX_CM = 40.0
KAON_PDG = 321

THEORY_PARTICLES = {
    "K": {"mass_MeV": 493.677, "pdgs": (321, -321)},
    "p": {"mass_MeV": 938.272, "pdgs": (2212,)},
    "mu": {"mass_MeV": 105.658, "pdgs": (-13, 13)},
    "pi": {"mass_MeV": 139.570, "pdgs": (211, -211)},
}

PDG_LABELS = {
    -13: "mu+",
    13: "mu-",
    -11: "e+",
    11: "e-",
    211: "pi+",
    -211: "pi-",
    321: "K+",
    -321: "K-",
    2212: "p",
    1000180380: "Ar38",
    1000180390: "Ar39",
    1000180400: "Ar40",
}


def find_g4_output(input_path):
    path = Path(input_path).expanduser()
    if path.is_file():
        if path.name.lower() != "g4_output.txt":
            raise FileNotFoundError(f"Input file is not named g4_output.txt: {path}")
        return path

    g4_output = path / "g4_output.txt"
    if g4_output.exists():
        return g4_output

    matches = sorted(path.rglob("g4_output.txt"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No g4_output.txt found under {path}")
    raise FileExistsError(
        "Found multiple g4_output.txt files; pass the event folder or file directly:\n"
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


def event_count_from_last_row(g4_hits):
    event_column = "events" if "events" in g4_hits.columns else "event"
    return int(g4_hits[event_column].iloc[-1]) + 1


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
        # Assign every complete step by its midpoint. Fine residual-range bins
        # are used only near the endpoint; the original track-start bins remain
        # in force above the transition.
        grouped, diagnostics = bin_steps_by_midpoint(
            hits=hits,
            group_columns=["event", "ParticleID", "PDG"],
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
        hits.groupby(["event", "ParticleID", "PDG", *bin_columns], as_index=False)
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


def bethe_bloch_lar(mass_MeV, max_range_cm, n_points=2500):
    # Approximate Bethe-Bloch curve for singly charged particles in liquid argon.
    # Density-effect and shell corrections are omitted; this is a clean guide curve.
    k_constant = 0.307075  # MeV mol^-1 cm^2
    electron_mass_MeV = 0.51099895
    z_over_a_argon = 18.0 / 39.948
    density_lar_g_cm3 = 1.396
    mean_excitation_MeV = 188.0e-6

    kinetic_energy = np.geomspace(0.1, 5000.0, n_points)
    gamma = 1.0 + kinetic_energy / mass_MeV
    beta2 = 1.0 - 1.0 / gamma**2
    beta2 = np.clip(beta2, 1e-8, None)
    mass_ratio = electron_mass_MeV / mass_MeV
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
    stopping_power = stopping_mass * density_lar_g_cm3
    stopping_power = np.clip(stopping_power, 1e-9, None)

    dT = np.diff(kinetic_energy)
    inv_stopping_mid = 0.5 * (1.0 / stopping_power[1:] + 1.0 / stopping_power[:-1])
    residual_range = np.concatenate([[0.0], np.cumsum(dT * inv_stopping_mid)])

    mask = residual_range <= max_range_cm
    if mask.sum() < 2:
        return residual_range, stopping_power
    return residual_range[mask], stopping_power[mask]


def bethe_bloch_kaon_lar(max_range_cm, n_points=2500):
    return bethe_bloch_lar(THEORY_PARTICLES["K"]["mass_MeV"], max_range_cm, n_points)


def pdg_label(pdg):
    return PDG_LABELS.get(int(pdg), str(int(pdg)))


def plot_kaons_with_bethe(
    binned,
    output_path,
    number_events,
    binning_label,
    x_max_cm=None,
    y_min=None,
    y_max=None,
):
    kaons = binned[binned["PDG"].abs() == KAON_PDG].copy()
    if kaons.empty:
        raise ValueError("No kaon hits with |PDG| == 321 were found in g4_output.txt")

    max_range = max(float(kaons["residual_range_cm"].max()), 1.0)
    plot_max_range = x_max_cm if x_max_cm is not None else max_range
    theory_range, theory_dedx = bethe_bloch_kaon_lar(max_range)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    kaon_color = None
    for pdg, group in sorted(kaons.groupby("PDG"), key=lambda item: item[0]):
        scatter = ax.scatter(
            group["residual_range_cm"],
            group["dEdx_MeV_per_cm"],
            s=9,
            alpha=0.45,
            linewidths=0,
            label=f"{pdg_label(pdg)} G4 segments",
            rasterized=True,
        )
        if kaon_color is None or int(pdg) == 321:
            kaon_color = scatter.get_facecolor()[0]

    ax.plot(
        theory_range,
        theory_dedx,
        color=kaon_color,
        linewidth=2.0,
        linestyle="--",
        label="_nolegend_",
    )
    ax.set_xlim(left=0.0, right=plot_max_range * 1.03)
    ax.set_ylim(
        bottom=0.0 if y_min is None else y_min,
        top=y_max,
    )
    ax.set_xlabel("Residual range [cm]")
    ax.set_ylabel("dE/dx [MeV/cm]")
    ax.set_title(f"Kaon dE/dx vs residual range ({number_events} events)")
    ax.text(
        0.98,
        0.96,
        "\n".join(
            [
                binning_label,
                *([f"x max: {x_max_cm:g} cm"] if x_max_cm is not None else []),
                *([f"y max: {y_max:g} MeV/cm"] if y_max is not None else []),
                f"Kaon segments: {len(kaons)}",
                "Truth G4 hits only",
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


def plot_all_particles(
    binned,
    output_path,
    number_events,
    binning_label,
    x_max_cm=None,
    y_min=None,
    y_max=None,
    log_y=True,
    theory_lines=False,
):
    pdg_counts = binned["PDG"].value_counts()
    top_pdgs = list(pdg_counts.head(10).index)

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
    pdg_colors = {}
    for index, pdg in enumerate(top_pdgs):
        group = binned[binned["PDG"] == pdg]
        # Keep charged-kaon data purple in every binning comparison. Without
        # this explicit assignment, changing bin width changes species counts,
        # which can reorder the automatic palette and make comparisons harder.
        color = "tab:purple" if abs(int(pdg)) == KAON_PDG else cmap(index % 10)
        pdg_colors[int(pdg)] = color
        ax.scatter(
            group["residual_range_cm"],
            group["dEdx_MeV_per_cm"],
            s=7,
            alpha=0.35,
            linewidths=0,
            color=color,
            label=f"{pdg_label(pdg)} ({len(group)})",
            rasterized=True,
        )

    if theory_lines:
        theory_max_range = x_max_cm
        if theory_max_range is None:
            theory_max_range = max(float(binned["residual_range_cm"].max()), 1.0)
        for label, config in THEORY_PARTICLES.items():
            theory_range, theory_dedx = bethe_bloch_lar(
                config["mass_MeV"], theory_max_range
            )
            ax.plot(
                theory_range,
                theory_dedx,
                color="black",
                linewidth=1.8,
                linestyle="--",
                label="_nolegend_",
            )

    if y_min is not None or y_max is not None:
        ax.set_ylim(
            bottom=0.0 if y_min is None else y_min,
            top=y_max,
        )
    else:
        ymin = max(float(binned["dEdx_MeV_per_cm"].quantile(0.001)), 1e-3)
        ymax = binned["dEdx_MeV_per_cm"].quantile(0.999)
        if np.isfinite(ymax) and ymax > ymin:
            ax.set_ylim(ymin, ymax * 1.2)
    if log_y:
        ax.set_yscale("log")
    if x_max_cm is None:
        ax.set_xlim(left=0.0)
    else:
        ax.set_xlim(left=0.0, right=x_max_cm)
    ax.set_xlabel("Residual range [cm]")
    ax.set_ylabel("dE/dx [MeV/cm]")
    ax.set_title(f"All G4 particles dE/dx vs residual range ({number_events} events)")
    ax.text(
        0.98,
        0.96,
        "\n".join(
            [
                binning_label,
                *([f"x max: {x_max_cm:g} cm"] if x_max_cm is not None else []),
                *([f"y max: {y_max:g} MeV/cm"] if y_max is not None else []),
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


def write_analysis_summary(output_dir, binned, diagnostics):
    """Write a human-readable record of the active midpoint configuration."""
    # Fixed comparison runs contain no endpoint/upper-region split. Give these
    # runs a dedicated width-specific summary and return before hybrid fields
    # are accessed.
    if diagnostics["assignment_method"] == "whole_step_track_start_midpoint":
        width = diagnostics["track_start_bin_width_cm"]
        width_token = f"{width:g}".replace(".", "p")
        summary_lines = [
            "All-particle uniform-bin comparison summary",
            "===========================================",
            "",
            "Midpoint assignment:",
            "  Every positive-energy Geant4 step is kept whole.",
            "  Every step is assigned from its distance-from-track-start midpoint.",
            f"  Uniform track-start bin width: {width:g} cm",
            "  No endpoint transition or special endpoint algorithm is used.",
            "  dE/dx = summed deposited energy / summed G4 step length.",
            "  Residual range is track length minus the path-weighted bin midpoint.",
            "",
            "Input and output:",
            f"  Positive-energy G4 steps: {diagnostics['input_step_count']}",
            f"  Output track-bin measurements: {diagnostics['output_track_bin_count']}",
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
            "  This is a comparison product for judging whether uniform fine bins",
            "  add useful structure or primarily expose individual G4-step noise.",
            "  Bethe-Bloch predictions remain black dashed lines outside legends.",
        ]
        (output_dir / f"dEdx_analysis_summary_fixed_{width_token}cm.txt").write_text(
            "\n".join(summary_lines) + "\n"
        )
        return

    # Count the output rows produced by each half of the hybrid algorithm.
    region_counts = binned["binning_region"].value_counts()

    # Assemble the explanation as explicit lines so the generated text file is
    # readable without opening either the Python source or diagnostics CSV.
    summary_lines = [
        "All-particle dE/dx analysis summary",
        "==================================",
        "",
        "Input and output:",
        f"  Positive-energy G4 steps: {diagnostics['input_step_count']}",
        f"  Output track-bin measurements: {diagnostics['output_track_bin_count']}",
        f"  Endpoint measurements: {int(region_counts.get('endpoint_residual_midpoint', 0))}",
        f"  Upper-track measurements: {int(region_counts.get('legacy_track_start_midpoint', 0))}",
        "",
        "Midpoint assignment:",
        "  Every positive-energy Geant4 step is kept whole.",
        "  Each step is assigned to exactly one bin from its midpoint.",
        "  No step is split and no energy is redistributed between bins.",
        "  dE/dx = summed deposited energy / summed G4 step length.",
        "  The plotted x coordinate is the path-weighted step-midpoint range.",
        "",
        "Hybrid bins:",
        f"  Endpoint region: 0 to {diagnostics['endpoint_max_cm']:g} cm residual range",
        f"  Endpoint bin width: {diagnostics['endpoint_bin_width_cm']:g} cm",
        f"  Number of endpoint bins: {diagnostics['endpoint_bin_count']}",
        f"  Above transition: {diagnostics['track_start_bin_width_cm']:g} cm bins from each track start",
        "  A midpoint exactly at the transition belongs to the upper region.",
        "",
        "Accounting checks:",
        f"  Input energy: {diagnostics['input_energy_MeV']:.12g} MeV",
        f"  Grouped energy: {diagnostics['grouped_energy_MeV']:.12g} MeV",
        f"  Energy difference: {diagnostics['energy_difference_MeV']:.12g} MeV",
        f"  Input path length: {diagnostics['input_path_cm']:.12g} cm",
        f"  Grouped path length: {diagnostics['grouped_path_cm']:.12g} cm",
        f"  Path-length difference: {diagnostics['path_difference_cm']:.12g} cm",
        "",
        "Saved tables:",
        "  dEdx_binned_segments_hybrid_midpoint_*.csv contains event, track,",
        "  PDG, energy, complete-step path, source-step count, weighted residual",
        "  range, dE/dx, nominal width, path fraction, and binning region.",
        "  dEdx_binning_diagnostics_hybrid_midpoint_*.csv contains the complete",
        "  midpoint configuration and energy/path accounting totals.",
        "",
        "Plot convention:",
        "  Simulated particle species retain their colored markers.",
        "  Bethe-Bloch predictions are black dashed lines and are not in legends.",
    ]

    # Replace the previous run's summary so it always matches current outputs.
    (output_dir / "dEdx_analysis_summary.txt").write_text(
        "\n".join(summary_lines) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Make dE/dx vs residual range plots from a generated event folder "
            "containing g4_output.txt."
        )
    )
    parser.add_argument(
        "input_path",
        help="Generated event folder, or the g4_output.txt file itself.",
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

    g4_path = find_g4_output(args.input_path)
    g4_hits = read_g4_hits(g4_path)
    number_events = event_count_from_last_row(g4_hits)
    binned, diagnostics = make_binned_dedx(
        g4_hits,
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

    output_dir = SCRIPT_DIR / "dEdx" / f"kaon_decay_{number_events}_events"
    output_dir.mkdir(parents=True, exist_ok=True)

    kaon_plot = output_dir / f"kaon_dEdx_vs_residual_range{file_suffix}_{number_events}_events.png"
    all_plot = output_dir / f"all_particles_dEdx_vs_residual_range{file_suffix}_{number_events}_events.png"
    kaon_zoom_plot = (
        output_dir
        / f"kaon_dEdx_vs_residual_range_xmax10cm{file_suffix}_{number_events}_events.png"
    )
    kaon_zoom_linear_plot = (
        output_dir
        / f"kaon_dEdx_vs_residual_range_xmax10cm_ymax40MeVcm{file_suffix}_{number_events}_events.png"
    )
    all_zoom_plot = (
        output_dir
        / f"all_particles_dEdx_vs_residual_range_xmax40cm{file_suffix}_{number_events}_events.png"
    )
    all_zoom_linear_plot = (
        output_dir
        / f"all_particles_dEdx_vs_residual_range_xmax10cm_ymax40MeVcm{file_suffix}_{number_events}_events.png"
    )
    binned_csv = output_dir / f"dEdx_binned_segments{file_suffix}_{number_events}_events.csv"
    diagnostics_csv = output_dir / f"dEdx_binning_diagnostics{file_suffix}_{number_events}_events.csv"

    plot_kaons_with_bethe(binned, kaon_plot, number_events, binning_label)
    plot_all_particles(binned, all_plot, number_events, binning_label)
    plot_kaons_with_bethe(
        binned,
        kaon_zoom_plot,
        number_events,
        binning_label,
        x_max_cm=DEFAULT_KAON_ZOOM_X_MAX_CM,
    )
    plot_kaons_with_bethe(
        binned,
        kaon_zoom_linear_plot,
        number_events,
        binning_label,
        x_max_cm=DEFAULT_KAON_ZOOM_X_MAX_CM,
        y_min=0.0,
        y_max=40.0,
    )
    plot_all_particles(
        binned,
        all_zoom_plot,
        number_events,
        binning_label,
        x_max_cm=DEFAULT_ALL_PARTICLES_ZOOM_X_MAX_CM,
    )
    plot_all_particles(
        binned,
        all_zoom_linear_plot,
        number_events,
        binning_label,
        x_max_cm=10.0,
        y_min=0.0,
        y_max=40.0,
        log_y=False,
        theory_lines=True,
    )
    binned.to_csv(binned_csv, index=False)
    pd.DataFrame([diagnostics]).to_csv(diagnostics_csv, index=False)
    write_analysis_summary(output_dir, binned, diagnostics)

    print(f"Read: {g4_path}")
    print(f"Number of events: {number_events}")
    print(f"G4 hit rows used: {len(g4_hits)}")
    print(f"Binned dE/dx segments: {len(binned)}")
    print(f"Saved kaon plot: {kaon_plot}")
    print(f"Saved all-particles plot: {all_plot}")
    print(f"Saved kaon 10 cm plot: {kaon_zoom_plot}")
    print(f"Saved kaon 10 cm, 40 MeV/cm plot: {kaon_zoom_linear_plot}")
    print(f"Saved all-particles 40 cm plot: {all_zoom_plot}")
    print(f"Saved all-particles 10 cm, 40 MeV/cm plot: {all_zoom_linear_plot}")
    print(f"Saved binned segment table: {binned_csv}")
    print(f"Saved accounting diagnostics: {diagnostics_csv}")
    print(f"Saved analysis summary: {output_dir / 'dEdx_analysis_summary.txt'}")


if __name__ == "__main__":
    main()

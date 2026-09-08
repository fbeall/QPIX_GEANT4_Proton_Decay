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


DEFAULT_ENERGY_MIN = 0.0
DEFAULT_ENERGY_MAX = 200.0
DEFAULT_BIN_SIZE = 5.0


def find_particle_files(input_path):
    path = Path(input_path).expanduser()

    if path.is_file():
        return [path]

    particle_file = path / "particles_output.txt"
    if particle_file.exists():
        return [particle_file]

    return sorted(path.rglob("particles_output.txt"))


def main():
    parser = argparse.ArgumentParser(
        description="Plot final kinetic energy for decayed kaons."
    )
    parser.add_argument(
        "input_path",
        help="Folder containing particles_output.txt, or a particles_output.txt file.",
    )
    args = parser.parse_args()

    particle_files = find_particle_files(args.input_path)
    if not particle_files:
        raise FileNotFoundError(
            f"No particles_output.txt files found under {args.input_path}"
        )

    particles = pd.concat(
        [pd.read_csv(particle_file) for particle_file in particle_files],
        ignore_index=True,
    )

    required_columns = {
        "event",
        "particle_pdg_code",
        "particle_decay_flag",
        "particle_final_kinetic_energy",
    }
    missing_columns = required_columns.difference(particles.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required column(s): {missing}")

    number_events = int(particles.iloc[-1]["event"]) + 1

    decayed_kaons = particles[
        (particles["particle_pdg_code"].abs() == 321)
        & (particles["particle_decay_flag"] == 1)
    ].copy()
    at_rest_kaons = decayed_kaons[
        decayed_kaons["particle_final_kinetic_energy"] == 0
    ]
    inflight_kaons = decayed_kaons[
        decayed_kaons["particle_final_kinetic_energy"] > 0
    ]

    log_dir = SCRIPT_DIR / "kaon_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"kaon_decay_log_{number_events}_events.txt"
    decayed_kaons.to_csv(log_path, index=False)

    hist_dir = SCRIPT_DIR / "hists" / f"kaon_decay_{number_events}_events" / "energy"
    hist_dir.mkdir(parents=True, exist_ok=True)
    bin_edges = np.arange(
        DEFAULT_ENERGY_MIN,
        DEFAULT_ENERGY_MAX + DEFAULT_BIN_SIZE,
        DEFAULT_BIN_SIZE,
    )

    summary_text = "\n".join(
        [
            f"Bin size: {DEFAULT_BIN_SIZE:g} MeV",
            f"Total decayed K: {len(decayed_kaons)}",
            f"At rest, KE = 0: {len(at_rest_kaons)}",
            f"In flight, KE > 0: {len(inflight_kaons)}",
        ]
    )

    output_paths = []

    def save_histogram(data, filename, title, log_y=False, annotation=None):
        output_path = hist_dir / filename
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(
            data["particle_final_kinetic_energy"],
            bins=bin_edges,
            edgecolor="black",
        )
        ax.set_xlim(DEFAULT_ENERGY_MIN, DEFAULT_ENERGY_MAX)
        if log_y:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Final kinetic energy [MeV]")
        ax.set_ylabel("Number of decayed kaons")
        ax.text(
            0.98,
            0.95,
            annotation or f"Bin size: {DEFAULT_BIN_SIZE:g} MeV",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "black"},
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        output_paths.append(output_path)

    save_histogram(
        decayed_kaons,
        f"kaon_decay_energy_hist_linear_{number_events}_events.png",
        f"Kaon decay energy ({number_events} events)",
        annotation=summary_text,
    )
    save_histogram(
        decayed_kaons,
        f"kaon_decay_energy_hist_logy_{number_events}_events.png",
        f"Kaon decay energy log y ({number_events} events)",
        log_y=True,
    )
    save_histogram(
        inflight_kaons,
        f"kaon_decay_energy_hist_inflight_only_{number_events}_events.png",
        f"Kaon decay energy in flight only ({len(inflight_kaons)}/{number_events} events)",
    )

    print(f"Read {len(particle_files)} particle file(s)")
    print(f"Number of events: {number_events}")
    print(f"Decayed kaons: {len(decayed_kaons)}")
    print(f"At-rest decayed kaons: {len(at_rest_kaons)}")
    print(f"In-flight decayed kaons: {len(inflight_kaons)}")
    print(f"Saved kaon log: {log_path}")
    for output_path in output_paths:
        print(f"Saved histogram: {output_path}")


if __name__ == "__main__":
    main()

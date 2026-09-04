#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of final kinetic energy bins to use. Default: 50.",
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
    ]

    hist_dir = SCRIPT_DIR / "hists"
    hist_dir.mkdir(parents=True, exist_ok=True)
    output_path = hist_dir / f"kaon_decay_energy_hist_{number_events}_events.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    _, bin_edges, _ = ax.hist(
        decayed_kaons["particle_final_kinetic_energy"],
        bins=args.bins,
        edgecolor="black",
    )
    bin_size = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 0.0
    ax.set_title(f"Kaon decay energy ({number_events} events)")
    ax.set_xlabel("Final kinetic energy [MeV]")
    ax.set_ylabel("Number of decayed kaons")
    ax.text(
        0.98,
        0.95,
        f"Bin size: {bin_size:.4g} MeV",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Read {len(particle_files)} particle file(s)")
    print(f"Number of events: {number_events}")
    print(f"Decayed kaons: {len(decayed_kaons)}")
    print(f"Saved histogram: {output_path}")


if __name__ == "__main__":
    main()

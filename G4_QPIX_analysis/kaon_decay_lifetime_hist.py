#!/usr/bin/env python3

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_BIN_WIDTH_NS = 1.0


def main():
    parser = argparse.ArgumentParser(
        description="Plot lab-frame lifetimes for decayed kaons from a kaon log."
    )
    parser.add_argument(
        "kaon_log",
        help="Path to kaon_decay_log_#_events.txt from kaon_decay_energy_hist.py.",
    )
    parser.add_argument(
        "--bin-width-ns",
        type=float,
        default=DEFAULT_BIN_WIDTH_NS,
        help="Lifetime bin width in ns. Default: 1.0.",
    )
    args = parser.parse_args()

    if args.bin_width_ns <= 0:
        raise ValueError("--bin-width-ns must be positive")

    kaon_log = Path(args.kaon_log).expanduser()
    kaons = pd.read_csv(kaon_log)

    required_columns = {
        "event",
        "particle_initial_t",
        "particle_final_t",
    }
    missing_columns = required_columns.difference(kaons.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required column(s): {missing}")

    number_events = int(kaons.iloc[-1]["event"]) + 1
    kaons["particle_lifetime_ns"] = (
        kaons["particle_final_t"] - kaons["particle_initial_t"]
    )

    max_lifetime = kaons["particle_lifetime_ns"].max()
    max_edge = args.bin_width_ns * math.ceil(max_lifetime / args.bin_width_ns)
    if max_edge == 0:
        max_edge = args.bin_width_ns
    bin_edges = np.arange(0.0, max_edge + args.bin_width_ns, args.bin_width_ns)

    hist_dir = SCRIPT_DIR / "hists" / f"kaon_decay_{number_events}_events" / "lifetime"
    hist_dir.mkdir(parents=True, exist_ok=True)
    output_path = hist_dir / f"kaon_decay_lifetime_hist_linear_{number_events}_events.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        kaons["particle_lifetime_ns"],
        bins=bin_edges,
        edgecolor="black",
    )
    ax.set_xlim(0.0, max_edge)
    ax.set_title(f"Kaon decay lifetime ({number_events} events)")
    ax.set_xlabel("Lab-frame lifetime [ns]")
    ax.set_ylabel("Number of decayed kaons")
    ax.text(
        0.98,
        0.95,
        "\n".join(
            [
                f"Bin length: {args.bin_width_ns:g} ns",
                f"Mean lifetime: {kaons['particle_lifetime_ns'].mean():.3g} ns",
                f"Median lifetime: {kaons['particle_lifetime_ns'].median():.3g} ns",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "black"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Read kaon log: {kaon_log}")
    print(f"Number of events: {number_events}")
    print(f"Decayed kaons: {len(kaons)}")
    print(f"Mean lab-frame lifetime: {kaons['particle_lifetime_ns'].mean():.4g} ns")
    print(f"Saved histogram: {output_path}")


if __name__ == "__main__":
    main()

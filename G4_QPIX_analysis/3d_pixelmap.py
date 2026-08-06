#!/usr/bin/env python

import argparse
import ast
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def electron_sum(value):
    """Return the total electron weight stored in one MC_Weights cell."""
    if pd.isna(value):
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    try:
        weights = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return 0.0

    if isinstance(weights, (int, float)):
        return float(weights)

    return float(sum(weights))


def output_dir_for(input_path):
    return SCRIPT_DIR / f"{input_path.parent.name}_pixelmap"


def add_electron_column(df):
    if "MC_Weights" in df.columns:
        # MC_Weights contains the electron contribution list for each reset.
        df["electrons"] = df["MC_Weights"].apply(electron_sum)
        return "Total electrons"

    # Fallback for reset-like files without MC truth weights.
    df["electrons"] = 1.0
    return "Reset count"


def render_pixelmap(df, png_path, title, colorbar_label):
    required_columns = {"pixel_x", "pixel_y", "electrons"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

    pixel_charge = (
        df.groupby(["pixel_x", "pixel_y"], as_index=False)["electrons"]
        .sum()
        .sort_values("electrons")
    )

    if pixel_charge.empty:
        raise ValueError(f"No pixel rows to plot for {title}")

    fig, ax = plt.subplots(figsize=(10, 9), dpi=160)

    scatter = ax.scatter(
        pixel_charge["pixel_x"],
        pixel_charge["pixel_y"],
        c=pixel_charge["electrons"],
        cmap="viridis",
        marker="s",
        s=14,
        edgecolors="none",
    )

    ax.set_title(title, pad=18)
    ax.set_xlabel("Pixel x")
    ax.set_ylabel("Pixel y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.35)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02, shrink=0.82)
    cbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    return len(pixel_charge), float(pixel_charge["electrons"].sum())


def make_pixelmaps(txt_path, plot_total=False):
    df = pd.read_csv(txt_path)

    required_columns = {"event", "pixel_x", "pixel_y"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"{txt_path} is missing required column(s): {', '.join(missing_columns)}"
        )

    colorbar_label = add_electron_column(df)
    output_dir = output_dir_for(txt_path)
    output_dir.mkdir(exist_ok=True)

    results = []

    if plot_total:
        png_path = output_dir / "pixelmap_total.png"
        n_pixels, total_electrons = render_pixelmap(
            df,
            png_path,
            f"Pixel Electron Map: all events ({txt_path.parent.name})",
            colorbar_label,
        )
        results.append(("total", png_path, n_pixels, total_electrons))
        return output_dir, results

    event_values = sorted(int(event) for event in df["event"].dropna().unique())
    event_width = max(2, len(str(max(event_values)))) if event_values else 2

    for event, event_df in df.groupby("event", sort=True):
        event_number = int(event)
        png_path = output_dir / f"pixelmap_E{event_number:0{event_width}d}.png"
        n_pixels, total_electrons = render_pixelmap(
            event_df,
            png_path,
            f"Pixel Electron Map: event {event_number} ({txt_path.parent.name})",
            colorbar_label,
        )
        results.append((event_number, png_path, n_pixels, total_electrons))

    return output_dir, results


def main():
    parser = argparse.ArgumentParser(
        description="Create a 2D pixel-plane map colored by total electrons per pixel."
    )
    parser.add_argument("txt_file", type=Path, help="Path to resets_output.txt")
    parser.add_argument(
        "--total",
        action="store_true",
        help="Plot one total map across all events instead of one map per event.",
    )
    args = parser.parse_args()

    txt_path = args.txt_file.expanduser().resolve()
    if not txt_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {txt_path}")

    output_dir, results = make_pixelmaps(txt_path, plot_total=args.total)
    print(f"Saved pixel maps in: {output_dir}")
    print(f"Maps generated: {len(results)}")
    for event, png_path, n_pixels, total_electrons in results:
        print(
            f"{event}: {png_path.name} "
            f"({n_pixels} pixels, {total_electrons:g} electrons)"
        )


if __name__ == "__main__":
    main()

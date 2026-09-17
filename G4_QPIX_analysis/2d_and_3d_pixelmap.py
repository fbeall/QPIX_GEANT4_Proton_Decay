#!/usr/bin/env python3

import argparse
import ast
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DRIFT_VELOCITY_CM_PER_S = 164800.0
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import uproot


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


def output_dir_for(input_path, base_dir=None):
    if base_dir is None:
        base_dir = SCRIPT_DIR / "pixel_maps"
    return base_dir / f"{input_path.parent.name}_pixelmap"


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
    names = ["pixel_size"]
    with uproot.open(root_path) as root_file:
        metadata = root_file["metadata"]
        values = {}
        for name in names:
            if name not in metadata:
                raise KeyError(f"Missing metadata branch '{name}' in {root_path}")
            value = metadata[name].array(library="np")[0]
            values[name] = value.item() if hasattr(value, "item") else value
    return values


def parse_event_selector(selectors):
    if not selectors:
        return None

    selected = set()
    for selector in selectors:
        for item in selector.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start, end = item.split("-", 1)
                start = int(start.strip())
                end = int(end.strip())
                if end < start:
                    raise ValueError(f"Invalid event range: {item}")
                selected.update(range(start, end + 1))
            else:
                selected.add(int(item))
    return selected


def add_electron_column(df):
    if "MC_Weights" in df.columns:
        # MC_Weights contains the electron contribution list for each reset.
        df["electrons"] = df["MC_Weights"].apply(electron_sum)
        return "Total electrons"

    # Fallback for reset-like files without MC truth weights.
    df["electrons"] = 1.0
    return "Reset count"


def select_events(df, selected_events):
    if selected_events is None:
        return df

    selected_df = df[df["event"].isin(selected_events)].copy()
    available_events = set(int(event) for event in df["event"].dropna().unique())
    missing_events = sorted(selected_events - available_events)
    if missing_events:
        print(
            "Warning: requested event(s) not found: "
            + ", ".join(str(event) for event in missing_events)
        )
    if selected_df.empty:
        raise ValueError("No rows remain after applying the event selector.")
    return selected_df


def event_width_for(df):
    event_values = sorted(int(event) for event in df["event"].dropna().unique())
    return max(2, len(str(max(event_values)))) if event_values else 2


def render_2d_pixelmap(df, png_path, title, colorbar_label):
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


def render_3d_pixelmap(df, png_path, title, colorbar_label, metadata):
    required_columns = {"pixel_x", "pixel_y", "reset_time", "electrons"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

    points = df.copy()
    points["x_cm"] = (points["pixel_x"] + 0.5) * float(metadata["pixel_size"])
    points["y_cm"] = (points["pixel_y"] + 0.5) * float(metadata["pixel_size"])
    points["z_cm"] = points["reset_time"] * DRIFT_VELOCITY_CM_PER_S

    point_charge = (
        points.groupby(["x_cm", "y_cm", "z_cm"], as_index=False)["electrons"]
        .sum()
        .sort_values("electrons")
    )

    if point_charge.empty:
        raise ValueError(f"No reset rows to plot for {title}")

    fig = plt.figure(figsize=(10, 9), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        point_charge["x_cm"],
        point_charge["y_cm"],
        point_charge["z_cm"],
        c=point_charge["electrons"],
        cmap="viridis",
        s=8,
        depthshade=False,
    )

    ax.set_title(title, pad=18)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_zlabel("z [cm]")
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, linewidth=0.35, alpha=0.35)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.08, shrink=0.72)
    cbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    return len(point_charge), float(point_charge["electrons"].sum())


def make_pixelmaps(
    txt_path,
    output_dir=None,
    plot_total=False,
    selected_events=None,
    views=("2d", "3d"),
    metadata_root=None,
):
    df = pd.read_csv(txt_path)

    required_columns = {"event", "pixel_x", "pixel_y"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"{txt_path} is missing required column(s): {', '.join(missing_columns)}"
        )

    df = select_events(df, selected_events)
    colorbar_label = add_electron_column(df)
    if output_dir is None:
        output_dir = output_dir_for(txt_path)
    else:
        output_dir = output_dir_for(txt_path, output_dir.expanduser().resolve())

    views = tuple(dict.fromkeys(views))
    view_dirs = {}
    for view in views:
        view_dir = output_dir / view
        view_dir.mkdir(parents=True, exist_ok=True)
        view_dirs[view] = view_dir

    metadata = None
    if "3d" in views:
        metadata_root = find_metadata_root(txt_path.parent, metadata_root)
        metadata = read_metadata(metadata_root)

    results = []

    if plot_total:
        if "2d" in views:
            png_path = view_dirs["2d"] / "pixelmap_total.png"
            n_points, total_electrons = render_2d_pixelmap(
                df,
                png_path,
                f"Pixel Electron Map: selected events ({txt_path.parent.name})",
                colorbar_label,
            )
            results.append(("total", "2d", png_path, n_points, total_electrons))
        if "3d" in views:
            png_path = view_dirs["3d"] / "pixelmap_total_3d.png"
            n_points, total_electrons = render_3d_pixelmap(
                df,
                png_path,
                f"3D Reset Electron Map: selected events ({txt_path.parent.name})",
                colorbar_label,
                metadata,
            )
            results.append(("total", "3d", png_path, n_points, total_electrons))
        return output_dir, results

    event_width = event_width_for(df)

    for event, event_df in df.groupby("event", sort=True):
        event_number = int(event)
        if "2d" in views:
            png_path = view_dirs["2d"] / f"pixelmap_E{event_number:0{event_width}d}.png"
            n_points, total_electrons = render_2d_pixelmap(
                event_df,
                png_path,
                f"Pixel Electron Map: event {event_number} ({txt_path.parent.name})",
                colorbar_label,
            )
            results.append((event_number, "2d", png_path, n_points, total_electrons))
        if "3d" in views:
            png_path = (
                view_dirs["3d"] / f"pixelmap_3d_E{event_number:0{event_width}d}.png"
            )
            n_points, total_electrons = render_3d_pixelmap(
                event_df,
                png_path,
                f"3D Reset Electron Map: event {event_number} ({txt_path.parent.name})",
                colorbar_label,
                metadata,
            )
            results.append((event_number, "3d", png_path, n_points, total_electrons))

    return output_dir, results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create 2D pixel-plane maps and 3D reset maps colored by total "
            "electrons per pixel/reset."
        )
    )
    parser.add_argument("txt_file", type=Path, help="Path to resets_output.txt")
    parser.add_argument(
        "--events",
        action="append",
        help=(
            "Event selector, e.g. '0', '0,3,7', or '10-20'. Can be passed "
            "more than once."
        ),
    )
    parser.add_argument(
        "--total",
        action="store_true",
        help="Plot one total map across selected events instead of one map per event.",
    )
    parser.add_argument(
        "--views",
        choices=("2d", "3d", "both"),
        default="both",
        help="Which view(s) to write. Default: both.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        help=(
            "Path to the matching RTD ROOT file used to read pixel_size. "
            "Required for 3D if auto-detect fails."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help=(
            "Parent directory for generated PNG pixel maps. A run-specific "
            "<input-folder>_pixelmap directory with 2d/ and/or 3d/ subfolders "
            "will be created inside it."
        ),
    )
    args = parser.parse_args()

    txt_path = args.txt_file.expanduser().resolve()
    if not txt_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {txt_path}")

    views = ("2d", "3d") if args.views == "both" else (args.views,)
    output_dir, results = make_pixelmaps(
        txt_path,
        output_dir=args.output_dir,
        plot_total=args.total,
        selected_events=parse_event_selector(args.events),
        views=views,
        metadata_root=args.metadata_root,
    )
    print(f"Saved pixel maps in: {output_dir}")
    print(f"Maps generated: {len(results)}")
    for event, view, png_path, n_points, total_electrons in results:
        print(
            f"{event} [{view}]: {png_path.relative_to(output_dir)} "
            f"({n_points} points, {total_electrons:g} electrons)"
        )


if __name__ == "__main__":
    main()

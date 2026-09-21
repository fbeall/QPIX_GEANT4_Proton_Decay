#!/usr/bin/env python3
"""Plot and quantify atmospheric-neutrino interaction-vertex uniformity."""

# Import command-line parsing so the input file, output directory, and geometry can be overridden.
import argparse
# Import operating-system support for configuring Matplotlib's cache directory.
import os
# Import Path for reliable filesystem path handling.
from pathlib import Path

# Resolve this script's directory so default paths work from any current directory.
SCRIPT_DIR = Path(__file__).resolve().parent
# Keep Matplotlib's cache inside the analysis directory instead of relying on the home directory.
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

# Import Awkward Array to read ROOT's variable-length generator particle vectors.
import awkward as ak
# Import Matplotlib after selecting a local cache location.
import matplotlib
# Select a non-interactive backend because this script writes image files.
matplotlib.use("Agg")
# Import the plotting interface used to construct all figures.
import matplotlib.pyplot as plt
# Import NumPy for histogramming, normalization, and statistical calculations.
import numpy as np
# Import uproot to read the Geant4 ROOT file without PyROOT.
import uproot

# Point the default input at the atmospheric-neutrino Geant4 production file.
DEFAULT_INPUT = SCRIPT_DIR.parent.parent / "02_qpix_geant4" / "output" / "atmospheric_g4.root"
# Store the new vertex products in a dedicated output folder.
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "vertex_distribution"
# Define the configured active-volume dimensions in centimeters.
DEFAULT_LENGTHS_CM = (230.0, 600.0, 360.0)
# Give the three detector coordinates stable colors across plots.
AXIS_COLORS = {"x": "#2166ac", "y": "#1b9e77", "z": "#b2182b"}


# Parse command-line options and detector dimensions.
def parse_args() -> argparse.Namespace:
    # Create the command-line parser with a concise description.
    parser = argparse.ArgumentParser(description="Check atmospheric-neutrino vertex uniformity in G4 truth.")
    # Allow analysis of a different Q-Pix Geant4 ROOT file.
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Geant4 ROOT file containing event_tree.")
    # Allow the output directory to be redirected.
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for vertex plots and summary.")
    # Allow the active-volume x length to be changed for another geometry.
    parser.add_argument("--x-length", type=float, default=DEFAULT_LENGTHS_CM[0], help="Active x length in cm.")
    # Allow the active-volume y length to be changed for another geometry.
    parser.add_argument("--y-length", type=float, default=DEFAULT_LENGTHS_CM[1], help="Active y length in cm.")
    # Allow the active-volume z length to be changed for another geometry.
    parser.add_argument("--z-length", type=float, default=DEFAULT_LENGTHS_CM[2], help="Active z length in cm.")
    # Set the number of bins used by each one-dimensional uniformity test.
    parser.add_argument("--bins", type=int, default=20, help="Number of bins per axis for uniformity checks.")
    # Return all parsed command-line values.
    return parser.parse_args()


# Read one common interaction vertex per event from the generator initial state.
def read_vertices(input_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    # List the three generator branches needed for the vertex coordinates.
    branches = [
        "generator_initial_particle_x",
        "generator_initial_particle_y",
        "generator_initial_particle_z",
    ]
    # Open the ROOT file in a context manager so it always closes cleanly.
    with uproot.open(input_path) as root_file:
        # Require the standard Q-Pix Geant4 event tree.
        if "event_tree" not in root_file:
            # Raise a direct schema error when the event tree is absent.
            raise KeyError(f"No event_tree was found in {input_path}")
        # Access the newest event_tree cycle.
        tree = root_file["event_tree"]
        # Identify any required coordinate branches missing from this file.
        missing = sorted(set(branches).difference(tree.keys()))
        # Reject incompatible files before attempting array operations.
        if missing:
            # Report every missing branch together.
            raise KeyError("Missing required event_tree branches: " + ", ".join(missing))
        # Load only the three small generator coordinate vectors.
        arrays = tree.arrays(branches, library="ak", how=dict)
        # Record the ROOT event count for cross-checking.
        event_count = int(tree.num_entries)
    # Prepare a dictionary for the validated event-level coordinates.
    coordinates: dict[str, np.ndarray] = {}
    # Process x, y, and z using the same consistency checks.
    for axis in ("x", "y", "z"):
        # Retrieve this coordinate's jagged initial-particle vector.
        values = arrays[f"generator_initial_particle_{axis}"]
        # Require at least one initial-state particle in every event.
        if bool(ak.any(ak.num(values, axis=1) == 0)):
            # Stop because a missing initial particle means the event vertex is undefined.
            raise ValueError(f"At least one event has no generator initial-state {axis} coordinate.")
        # Verify that all initial particles in an event share one interaction vertex.
        event_spread = ak.max(values, axis=1) - ak.min(values, axis=1)
        # Reject inconsistent event records rather than silently choosing one particle's coordinate.
        if bool(ak.any(abs(event_spread) > 1.0e-9)):
            # Explain which coordinate failed the shared-vertex requirement.
            raise ValueError(f"Initial-state particles do not share one {axis} coordinate in every event.")
        # Use the first initial particle's coordinate as the common event vertex.
        coordinates[axis] = np.asarray(ak.firsts(values), dtype=float)
    # Return aligned x, y, and z coordinate arrays and the event count.
    return coordinates["x"], coordinates["y"], coordinates["z"], event_count


# Draw the twelve edges of a rectangular detector volume on a 3D axis.
def draw_detector_box(axis: plt.Axes, lengths: np.ndarray) -> None:
    # Unpack the three active-volume dimensions.
    x_length, y_length, z_length = lengths
    # Enumerate all eight box corners.
    corners = np.array([
        [x, y, z]
        for x in (0.0, x_length)
        for y in (0.0, y_length)
        for z in (0.0, z_length)
    ])
    # Visit every unique pair of corners.
    for first_index in range(len(corners)):
        # Compare the first corner with only later corners to avoid duplicate edges.
        for second_index in range(first_index + 1, len(corners)):
            # Select pairs that differ in exactly one coordinate, which are box edges.
            if np.count_nonzero(corners[first_index] != corners[second_index]) == 1:
                # Draw this detector edge behind the interaction vertices.
                axis.plot(*zip(corners[first_index], corners[second_index]), color="0.25", linewidth=0.8, alpha=0.8)


# Calculate simple, dependency-free uniformity diagnostics for one coordinate.
def uniformity_statistics(values: np.ndarray, length: float, bins: int) -> dict[str, float]:
    # Count entries in equal-width bins over the configured active extent.
    counts, _ = np.histogram(values, bins=bins, range=(0.0, length))
    # Calculate the expected count per bin under a uniform distribution.
    expected = len(values) / bins
    # Calculate Pearson's chi-square statistic relative to equal occupancy.
    chi_square = float(np.sum((counts - expected) ** 2 / expected))
    # Divide by the number of independent bin differences for a reduced diagnostic.
    reduced_chi_square = chi_square / (bins - 1)
    # Express the largest bin deviation in approximate Poisson standard deviations.
    maximum_pull = float(np.max(np.abs(counts - expected) / np.sqrt(expected)))
    # Return the metrics needed by the plots and text report.
    return {
        "expected": expected,
        "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square,
        "maximum_pull": maximum_pull,
    }


# Save the transparent 3D event-display-style point cloud.
def plot_3d(vertices: np.ndarray, lengths: np.ndarray, output_path: Path) -> None:
    # Create a wide figure that leaves room for the colorbar and labels.
    figure = plt.figure(figsize=(10.5, 7.2))
    # Add a perspective 3D plotting axis.
    axis = figure.add_subplot(111, projection="3d")
    # Draw every interaction vertex with transparency so dense regions do not become opaque immediately.
    points = axis.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=vertices[:, 2], cmap="viridis",
                          s=5, alpha=0.22, linewidths=0, rasterized=True)
    # Draw the active-volume boundary as a wireframe box.
    draw_detector_box(axis, lengths)
    # Set the configured coordinate limits.
    axis.set(xlim=(0.0, lengths[0]), ylim=(0.0, lengths[1]), zlim=(0.0, lengths[2]))
    # Label each detector coordinate and its unit.
    axis.set(xlabel="x [cm]", ylabel="y [cm]", zlabel="z [cm]")
    # Use physical box proportions while passing a copy because Matplotlib normalizes this array in place.
    axis.set_box_aspect(lengths.copy())
    # Choose an oblique view that exposes all three dimensions.
    axis.view_init(elev=22.0, azim=-58.0)
    # Add a concise title and event count.
    axis.set_title(f"Atmospheric-neutrino interaction vertices ({len(vertices):,} events)", pad=16)
    # Explain the z-based color encoding.
    figure.colorbar(points, ax=axis, shrink=0.68, pad=0.08, label="Vertex z [cm]")
    # Resolve spacing around the 3D labels.
    figure.tight_layout()
    # Save the event display as a high-resolution PNG.
    figure.savefig(output_path, dpi=220)
    # Release figure memory.
    plt.close(figure)


# Save orthogonal two-dimensional density projections of the same vertices.
def plot_projections(vertices: np.ndarray, lengths: np.ndarray, output_path: Path) -> None:
    # Create three side-by-side detector projections.
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    # Describe each projection using column indices, labels, and physical bounds.
    projections = [
        (0, 1, "x [cm]", "y [cm]", lengths[0], lengths[1], "x-y projection"),
        (0, 2, "x [cm]", "z [cm]", lengths[0], lengths[2], "x-z projection"),
        (1, 2, "y [cm]", "z [cm]", lengths[1], lengths[2], "y-z projection"),
    ]
    # Draw each projection with a common number of bins along each displayed coordinate.
    for axis, (horizontal, vertical, horizontal_label, vertical_label, horizontal_length, vertical_length, title) in zip(axes, projections):
        # Calculate a fixed-extent two-dimensional occupancy grid in detector coordinates.
        density, _, _ = np.histogram2d(vertices[:, horizontal], vertices[:, vertical], bins=(24, 24),
                                       range=((0.0, horizontal_length), (0.0, vertical_length)))
        # Render the transposed grid because image rows represent the displayed vertical coordinate.
        density_image = axis.imshow(density.T, origin="lower",
                                    extent=(0.0, horizontal_length, 0.0, vertical_length),
                                    aspect="equal", interpolation="nearest", cmap="magma")
        # Label this projection and its coordinates.
        axis.set(title=title, xlabel=horizontal_label, ylabel=vertical_label,
                 xlim=(0.0, horizontal_length), ylim=(0.0, vertical_length))
        # Add a separate count scale because equal physical bins have different areas across projections.
        figure.colorbar(density_image, ax=axis, fraction=0.046, pad=0.04, label="Events per bin")
    # Add a figure-level description.
    figure.suptitle("Orthogonal views of atmospheric-neutrino interaction vertices")
    # Resolve spacing while preserving the title.
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    # Save the projection figure.
    figure.savefig(output_path, dpi=220)
    # Release figure memory.
    plt.close(figure)


# Save one-dimensional occupancy and pull plots for quantitative uniformity checks.
def plot_uniformity(coordinates: dict[str, np.ndarray], lengths: dict[str, float], bins: int,
                    statistics: dict[str, dict[str, float]], output_path: Path) -> None:
    # Create one occupancy row and one standardized-residual row for x, y, and z.
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), sharex="col")
    # Process all detector coordinates with identical plotting conventions.
    for column, axis_name in enumerate(("x", "y", "z")):
        # Read this coordinate and configured active length.
        values = coordinates[axis_name]
        # Read the precomputed expected occupancy.
        expected = statistics[axis_name]["expected"]
        # Construct equal-width detector-coordinate bin edges.
        edges = np.linspace(0.0, lengths[axis_name], bins + 1)
        # Count observed interactions in each bin.
        counts, _ = np.histogram(values, bins=edges)
        # Calculate the bin centers used by the pull plot.
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Calculate approximate Poisson pulls from equal occupancy.
        pulls = (counts - expected) / np.sqrt(expected)
        # Draw the observed occupancy histogram.
        axes[0, column].stairs(counts, edges, fill=True, color=AXIS_COLORS[axis_name], alpha=0.72)
        # Draw the uniform expectation as a dark horizontal line.
        axes[0, column].axhline(expected, color="black", linewidth=1.3, label="Uniform expectation")
        # Draw a one-standard-deviation counting band around the expectation.
        axes[0, column].axhspan(expected - np.sqrt(expected), expected + np.sqrt(expected), color="0.5", alpha=0.18,
                                label=r"Expected $\pm1\sqrt{N}$")
        # Label the occupancy panel and quote its reduced chi-square diagnostic.
        axes[0, column].set(title=f"{axis_name} occupancy", ylabel="Events per bin")
        # Add the numeric test results inside the occupancy panel.
        axes[0, column].text(0.97, 0.95,
                             f"reduced chi-square = {statistics[axis_name]['reduced_chi_square']:.2f}\n"
                             f"max |pull| = {statistics[axis_name]['maximum_pull']:.2f}",
                             transform=axes[0, column].transAxes, ha="right", va="top", fontsize=9,
                             bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9})
        # Draw standardized bin residuals around zero.
        axes[1, column].bar(centers, pulls, width=np.diff(edges) * 0.88, color=AXIS_COLORS[axis_name], alpha=0.8)
        # Mark zero residual with a dark reference line.
        axes[1, column].axhline(0.0, color="black", linewidth=1.0)
        # Mark approximate two-standard-deviation guide lines.
        axes[1, column].axhline(2.0, color="0.4", linewidth=0.9, linestyle="--")
        # Mark the matching negative guide line.
        axes[1, column].axhline(-2.0, color="0.4", linewidth=0.9, linestyle="--")
        # Label the pull and coordinate axes.
        axes[1, column].set(xlabel=f"Vertex {axis_name} [cm]", ylabel="(observed - expected) / sqrt(expected)")
        # Set the full configured coordinate extent.
        axes[1, column].set_xlim(0.0, lengths[axis_name])
    # Show the expectation legend once to avoid repeating identical labels.
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    # Add a figure-level title that states the null hypothesis.
    figure.suptitle(f"Vertex uniformity along detector axes ({bins} equal-width bins per axis)")
    # Resolve spacing while preserving the title.
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    # Save the quantitative uniformity figure.
    figure.savefig(output_path, dpi=220)
    # Release figure memory.
    plt.close(figure)


# Run the complete vertex-distribution analysis.
def main() -> None:
    # Read command-line options.
    args = parse_args()
    # Normalize the input path for validation and reporting.
    input_path = args.input.expanduser().resolve()
    # Normalize the output directory.
    output_directory = args.output.expanduser().resolve()
    # Stop immediately when the requested ROOT file does not exist.
    if not input_path.is_file():
        # Include the exact resolved path in the exception.
        raise FileNotFoundError(f"Geant4 ROOT file does not exist: {input_path}")
    # Require positive active-volume dimensions.
    if min(args.x_length, args.y_length, args.z_length) <= 0.0:
        # Reject invalid geometry before normalizing coordinates.
        raise ValueError("All detector lengths must be positive.")
    # Require enough bins for a meaningful occupancy comparison.
    if args.bins < 2:
        # Reject a degenerate one-bin uniformity test.
        raise ValueError("--bins must be at least 2.")
    # Create the dedicated output directory and missing parents.
    output_directory.mkdir(parents=True, exist_ok=True)
    # Read and validate one interaction vertex per event.
    x_values, y_values, z_values, event_count = read_vertices(input_path)
    # Assemble the aligned coordinates into one event-by-three array.
    vertices = np.column_stack((x_values, y_values, z_values))
    # Store the configured detector dimensions as an ordered numeric array.
    length_array = np.array((args.x_length, args.y_length, args.z_length), dtype=float)
    # Store coordinates by name for plots and reports.
    coordinates = {"x": x_values, "y": y_values, "z": z_values}
    # Store dimensions by name for the same shared loops.
    lengths = {"x": args.x_length, "y": args.y_length, "z": args.z_length}
    # Identify vertices falling outside any configured active-volume boundary.
    outside = np.any((vertices < 0.0) | (vertices > length_array), axis=1)
    # Calculate uniformity diagnostics independently along each coordinate.
    statistics = {
        axis: uniformity_statistics(coordinates[axis], lengths[axis], args.bins)
        for axis in ("x", "y", "z")
    }
    # Define the stable 3D display output path.
    display_path = output_directory / "01_vertex_3d_display.png"
    # Define the stable orthogonal-projection output path.
    projections_path = output_directory / "02_vertex_orthogonal_projections.png"
    # Define the stable quantitative-uniformity output path.
    uniformity_path = output_directory / "03_vertex_uniformity_checks.png"
    # Produce the 3D point-cloud event display.
    plot_3d(vertices, length_array, display_path)
    # Produce the three orthogonal density projections.
    plot_projections(vertices, length_array, projections_path)
    # Produce the one-dimensional occupancy and pull checks.
    plot_uniformity(coordinates, lengths, args.bins, statistics, uniformity_path)
    # Build a concise text summary for batch validation and future regression comparisons.
    summary_lines = [
        "Atmospheric-neutrino interaction-vertex uniformity summary",
        f"Input: {input_path}",
        f"Events: {event_count}",
        f"Configured active volume [cm]: x=[0,{args.x_length:g}], y=[0,{args.y_length:g}], z=[0,{args.z_length:g}]",
        f"Vertices outside configured active volume: {int(np.count_nonzero(outside))}",
        f"Uniformity bins per axis: {args.bins}",
    ]
    # Add one line of observed range and uniformity diagnostics per coordinate.
    for axis in ("x", "y", "z"):
        # Append this coordinate's measured statistics in a machine-readable form.
        summary_lines.append(
            f"{axis}: min={np.min(coordinates[axis]):.6f} cm, max={np.max(coordinates[axis]):.6f} cm, "
            f"mean={np.mean(coordinates[axis]):.6f} cm, expected_mean={0.5 * lengths[axis]:.6f} cm, "
            f"reduced_chi_square={statistics[axis]['reduced_chi_square']:.4f}, "
            f"max_abs_pull={statistics[axis]['maximum_pull']:.4f}"
        )
    # Explain how to interpret the dependency-free statistics.
    summary_lines.extend([
        "",
        "Interpretation:",
        "A reduced chi-square near 1 is typical of statistical fluctuations around equal bin occupancy.",
        "Values modestly above or below 1 are expected in a finite sample; coherent trends across adjacent bins matter more.",
        "The pull panels should fluctuate around zero without a sustained slope, central excess, edge deficit, or periodic structure.",
    ])
    # Write the summary using a trailing newline for command-line readability.
    (output_directory / "vertex_uniformity_summary.txt").write_text("\n".join(summary_lines) + "\n")
    # Report the analyzed event count and containment result.
    print(f"Read {event_count:,} interaction vertices from {input_path}")
    # Report the number of out-of-volume vertices.
    print(f"Vertices outside configured active volume: {int(np.count_nonzero(outside))}")
    # Report each uniformity diagnostic to the terminal.
    for axis in ("x", "y", "z"):
        # Print reduced chi-square and maximum pull for this coordinate.
        print(f"{axis}: reduced chi-square={statistics[axis]['reduced_chi_square']:.3f}, "
              f"max |pull|={statistics[axis]['maximum_pull']:.3f}")
    # Confirm the output directory containing all products.
    print(f"Saved vertex checks to {output_directory}")


# Execute the workflow only when this file is run directly.
if __name__ == "__main__":
    # Enter the command-line analysis.
    main()

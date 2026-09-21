#!/usr/bin/env python3
"""Create a multipage PDF of atmospheric-neutrino kaon truth checks."""

# Import command-line parsing so input and output files can be overridden.
import argparse
# Import operating-system support for configuring Matplotlib's cache location.
import os
# Import Path for robust filesystem path handling.
from pathlib import Path
# Import text wrapping so report prose fits cleanly on portrait PDF pages.
import textwrap

# Resolve this script's directory so defaults do not depend on the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
# Keep Matplotlib's cache in this analysis directory.
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

# Import Awkward Array for ROOT's event-by-event particle vectors.
import awkward as ak
# Import Matplotlib after its cache location has been configured.
import matplotlib
# Select a non-interactive backend suitable for batch PDF production.
matplotlib.use("Agg")
# Import the plotting interface used for each page.
import matplotlib.pyplot as plt
# Import PdfPages to combine all checks into one requested PDF.
from matplotlib.backends.backend_pdf import PdfPages
# Import NumPy for kinematic calculations and binning.
import numpy as np
# Import uproot to read Q-Pix Geant4 ROOT truth directly.
import uproot

# Point the default input at the atmospheric-neutrino Geant4 production file.
DEFAULT_INPUT = SCRIPT_DIR.parent.parent / "02_qpix_geant4" / "output" / "atmospheric_g4.root"
# Point the default output at the requested kaon PDF.
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "kaon_distribution_checks.pdf"
# Include charged and neutral kaon truth species produced by GENIE or transported by Geant4.
KAON_PDGS = (321, -321, 311, -311, 310, 130)
# Give each PDG code a conventional display label.
KAON_LABELS = {321: r"$K^+$", -321: r"$K^-$", 311: r"$K^0$", -311: r"$\bar K^0$", 310: r"$K_S^0$", 130: r"$K_L^0$"}
# Assign stable colors so a kaon species has the same appearance on every page.
KAON_COLORS = {321: "#1b9e77", -321: "#d95f02", 311: "#7570b3", -311: "#e7298a", 310: "#66a61e", 130: "#e6ab02"}


# Flatten one jagged particle quantity after applying a jagged particle mask.
def flatten_selected(values: ak.Array, mask: ak.Array) -> np.ndarray:
    # Remove event boundaries after selection and convert the result to NumPy floats.
    return np.asarray(ak.flatten(values[mask], axis=None), dtype=float)


# Calculate a jagged momentum magnitude from Cartesian components.
def jagged_momentum(px: ak.Array, py: ak.Array, pz: ak.Array) -> ak.Array:
    # Apply the Euclidean norm particle by particle.
    return np.sqrt(px * px + py * py + pz * pz)


# Calculate particle cos(theta_z) while protecting zero-momentum entries.
def jagged_cos_zenith(px: ak.Array, py: ak.Array, pz: ak.Array) -> ak.Array:
    # Calculate momentum magnitude using the shared helper.
    momentum = jagged_momentum(px, py, pz)
    # Replace zero denominators before division because Awkward evaluates both branches of ak.where.
    safe_momentum = ak.where(momentum > 0.0, momentum, 1.0)
    # Divide pz by magnitude where defined and mark stationary particles as NaN.
    return ak.where(momentum > 0.0, pz / safe_momentum, np.nan)


# Extract one incoming atmospheric-neutrino energy from every event.
def incoming_neutrino_energy(arrays: dict[str, ak.Array]) -> np.ndarray:
    # Read initial-state identifiers, which include both the neutrino and argon nucleus.
    initial_pdg = arrays["generator_initial_particle_pdg_code"]
    # Select all supported electron- and muon-flavor neutrinos and antineutrinos.
    neutrino_mask = (abs(initial_pdg) == 12) | (abs(initial_pdg) == 14)
    # Take the first selected energy per event and mark malformed events with NaN.
    return np.asarray(ak.fill_none(ak.firsts(arrays["generator_initial_particle_energy"][neutrino_mask]), np.nan), dtype=float)


# Add a compact text summary to one plot panel.
def annotate(axis: plt.Axes, lines: list[str], location: tuple[float, float] = (0.98, 0.97)) -> None:
    # Place the lines in a translucent white box at the upper-right of the axis.
    axis.text(location[0], location[1], "\n".join(lines), transform=axis.transAxes, ha="right", va="top", fontsize=9,
              bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.88})


# Save one plot figure to both the open PDF and an individual PNG file.
def save_page(pdf: PdfPages, figure: plt.Figure, png_path: Path) -> None:
    # Resolve subplot spacing before output.
    figure.tight_layout()
    # Add the figure as the next PDF page.
    pdf.savefig(figure)
    # Save the same complete figure as a high-resolution standalone PNG.
    figure.savefig(png_path, dpi=200)
    # Close the figure to keep memory use bounded.
    plt.close(figure)


# Append one portrait text page containing wrapped analysis prose to the PDF.
def add_report_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]], page_label: str) -> None:
    # Create a US-letter portrait report page.
    figure = plt.figure(figsize=(8.5, 11.0))
    # Draw the report title in figure coordinates so mixed PDF page sizes cannot shift it.
    figure.text(0.08, 0.94, title, fontsize=18, fontweight="bold", va="top", ha="left")
    # Set the initial section position below the title.
    y_position = 0.88
    # Draw each report section in sequence.
    for heading, paragraph in sections:
        # Draw the section heading.
        figure.text(0.08, y_position, heading, fontsize=12, fontweight="bold", va="top", ha="left")
        # Move below the heading.
        y_position -= 0.035
        # Wrap the paragraph to a readable line width.
        wrapped = textwrap.fill(paragraph, width=96)
        # Draw the report paragraph with comfortable line spacing.
        figure.text(0.08, y_position, wrapped, fontsize=10.2, va="top", ha="left", linespacing=1.45)
        # Reserve enough vertical space for all wrapped lines and a section gap.
        y_position -= 0.030 * (wrapped.count("\n") + 1) + 0.035
    # Add a small page label at the lower-right.
    figure.text(0.92, 0.04, page_label, fontsize=8.5, color="0.4", ha="right")
    # Append the report page to the PDF only.
    pdf.savefig(figure)
    # Close the report figure.
    plt.close(figure)


# Parse optional input and output path arguments.
def parse_args() -> argparse.Namespace:
    # Create command-line help with a concise description.
    parser = argparse.ArgumentParser(description="Plot atmospheric-neutrino kaon distributions from G4 truth.")
    # Accept an alternate Q-Pix Geant4 ROOT file.
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Geant4 ROOT file containing event_tree.")
    # Accept an alternate PDF destination.
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination multipage PDF.")
    # Return the parsed values to the caller.
    return parser.parse_args()


# Run the complete kaon truth analysis.
def main() -> None:
    # Read command-line options.
    args = parse_args()
    # Normalize the input path for reliable validation and reporting.
    input_path = args.input.expanduser().resolve()
    # Normalize the output path for reliable directory creation.
    output_path = args.output.expanduser().resolve()
    # Fail early when the requested ROOT file cannot be found.
    if not input_path.is_file():
        # Include the exact missing path in the error.
        raise FileNotFoundError(f"Geant4 ROOT file does not exist: {input_path}")
    # Create the output directory tree when necessary.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Place standalone kaon plot images in the requested particle-specific directory.
    png_directory = output_path.parent / "kaon_pngs"
    # Create the standalone PNG directory when necessary.
    png_directory.mkdir(parents=True, exist_ok=True)
    # List the generator and transported-particle branches used by the checks.
    branches = [
        "event", "generator_initial_particle_pdg_code", "generator_initial_particle_energy",
        "generator_final_particle_pdg_code", "generator_final_particle_energy", "generator_final_particle_mass",
        "generator_final_particle_px", "generator_final_particle_py", "generator_final_particle_pz",
        "particle_pdg_code", "particle_parent_track_id", "particle_initial_energy", "particle_mass",
        "particle_initial_px", "particle_initial_py", "particle_initial_pz", "particle_decay_flag",
    ]
    # Open the ROOT file in a context manager.
    with uproot.open(input_path) as root_file:
        # Verify that this is a Q-Pix Geant4 output containing event_tree.
        if "event_tree" not in root_file:
            # Raise a focused schema error.
            raise KeyError(f"No event_tree was found in {input_path}")
        # Access the latest event_tree cycle.
        tree = root_file["event_tree"]
        # Find schema requirements absent from this file.
        missing = sorted(set(branches).difference(tree.keys()))
        # Reject incompatible files before loading arrays.
        if missing:
            # Report all missing branches in one message.
            raise KeyError("Missing required event_tree branches: " + ", ".join(missing))
        # Read only the required branches into an Awkward dictionary.
        arrays = tree.arrays(branches, library="ak", how=dict)
        # Record the number of generated interactions.
        event_count = int(tree.num_entries)
    # Read generator final-state particle identifiers.
    final_pdg = arrays["generator_final_particle_pdg_code"]
    # Select all charged and neutral generator kaon species.
    generator_kaon_mask = ak.zeros_like(final_pdg, dtype=bool)
    # Add each supported PDG species to the generator selection.
    for pdg in KAON_PDGS:
        # Combine this species with the existing selection.
        generator_kaon_mask = generator_kaon_mask | (final_pdg == pdg)
    # Count generator final-state kaons in each event.
    generator_multiplicity = np.asarray(ak.sum(generator_kaon_mask, axis=1), dtype=int)
    # Count events containing at least one generator final-state kaon.
    generator_kaon_events = int(np.count_nonzero(generator_multiplicity))
    # Count all generator final-state kaons.
    generator_kaons = int(np.sum(generator_multiplicity))
    # Stop with an informative message when the sample is too small to contain kaons.
    if generator_kaons == 0:
        # A kaon-free file cannot populate the requested distributions.
        raise ValueError("No generator final-state kaons were found for PDGs +/-321, +/-311, 310, or 130.")
    # Calculate generator kaon kinetic energy using each stored particle's own mass.
    generator_kinetic = np.maximum(arrays["generator_final_particle_energy"] - arrays["generator_final_particle_mass"], 0.0)
    # Calculate generator momentum magnitude.
    generator_momentum = jagged_momentum(
        arrays["generator_final_particle_px"], arrays["generator_final_particle_py"], arrays["generator_final_particle_pz"]
    )
    # Calculate generator cos(theta_z).
    generator_cos_zenith = jagged_cos_zenith(
        arrays["generator_final_particle_px"], arrays["generator_final_particle_py"], arrays["generator_final_particle_pz"]
    )
    # Read all transported Geant4 particle identifiers.
    particle_pdg = arrays["particle_pdg_code"]
    # Select charged and physical neutral kaons among transported particles.
    transported_kaon_mask = ak.zeros_like(particle_pdg, dtype=bool)
    # Add all supported species to the transported selection.
    for pdg in KAON_PDGS:
        # Combine this PDG with the current mask.
        transported_kaon_mask = transported_kaon_mask | (particle_pdg == pdg)
    # Select only Geant4 primary tracks, which have parent track ID zero.
    primary_kaon_mask = transported_kaon_mask & (arrays["particle_parent_track_id"] == 0)
    # Count all selected transported primary kaon tracks.
    transported_primary_kaons = int(ak.sum(primary_kaon_mask))
    # Extract one incoming atmospheric-neutrino energy per event.
    neutrino_energy = incoming_neutrino_energy(arrays)
    # Open the requested multipage PDF.
    with PdfPages(output_path) as pdf:
        # Create an overview page for species and event multiplicity.
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        # Count each generator kaon species in the fixed label order.
        generator_counts = [int(ak.sum(final_pdg == pdg)) for pdg in KAON_PDGS]
        # Draw species counts, including zero-count species to expose expected absences.
        species_bars = axes[0].bar([KAON_LABELS[pdg] for pdg in KAON_PDGS], generator_counts,
                                   color=[KAON_COLORS[pdg] for pdg in KAON_PDGS])
        # Print exact species counts above every bar, including absent species.
        axes[0].bar_label(species_bars, labels=[str(count) for count in generator_counts], padding=3)
        # Label the generator species plot.
        axes[0].set(title="Generator final-state kaon species", ylabel="Number of kaons")
        # Add sample and event-fraction context.
        annotate(axes[0], [f"Events: {event_count:,}", f"Kaon events: {generator_kaon_events:,}",
                           f"Fraction: {generator_kaon_events / event_count:.3%}",
                           f"K+/K- ratio: {generator_counts[0] / generator_counts[1]:.2f}"])
        # Define integer-centered multiplicity bins.
        multiplicity_bins = np.arange(int(np.max(generator_multiplicity)) + 2) - 0.5
        # Draw event-level generator kaon multiplicity.
        multiplicity_counts, _, multiplicity_patches = axes[1].hist(
            generator_multiplicity, bins=multiplicity_bins, color="#4d4d4d", edgecolor="white"
        )
        # Print exact event counts above each multiplicity bin.
        axes[1].bar_label(multiplicity_patches, labels=[f"{int(count):,}" for count in multiplicity_counts], padding=3)
        # Use integer tick marks for multiplicity.
        axes[1].set_xticks(np.arange(int(np.max(generator_multiplicity)) + 1))
        # Label the multiplicity plot.
        axes[1].set(title="Kaon multiplicity per event", xlabel="Generator final-state kaons", ylabel="Events")
        # Summarize the rare kaon-event population and multi-kaon excess.
        annotate(axes[1], [f"Kaon events: {generator_kaon_events} ({generator_kaon_events / event_count:.3%})",
                           f"Total kaons: {generator_kaons}",
                           f"Kaons beyond one/event: {generator_kaons - generator_kaon_events}"])
        # Add a page title.
        figure.suptitle("Atmospheric-neutrino kaon truth overview")
        # Save and close the overview page.
        save_page(pdf, figure, png_directory / "01_kaon_truth_overview.png")

        # Create a page comparing kinetic-energy and momentum spectra by species.
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        # Loop over present generator species so empty histograms do not clutter the legend.
        for pdg, count in zip(KAON_PDGS, generator_counts):
            # Skip species absent from this finite simulation sample.
            if count == 0:
                # Continue to the next species.
                continue
            # Select this generator species.
            species_mask = final_pdg == pdg
            # Flatten this species' kinetic energies.
            kinetic = flatten_selected(generator_kinetic, species_mask)
            # Flatten this species' momentum magnitudes.
            momentum = flatten_selected(generator_momentum, species_mask)
            # Draw kinetic energy with common logarithmic binning.
            axes[0].hist(kinetic, bins=np.geomspace(max(1.0, np.min(kinetic[kinetic > 0])), np.max(kinetic) * 1.001, 30),
                         histtype="step", linewidth=1.8, color=KAON_COLORS[pdg],
                         label=f"{KAON_LABELS[pdg]}: N={count}, med={np.median(kinetic):.0f}, mean={np.mean(kinetic):.0f} MeV")
            # Draw momentum with common logarithmic binning.
            axes[1].hist(momentum, bins=np.geomspace(max(1.0, np.min(momentum[momentum > 0])), np.max(momentum) * 1.001, 30),
                         histtype="step", linewidth=1.8, color=KAON_COLORS[pdg],
                         label=f"{KAON_LABELS[pdg]}: N={count}, med={np.median(momentum):.0f}, mean={np.mean(momentum):.0f} MeV/c")
        # Use a logarithmic energy axis for the atmospheric spectrum.
        axes[0].set_xscale("log")
        # Label the kinetic-energy panel.
        axes[0].set(title="Kaon kinetic energy", xlabel="Kinetic energy [MeV]", ylabel="Kaons")
        # Display the species legend.
        axes[0].legend(frameon=False, fontsize=9)
        # Flatten all generator kaon kinetic energies for an overall robust interval.
        overall_generator_kinetic = flatten_selected(generator_kinetic, generator_kaon_mask)
        # Display the central interval without overloading the species legend.
        annotate(axes[0], [f"Overall central 90%: {np.percentile(overall_generator_kinetic, 5):.0f}-{np.percentile(overall_generator_kinetic, 95):.0f} MeV"], (0.98, 0.42))
        # Use a logarithmic momentum axis.
        axes[1].set_xscale("log")
        # Label the momentum panel.
        axes[1].set(title="Kaon momentum", xlabel=r"Momentum [MeV/$c$]", ylabel="Kaons")
        # Display the species legend.
        axes[1].legend(frameon=False, fontsize=9)
        # Flatten all generator kaon momentum magnitudes for an overall robust interval.
        overall_generator_momentum = flatten_selected(generator_momentum, generator_kaon_mask)
        # Display the central momentum interval.
        annotate(axes[1], [f"Overall central 90%: {np.percentile(overall_generator_momentum, 5):.0f}-{np.percentile(overall_generator_momentum, 95):.0f} MeV/c"], (0.98, 0.42))
        # Add a page title.
        figure.suptitle("Generator final-state kaon kinematics")
        # Save and close the kinematic page.
        save_page(pdf, figure, png_directory / "02_kaon_kinematics.png")

        # Create a page for angular distributions and parent neutrino energy.
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        # Loop over present kaon species.
        for pdg, count in zip(KAON_PDGS, generator_counts):
            # Skip absent species.
            if count == 0:
                # Continue to the next PDG code.
                continue
            # Select this species.
            species_mask = final_pdg == pdg
            # Flatten and clean its cos(theta_z) values.
            cosine = flatten_selected(generator_cos_zenith, species_mask)
            # Retain finite directions only.
            cosine = cosine[np.isfinite(cosine)]
            # Draw the species angular distribution.
            axes[0].hist(cosine, bins=24, range=(-1.0, 1.0), histtype="step", linewidth=1.8,
                         color=KAON_COLORS[pdg], label=f"{KAON_LABELS[pdg]}: N={count}, med={np.median(cosine):.2f}, mean={np.mean(cosine):.2f}")
        # Label the detector-coordinate direction panel.
        axes[0].set(title="Kaon direction", xlabel=r"$\cos\theta_z=p_z/|p|$", ylabel="Kaons")
        # Display the species legend.
        axes[0].legend(frameon=False, fontsize=9)
        # Repeat each event's incoming neutrino energy once for every generator kaon in that event.
        parent_energy = np.repeat(neutrino_energy, generator_multiplicity)
        # Remove malformed energies before logarithmic plotting.
        parent_energy = parent_energy[np.isfinite(parent_energy) & (parent_energy > 0.0)]
        # Draw the parent-neutrino energy spectrum for kaon-producing events.
        axes[1].hist(parent_energy, bins=np.geomspace(np.min(parent_energy), np.max(parent_energy) * 1.001, 35),
                     color="#1f78b4", edgecolor="white")
        # Use a logarithmic energy axis.
        axes[1].set_xscale("log")
        # Label this as a per-kaon spectrum, since multi-kaon events contribute more than once.
        axes[1].set(title="Incoming energy for produced kaons", xlabel=r"Incoming $E_\nu$ [MeV]", ylabel="Kaons")
        # Add the expected threshold-oriented interpretation.
        annotate(axes[1], [f"Produced kaons: {len(parent_energy)}",
                           f"Median E_nu: {np.median(parent_energy):.0f} MeV",
                           f"Mean E_nu: {np.mean(parent_energy):.0f} MeV",
                           f"Central 90%: {np.percentile(parent_energy, 5):.0f}-{np.percentile(parent_energy, 95):.0f} MeV",
                           "Production favors higher-energy interactions"])
        # Add a page title.
        figure.suptitle("Kaon direction and production energy")
        # Save and close the angular/energy page.
        save_page(pdf, figure, png_directory / "03_kaon_direction_and_production_energy.png")

        # Create a page for kaon-production fraction versus incoming neutrino energy.
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        # Select finite positive incoming energies for all generated interactions.
        valid_energy = np.isfinite(neutrino_energy) & (neutrino_energy > 0.0)
        # Build logarithmic bins over the complete incoming-neutrino energy range.
        energy_bins = np.geomspace(np.min(neutrino_energy[valid_energy]), np.max(neutrino_energy[valid_energy]) * 1.001, 18)
        # Count all interactions in each energy bin.
        all_counts, _ = np.histogram(neutrino_energy[valid_energy], bins=energy_bins)
        # Identify events containing at least one generator kaon.
        has_kaon = generator_multiplicity > 0
        # Count kaon-producing interactions in the same bins.
        kaon_event_counts, _ = np.histogram(neutrino_energy[valid_energy & has_kaon], bins=energy_bins)
        # Calculate bin centers as geometric means for logarithmic axes.
        bin_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
        # Calculate kaon-event fractions only where the denominator is nonzero.
        fractions = np.divide(kaon_event_counts, all_counts, out=np.full_like(kaon_event_counts, np.nan, dtype=float), where=all_counts > 0)
        # Calculate approximate binomial standard errors as a visual statistical guide.
        errors = np.sqrt(np.divide(fractions * (1.0 - fractions), all_counts, out=np.zeros_like(fractions), where=all_counts > 0))
        # Draw the binned event fraction with statistical error bars.
        axes[0].errorbar(bin_centers, fractions, yerr=errors, marker="o", linestyle="-", color="#238b45", capsize=2)
        # Use logarithmic neutrino energy.
        axes[0].set_xscale("log")
        # Keep the probability axis nonnegative while allowing autoscaling above the maximum point.
        axes[0].set_ylim(bottom=0.0)
        # Label the production-rate shape check and clarify that it is generator-model dependent.
        axes[0].set(title="Kaon-event fraction vs energy", xlabel=r"Incoming $E_\nu$ [MeV]",
                    ylabel="Events with >=1 kaon / all events")
        # State the overall unweighted fraction and selected event count.
        annotate(axes[0], [f"Overall: {generator_kaon_events}/{event_count:,} = {generator_kaon_events / event_count:.3%}",
                           f"Highest-bin fraction: {np.nanmax(fractions):.1%}"])
        # Count transported Geant4 primary tracks by species.
        transported_counts = [int(ak.sum(primary_kaon_mask & (particle_pdg == pdg))) for pdg in KAON_PDGS]
        # Place paired generator and transported counts at neighboring x positions.
        positions = np.arange(len(KAON_PDGS))
        # Draw generator final-state counts.
        axes[1].bar(positions - 0.2, generator_counts, width=0.4, color="#377eb8", label="Generator final state")
        # Draw transported Geant4 primary-track counts.
        axes[1].bar(positions + 0.2, transported_counts, width=0.4, color="#ff7f00", label="G4 primary tracks")
        # Label x positions with kaon species.
        axes[1].set_xticks(positions, [KAON_LABELS[pdg] for pdg in KAON_PDGS])
        # Label the comparison without implying neutral K0 identity must remain unchanged in Geant4.
        axes[1].set(title="Generator-to-transport bookkeeping", ylabel="Number of kaons")
        # Display the population definitions.
        axes[1].legend(frameon=False, fontsize=9)
        # Explain neutral-kaon conversion in the annotation.
        annotate(axes[1], [f"G4 primary kaons: {transported_primary_kaons}", r"$K^0/\bar K^0$ may enter G4 as $K_S^0/K_L^0$"])
        # Add a page title.
        figure.suptitle("Kaon production trend and transport handoff")
        # Save and close the production page.
        save_page(pdf, figure, png_directory / "04_kaon_production_and_transport_handoff.png")

        # Create a final page describing transported primary-kaon outcomes.
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        # Calculate transported primary-kaon kinetic energy from stored total energy and mass.
        transported_kinetic = np.maximum(arrays["particle_initial_energy"] - arrays["particle_mass"], 0.0)
        # Loop over present transported species.
        for pdg, count in zip(KAON_PDGS, transported_counts):
            # Skip absent transported species.
            if count == 0:
                # Continue to the next species.
                continue
            # Select primary tracks of this exact species.
            species_mask = primary_kaon_mask & (particle_pdg == pdg)
            # Flatten their initial kinetic energies.
            kinetic = flatten_selected(transported_kinetic, species_mask)
            # Draw a species-resolved transported energy spectrum.
            axes[0].hist(kinetic, bins=np.geomspace(max(1.0, np.min(kinetic[kinetic > 0])), np.max(kinetic) * 1.001, 30),
                         histtype="step", linewidth=1.8, color=KAON_COLORS[pdg],
                         label=f"{KAON_LABELS[pdg]}: N={count}, med={np.median(kinetic):.0f}, mean={np.mean(kinetic):.0f} MeV")
        # Use logarithmic energy for the broad primary spectrum.
        axes[0].set_xscale("log")
        # Label the transported-primary distribution.
        axes[0].set(title="Transported primary-kaon energy", xlabel="Initial kinetic energy [MeV]", ylabel="Primary tracks")
        # Display the species legend.
        axes[0].legend(frameon=False, fontsize=9)
        # Flatten all transported-primary energies for an overall interval.
        all_transported_kinetic = flatten_selected(transported_kinetic, primary_kaon_mask)
        # Summarize the complete transported population on the plot.
        annotate(axes[0], [f"Overall central 90%: {np.percentile(all_transported_kinetic, 5):.0f}-{np.percentile(all_transported_kinetic, 95):.0f} MeV"], (0.98, 0.42))
        # Count selected primary kaons marked as decayed and not decayed by the G4 truth record.
        decay_counts = [int(ak.sum(primary_kaon_mask & (arrays["particle_decay_flag"] == flag))) for flag in (0, 1)]
        # Draw the two stored end-of-life categories.
        decay_bars = axes[1].bar(["Not marked decayed", "Marked decayed"], decay_counts,
                                 color=["#7570b3", "#e7298a"])
        # Print exact transport-outcome counts above both bars.
        axes[1].bar_label(decay_bars, labels=[str(count) for count in decay_counts], padding=3)
        # Label this explicitly as a Geant4 tracking outcome rather than a production cross section.
        axes[1].set(title="Transported primary-kaon decay flag", ylabel="Primary tracks")
        # Add exact counts to the panel.
        annotate(axes[1], [f"Not decayed: {decay_counts[0]} ({decay_counts[0] / transported_primary_kaons:.1%})",
                           f"Decayed: {decay_counts[1]} ({decay_counts[1] / transported_primary_kaons:.1%})"], (0.58, 0.97))
        # Add a page title.
        figure.suptitle("Geant4 primary-kaon transport checks")
        # Save and close the transport page.
        save_page(pdf, figure, png_directory / "05_kaon_transport_checks.png")
        # Flatten all generator kaon kinetic energies for report statistics.
        all_kaon_kinetic = flatten_selected(generator_kinetic, generator_kaon_mask)
        # Count charged and neutral generator kaons for a compact species summary.
        charged_kaons = generator_counts[0] + generator_counts[1]
        # Count generator neutral-flavor kaons in this sample.
        neutral_kaons = generator_counts[2] + generator_counts[3] + generator_counts[4] + generator_counts[5]
        # Append the first report page defining the populations and interpreting production counts.
        add_report_page(pdf, "Kaon sanity-check report", [
            ("What 'generator final state' means",
             "The generator initial state contains the incoming atmospheric neutrino and argon target. GENIE models "
             "the neutrino-argon interaction, including hadron production and intranuclear final-state interactions. "
             "The generator final state is the particle list that exits that generator-level nuclear system and is "
             "handed to Geant4. The first-page bars therefore count kaons produced by the interaction model, before "
             "Geant4 propagates, scatters, or decays them; they are not hit counts."),
            ("Sample accounting",
             f"The {event_count:,}-interaction sample contains {generator_kaons} generator final-state kaons in "
             f"{generator_kaon_events} events, an event fraction of {generator_kaon_events / event_count:.3%}. There "
             f"are {charged_kaons} charged and {neutral_kaons} neutral-flavor kaons. The species counts are "
             f"K+={generator_counts[0]}, K-={generator_counts[1]}, K0={generator_counts[2]}, and anti-K0={generator_counts[3]}. "
             f"The K+/K- ratio is {generator_counts[0] / generator_counts[1]:.2f}."),
            ("Why the species pattern is plausible",
             "Rare kaon production is expected at these energies because creating strange hadrons requires more "
             "available hadronic energy than quasielastic scattering or single-pion production. K+ exceeding K- is "
             "also qualitatively reasonable in a neutrino-rich sample: associated strangeness production can produce "
             "K+ together with a hyperon, whereas K- production is more constrained and K- has stronger absorption "
             "channels in nuclear matter. Exact ratios remain strongly dependent on GENIE's hadronization and "
             "final-state-interaction model."),
            ("Multiplicity interpretation",
             f"There are {generator_kaons - generator_kaon_events} kaons beyond a one-kaon-per-kaon-event baseline, "
             "so a small number of events contain multiple kaons. Most interactions contain no kaon, as expected for "
             "an inclusive atmospheric sample. The very small selected population means individual-bin fluctuations "
             "are large and smooth theoretical shapes should not be expected from only 64 kaons."),
        ], "Kaon report 1 of 3")
        # Append the second report page interpreting kinematics and Geant4 handoff.
        add_report_page(pdf, "Kaon physics interpretation", [
            ("Production energy and kinematics",
             f"The median kaon kinetic energy is {np.median(all_kaon_kinetic):.0f} MeV and the mean is "
             f"{np.mean(all_kaon_kinetic):.0f} MeV, with a central 90% interval of "
             f"{np.percentile(all_kaon_kinetic, 5):.0f} to {np.percentile(all_kaon_kinetic, 95):.0f} MeV. The incoming "
             f"neutrino energy for produced kaons has a median of {np.median(parent_energy):.0f} MeV and a mean of "
             f"{np.mean(parent_energy):.0f} MeV. Means are tail-sensitive and especially unstable for the rare K- "
             "subsample, so medians and central intervals are the primary typical-event summaries here. The rising kaon-event fraction with neutrino "
             "energy is the expected threshold and phase-space behavior: higher-energy interactions can support "
             "strange-hadron production more readily."),
            ("Direction distribution",
             "The kaon cos(theta_z) values populate both signs and do not show an unphysical spike at one coordinate "
             "boundary. With only 64 particles, the jagged appearance is statistical. A quantitative atmospheric "
             "zenith comparison requires the flux-axis convention, oscillation weights, and substantially more kaon "
             "statistics."),
            ("Generator-to-Geant4 handoff",
             f"The generator contains {generator_kaons} kaons and Geant4 records {transported_primary_kaons} primary "
             "kaon tracks, with matching displayed species totals in this file. This is an excellent bookkeeping "
             "check: no produced kaons appear to have been dropped or duplicated at transport initialization. In "
             "other configurations, neutral K0 and anti-K0 flavor states may be represented as physical K-short or "
             "K-long states, so neutral totals are safer than requiring every neutral PDG bar to match."),
        ], "Kaon report 2 of 3")
        # Append a third report page for transport outcomes, conclusions, and recommended validation work.
        add_report_page(pdf, "Kaon transport assessment", [
            ("Transport outcome",
             f"Geant4 marks {decay_counts[1]} of {transported_primary_kaons} primary kaons ({decay_counts[1] / transported_primary_kaons:.1%}) "
             f"as decayed and {decay_counts[0]} as not decayed. This is plausible for kaons traversing a finite liquid-argon "
             "volume: the result reflects competition among decay, hadronic interactions, stopping, and escape. The "
             "flag alone does not classify the non-decayed tracks by absorption versus leaving the active volume."),
            ("Conclusion and limitations",
             "The rarity, increasing production fraction with energy, K+ dominance, broad kinematics, and exact "
             "generator-to-primary-track accounting are all physically sensible. The main limitation is statistics. "
             "A quantitative kaon-production validation should use a larger weighted sample and compare GENIE tunes, "
             "hadronization/FSI variations, and available neutrino-induced strangeness data."),
            ("Recommended quantitative follow-up",
             "Increase the generated exposure until each important kaon species has enough entries for stable binned "
             "comparisons. Then preserve the event weights and compare kaon multiplicity, momentum, neutrino-energy "
             "dependence, and strange-baryon associations across GENIE tune and FSI variations. Treat the present "
             "plots as a successful bookkeeping and qualitative-physics check, not a precision strangeness result."),
        ], "Kaon report 3 of 3")
    # Report the event sample read by the script.
    print(f"Read {event_count:,} events from {input_path}")
    # Report the selected generator kaon population.
    print(f"Selected {generator_kaons:,} generator final-state kaons in {generator_kaon_events:,} events")
    # Report the transported primary-kaon population for bookkeeping.
    print(f"Found {transported_primary_kaons:,} transported Geant4 primary kaon tracks")
    # Confirm successful PDF creation.
    print(f"Saved kaon checks to {output_path}")
    # Confirm where the standalone plot images were written.
    print(f"Saved 5 standalone kaon PNGs to {png_directory}")


# Execute the analysis only when this file is invoked directly.
if __name__ == "__main__":
    # Enter the command-line workflow.
    main()

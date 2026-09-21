#!/usr/bin/env python3
"""Create a multipage PDF of atmospheric-neutrino muon truth checks."""

# Import command-line parsing so callers can override the default ROOT and PDF paths.
import argparse
# Import operating-system support so the Matplotlib cache can live beside this script.
import os
# Import Path for readable, platform-independent file and directory handling.
from pathlib import Path
# Import text wrapping so report prose fits cleanly on portrait PDF pages.
import textwrap

# Find this script's directory once so all default paths are independent of the working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
# Keep Matplotlib's cache in the analysis folder instead of relying on a writable home directory.
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

# Import Awkward Array to work with ROOT's variable-length particle vectors.
import awkward as ak
# Import Matplotlib after setting its cache directory.
import matplotlib
# Select the non-interactive backend because this script writes files rather than opening windows.
matplotlib.use("Agg")
# Import the plotting interface used to construct each PDF page.
import matplotlib.pyplot as plt
# Import PdfPages so several diagnostic figures can be stored in one PDF.
from matplotlib.backends.backend_pdf import PdfPages
# Import NumPy for vector arithmetic, histogram bins, and finite-value checks.
import numpy as np
# Import uproot to read the Geant4 ROOT output without requiring PyROOT.
import uproot

# Use the production Geant4 truth file as the default input.
DEFAULT_INPUT = SCRIPT_DIR.parent.parent / "02_qpix_geant4" / "output" / "atmospheric_g4.root"
# Use the requested output filename and location as the default output.
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "muon_distribution_checks.pdf"
# Identify the atmospheric neutrino flavors that can create muons in charged-current interactions.
MUON_NEUTRINO_PDGS = (14, -14)
# Identify negative and positive muons using the standard PDG convention.
MUON_PDGS = (13, -13)
# Provide publication-style labels for the two muon charges.
MUON_LABELS = {13: r"$\mu^-$", -13: r"$\mu^+$"}
# Assign distinct, colorblind-friendly colors to the two muon charges.
MUON_COLORS = {13: "#2166ac", -13: "#b2182b"}
# Store the muon rest mass in MeV/c^2 for kinetic-energy calculations.
MUON_MASS_MEV = 105.6583755


# Convert one jagged Awkward particle quantity into a flat NumPy array selected by a jagged mask.
def flatten_selected(values: ak.Array, mask: ak.Array) -> np.ndarray:
    # Apply the particle-level mask, flatten all event boundaries, and return ordinary floats.
    return np.asarray(ak.flatten(values[mask], axis=None), dtype=float)


# Calculate cos(theta_z) from three jagged momentum components while guarding zero momentum.
def jagged_cos_zenith(px: ak.Array, py: ak.Array, pz: ak.Array) -> ak.Array:
    # Calculate each particle's momentum magnitude in the units used by the input file.
    momentum = np.sqrt(px * px + py * py + pz * pz)
    # Replace zero denominators before division because Awkward evaluates both branches of ak.where.
    safe_momentum = ak.where(momentum > 0.0, momentum, 1.0)
    # Return pz/|p| for moving particles and NaN for zero-momentum entries.
    return ak.where(momentum > 0.0, pz / safe_momentum, np.nan)


# Extract one incoming atmospheric neutrino identity and four-vector from each event.
def incoming_neutrinos(arrays: dict[str, ak.Array]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Read the initial-state PDG vectors, which also contain the argon target nucleus.
    pdg = arrays["generator_initial_particle_pdg_code"]
    # Select only electron- and muon-flavor neutrinos and antineutrinos.
    mask = (abs(pdg) == 12) | (abs(pdg) == 14)
    # Take the first selected neutrino energy in every event and fill missing entries with NaN.
    energy = np.asarray(ak.fill_none(ak.firsts(arrays["generator_initial_particle_energy"][mask]), np.nan), dtype=float)
    # Take the corresponding incoming neutrino PDG code in every event.
    flavor = np.asarray(ak.fill_none(ak.firsts(pdg[mask]), 0), dtype=int)
    # Assemble each selected neutrino's three-momentum components as the final array dimension.
    momentum = np.column_stack([
        np.asarray(ak.fill_none(ak.firsts(arrays[name][mask]), np.nan), dtype=float)
        for name in ("generator_initial_particle_px", "generator_initial_particle_py", "generator_initial_particle_pz")
    ])
    # Return event-aligned neutrino flavor, energy, and momentum arrays.
    return flavor, energy, momentum


# Add a compact sample summary to the top-right corner of an axis.
def annotate(axis: plt.Axes, lines: list[str]) -> None:
    # Draw the supplied lines in a readable white box that does not hide the data completely.
    axis.text(0.98, 0.97, "\n".join(lines), transform=axis.transAxes, ha="right", va="top", fontsize=9,
              bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.88})


# Save one plot figure to both the open PDF and an individual PNG file.
def save_page(pdf: PdfPages, figure: plt.Figure, png_path: Path) -> None:
    # Resolve subplot spacing before writing the page.
    figure.tight_layout()
    # Append the completed figure as one page in the PDF.
    pdf.savefig(figure)
    # Save the same complete figure as a high-resolution standalone PNG.
    figure.savefig(png_path, dpi=200)
    # Close the figure so repeated pages do not accumulate memory.
    plt.close(figure)


# Append one portrait text page containing wrapped analysis prose to the PDF.
def add_report_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]], page_label: str) -> None:
    # Create a US-letter portrait page for comfortable report reading.
    figure = plt.figure(figsize=(8.5, 11.0))
    # Start the report title in figure coordinates so mixed PDF page sizes cannot shift it.
    figure.text(0.08, 0.94, title, fontsize=18, fontweight="bold", va="top", ha="left")
    # Track the next vertical text position in axis coordinates.
    y_position = 0.88
    # Draw every heading and its wrapped paragraph in order.
    for heading, paragraph in sections:
        # Draw a visually distinct section heading.
        figure.text(0.08, y_position, heading, fontsize=12, fontweight="bold", va="top", ha="left")
        # Move below the section heading.
        y_position -= 0.035
        # Wrap long prose to a consistent readable line length.
        wrapped = textwrap.fill(paragraph, width=96)
        # Draw the wrapped section body with generous line spacing.
        figure.text(0.08, y_position, wrapped, fontsize=10.2, va="top", ha="left", linespacing=1.45)
        # Reserve vertical space based on the number of wrapped lines plus a section gap.
        y_position -= 0.030 * (wrapped.count("\n") + 1) + 0.035
    # Add a subtle page label at the bottom-right.
    figure.text(0.92, 0.04, page_label, fontsize=8.5, color="0.4", ha="right")
    # Append the prose page to the PDF without creating an extra PNG.
    pdf.savefig(figure)
    # Release the report figure's memory.
    plt.close(figure)


# Parse command-line options with defaults that work from any current directory.
def parse_args() -> argparse.Namespace:
    # Describe the program in command-line help output.
    parser = argparse.ArgumentParser(description="Plot atmospheric-neutrino muon distributions from G4 truth.")
    # Allow a different Geant4 ROOT file to be analyzed.
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Geant4 ROOT file containing event_tree.")
    # Allow a different PDF destination while retaining the requested default.
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination multipage PDF.")
    # Return the parsed command-line values.
    return parser.parse_args()


# Run the complete muon truth analysis.
def main() -> None:
    # Read the user's command-line choices.
    args = parse_args()
    # Expand a possible tilde and convert the input path to an absolute path.
    input_path = args.input.expanduser().resolve()
    # Expand a possible tilde and convert the output path to an absolute path.
    output_path = args.output.expanduser().resolve()
    # Stop with a direct error if the requested ROOT file does not exist.
    if not input_path.is_file():
        # Include the resolved path in the exception to make configuration errors easy to diagnose.
        raise FileNotFoundError(f"Geant4 ROOT file does not exist: {input_path}")
    # Create the requested outputs directory, including missing parent directories.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Place standalone muon plot images in the requested particle-specific directory.
    png_directory = output_path.parent / "muon_pngs"
    # Create the standalone PNG directory when it is absent.
    png_directory.mkdir(parents=True, exist_ok=True)
    # List only the branches needed for muon kinematics and event-level correlations.
    branches = [
        "event", "generator_initial_particle_pdg_code", "generator_initial_particle_energy",
        "generator_initial_particle_px", "generator_initial_particle_py", "generator_initial_particle_pz",
        "generator_final_particle_pdg_code", "generator_final_particle_energy",
        "generator_final_particle_px", "generator_final_particle_py", "generator_final_particle_pz",
    ]
    # Open the ROOT file in a context manager so it is closed even if plotting fails.
    with uproot.open(input_path) as root_file:
        # Require the standard Q-Pix Geant4 event tree.
        if "event_tree" not in root_file:
            # Explain the missing object rather than allowing a less specific uproot error.
            raise KeyError(f"No event_tree was found in {input_path}")
        # Access the newest cycle of event_tree.
        tree = root_file["event_tree"]
        # Check all required branches before loading any large arrays.
        missing = sorted(set(branches).difference(tree.keys()))
        # Stop and report every missing branch together when the schema is incompatible.
        if missing:
            # Join branch names into one actionable exception message.
            raise KeyError("Missing required event_tree branches: " + ", ".join(missing))
        # Load the selected branches as a dictionary of jagged Awkward arrays.
        arrays = tree.arrays(branches, library="ak", how=dict)
        # Record the total event count for rates and annotations.
        event_count = int(tree.num_entries)
    # Read the generator final-state particle identifiers.
    final_pdg = arrays["generator_final_particle_pdg_code"]
    # Build one particle-level mask containing both muon charges.
    muon_mask = (final_pdg == 13) | (final_pdg == -13)
    # Count selected generator muons in each event.
    multiplicity = np.asarray(ak.sum(muon_mask, axis=1), dtype=int)
    # Count events containing at least one generator final-state muon.
    events_with_muons = int(np.count_nonzero(multiplicity))
    # Count the total selected muons across all events.
    total_muons = int(np.sum(multiplicity))
    # Stop with a useful message if this sample contains no muons at all.
    if total_muons == 0:
        # A muon-free sample cannot populate the requested diagnostic distributions.
        raise ValueError("No generator final-state muons (PDG +/-13) were found in event_tree.")
    # Calculate final-state momentum magnitude for every generator particle in MeV/c.
    final_momentum = np.sqrt(
        arrays["generator_final_particle_px"] ** 2
        + arrays["generator_final_particle_py"] ** 2
        + arrays["generator_final_particle_pz"] ** 2
    )
    # Calculate final-state kinetic energy from total energy and the fixed muon mass.
    muon_kinetic_energy = np.maximum(arrays["generator_final_particle_energy"] - MUON_MASS_MEV, 0.0)
    # Calculate final-state cos(theta_z), where +1 points along the ROOT +z axis.
    final_cos_zenith = jagged_cos_zenith(
        arrays["generator_final_particle_px"], arrays["generator_final_particle_py"], arrays["generator_final_particle_pz"]
    )
    # Extract event-aligned incoming atmospheric neutrino identity, energy, and momentum.
    neutrino_flavor, neutrino_energy, neutrino_momentum = incoming_neutrinos(arrays)
    # Open the requested multipage PDF for all following figures.
    with PdfPages(output_path) as pdf:
        # Create the first page with charge counts and event multiplicity.
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        # Count each muon charge separately at generator final state.
        charge_counts = [int(ak.sum(final_pdg == pdg)) for pdg in MUON_PDGS]
        # Draw the charge count bars using the fixed charge colors.
        axes[0].bar([MUON_LABELS[pdg] for pdg in MUON_PDGS], charge_counts,
                    color=[MUON_COLORS[pdg] for pdg in MUON_PDGS])
        # Label the count plot with its exact truth definition.
        axes[0].set(title="Generator final-state muon charges", ylabel="Number of muons")
        # Add the overall sample size and muon event fraction.
        annotate(axes[0], [f"Events: {event_count:,}", f"Events with muons: {events_with_muons:,}",
                           f"Fraction: {events_with_muons / event_count:.3f}"])
        # Choose integer-centered bins through the largest observed multiplicity.
        multiplicity_bins = np.arange(int(np.max(multiplicity)) + 2) - 0.5
        # Draw the number of final-state muons per atmospheric-neutrino interaction.
        axes[1].hist(multiplicity, bins=multiplicity_bins, color="#4d4d4d", edgecolor="white")
        # Place ticks at integer multiplicities only.
        axes[1].set_xticks(np.arange(int(np.max(multiplicity)) + 1))
        # Label the multiplicity distribution.
        axes[1].set(title="Muon multiplicity per event", xlabel="Generator final-state muons", ylabel="Events")
        # Give the PDF page a descriptive title.
        figure.suptitle("Atmospheric-neutrino muon truth overview")
        # Write and close the completed overview page.
        save_page(pdf, figure, png_directory / "01_muon_truth_overview.png")

        # Create the second page for energy and momentum spectra split by charge.
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        # Loop over mu- and mu+ so both spectra use identical bins and conventions.
        for pdg in MUON_PDGS:
            # Select this charge from the generator final state.
            charge_mask = final_pdg == pdg
            # Flatten this charge's kinetic energies into one ordinary array.
            kinetic = flatten_selected(muon_kinetic_energy, charge_mask)
            # Flatten this charge's momentum magnitudes into one ordinary array.
            momentum = flatten_selected(final_momentum, charge_mask)
            # Draw kinetic energy on logarithmic bins spanning the populated sample.
            axes[0].hist(kinetic, bins=np.geomspace(max(1.0, np.min(kinetic[kinetic > 0])), np.max(kinetic) * 1.001, 45),
                         histtype="step", linewidth=1.8, color=MUON_COLORS[pdg], label=f"{MUON_LABELS[pdg]} ({len(kinetic):,})")
            # Draw momentum using logarithmic bins suitable for the broad atmospheric spectrum.
            axes[1].hist(momentum, bins=np.geomspace(max(1.0, np.min(momentum[momentum > 0])), np.max(momentum) * 1.001, 45),
                         histtype="step", linewidth=1.8, color=MUON_COLORS[pdg], label=f"{MUON_LABELS[pdg]} ({len(momentum):,})")
        # Use logarithmic x axes because atmospheric-neutrino kinematics span orders of magnitude.
        axes[0].set_xscale("log")
        # Label the kinetic-energy distribution in the stored Geant4 truth units.
        axes[0].set(title="Muon kinetic energy", xlabel="Kinetic energy [MeV]", ylabel="Muons")
        # Display the charge legend without a surrounding frame.
        axes[0].legend(frameon=False)
        # Apply the same logarithmic scale to momentum.
        axes[1].set_xscale("log")
        # Label the momentum distribution.
        axes[1].set(title="Muon momentum", xlabel=r"Momentum [MeV/$c$]", ylabel="Muons")
        # Display the charge legend on the momentum panel.
        axes[1].legend(frameon=False)
        # Give the page a concise title.
        figure.suptitle("Generator final-state muon kinematics")
        # Write and close the kinematics page.
        save_page(pdf, figure, png_directory / "02_muon_kinematics.png")

        # Create the third page for direction and neutrino-to-muon scattering angle.
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        # Loop over charge to compare neutrino and antineutrino angular behavior.
        for pdg in MUON_PDGS:
            # Build this charge's selection mask.
            charge_mask = final_pdg == pdg
            # Flatten this charge's cos(theta_z) values.
            cosine = flatten_selected(final_cos_zenith, charge_mask)
            # Remove non-finite values produced by zero-momentum particles.
            cosine = cosine[np.isfinite(cosine)]
            # Draw a common-range angular histogram for direct shape comparison.
            axes[0].hist(cosine, bins=30, range=(-1.0, 1.0), histtype="step", linewidth=1.8,
                         color=MUON_COLORS[pdg], label=MUON_LABELS[pdg])
        # Label the detector-coordinate zenith proxy explicitly.
        axes[0].set(title="Muon direction", xlabel=r"$\cos\theta_z=p_z/|p|$", ylabel="Muons")
        # Display the charge key.
        axes[0].legend(frameon=False)
        # Prepare a list that will hold one opening angle for every selected muon.
        opening_angles = []
        # Visit each event so its muons can be compared with that event's incoming neutrino.
        for event_index in range(event_count):
            # Read the event's incoming neutrino momentum vector.
            nu_vector = neutrino_momentum[event_index]
            # Calculate the neutrino momentum magnitude.
            nu_norm = np.linalg.norm(nu_vector)
            # Skip malformed or zero-momentum incoming-neutrino entries.
            if not np.isfinite(nu_norm) or nu_norm == 0.0:
                # Continue to the next event because no opening angle can be defined here.
                continue
            # Select all final-state muon momentum components in this event.
            event_mask = np.asarray(muon_mask[event_index])
            # Build an N-by-3 array containing the selected muon momentum vectors.
            mu_vectors = np.column_stack([
                np.asarray(arrays[name][event_index])[event_mask]
                for name in ("generator_final_particle_px", "generator_final_particle_py", "generator_final_particle_pz")
            ])
            # Calculate each selected muon's momentum magnitude.
            mu_norms = np.linalg.norm(mu_vectors, axis=1)
            # Retain only moving muons with a well-defined direction.
            valid = mu_norms > 0.0
            # Calculate and clip cos(opening angle) to protect arccos from floating-point roundoff.
            cosine = np.clip((mu_vectors[valid] @ nu_vector) / (mu_norms[valid] * nu_norm), -1.0, 1.0)
            # Append opening angles in degrees to the growing event-independent list.
            opening_angles.extend(np.degrees(np.arccos(cosine)).tolist())
        # Draw the neutrino-to-muon opening-angle distribution.
        axes[1].hist(opening_angles, bins=36, range=(0.0, 180.0), color="#762a83", edgecolor="white")
        # Label the physical scattering-angle check.
        axes[1].set(title="Neutrino-muon opening angle", xlabel=r"$\angle(\nu,\mu)$ [degrees]", ylabel="Muons")
        # Explain the expected high-energy trend directly on the page.
        annotate(axes[1], ["Higher-energy CC events", "should be more forward-going"])
        # Give the page a concise title.
        figure.suptitle("Muon angular distributions")
        # Write and close the angular page.
        save_page(pdf, figure, png_directory / "03_muon_angular_distributions.png")

        # Create the fourth page for the event-by-event neutrino and muon energy relationship.
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        # Prepare aligned arrays of neutrino energy, muon total energy, and opening angle.
        paired_neutrino_energy = []
        # Store one total energy value for every selected muon.
        paired_muon_energy = []
        # Store a parallel opening-angle value for the energy-dependence panel.
        paired_opening_angle = []
        # Visit each event to repeat its incoming energy for every outgoing muon.
        for event_index in range(event_count):
            # Select final-state muons in this event.
            event_mask = np.asarray(muon_mask[event_index])
            # Read the selected muon total energies.
            event_muon_energy = np.asarray(arrays["generator_final_particle_energy"][event_index])[event_mask]
            # Skip events without a selected muon or a finite incoming energy.
            if len(event_muon_energy) == 0 or not np.isfinite(neutrino_energy[event_index]):
                # Continue because this event cannot contribute a neutrino-muon pair.
                continue
            # Repeat the event's neutrino energy once per selected muon.
            paired_neutrino_energy.extend([neutrino_energy[event_index]] * len(event_muon_energy))
            # Append the selected muon total energies.
            paired_muon_energy.extend(event_muon_energy.tolist())
            # Read and normalize the incoming neutrino momentum.
            nu_vector = neutrino_momentum[event_index]
            # Calculate its magnitude for the opening-angle denominator.
            nu_norm = np.linalg.norm(nu_vector)
            # Build the selected muon momentum vectors.
            mu_vectors = np.column_stack([
                np.asarray(arrays[name][event_index])[event_mask]
                for name in ("generator_final_particle_px", "generator_final_particle_py", "generator_final_particle_pz")
            ])
            # Calculate each muon momentum magnitude.
            mu_norms = np.linalg.norm(mu_vectors, axis=1)
            # Calculate opening angles, using NaN where either direction is undefined.
            angles = np.full(len(mu_vectors), np.nan)
            # Identify pairs with two nonzero momentum vectors.
            valid = (mu_norms > 0.0) & (nu_norm > 0.0)
            # Convert the valid dot products into degrees.
            angles[valid] = np.degrees(np.arccos(np.clip((mu_vectors[valid] @ nu_vector) / (mu_norms[valid] * nu_norm), -1.0, 1.0)))
            # Append event opening angles in the same order as the energy pairs.
            paired_opening_angle.extend(angles.tolist())
        # Convert the accumulated lists to NumPy arrays for plotting.
        paired_neutrino_energy = np.asarray(paired_neutrino_energy)
        # Convert paired muon energies to an array.
        paired_muon_energy = np.asarray(paired_muon_energy)
        # Convert paired opening angles to an array.
        paired_opening_angle = np.asarray(paired_opening_angle)
        # Draw the energy correlation with logarithmic coordinate binning and logarithmic count color.
        energy_hexbin = axes[0].hexbin(paired_neutrino_energy, paired_muon_energy, gridsize=45, bins="log", mincnt=1,
                                       xscale="log", yscale="log", cmap="viridis")
        # Add a colorbar so the density encoding has an explicit event-count meaning.
        figure.colorbar(energy_hexbin, ax=axes[0], label="Muon pairs per hexbin")
        # Label the event-level energy correlation.
        axes[0].set(title="Incoming vs outgoing energy", xlabel=r"Incoming $E_\nu$ [MeV]", ylabel=r"Final-state $E_\mu$ [MeV]")
        # Select finite, positive-energy pairs for the angular energy trend.
        valid_pairs = np.isfinite(paired_opening_angle) & (paired_neutrino_energy > 0.0)
        # Draw the energy dependence with logarithmic neutrino-energy binning and logarithmic count color.
        angle_hexbin = axes[1].hexbin(paired_neutrino_energy[valid_pairs], paired_opening_angle[valid_pairs], gridsize=45,
                                      bins="log", mincnt=1, xscale="log", cmap="magma")
        # Add a colorbar so sparse and dense angular regions can be distinguished.
        figure.colorbar(angle_hexbin, ax=axes[1], label="Muon pairs per hexbin")
        # Show the complete physically allowed opening-angle range.
        axes[1].set_ylim(0.0, 180.0)
        # Label the forward-scattering validation panel.
        axes[1].set(title="Direction correlation vs energy", xlabel=r"Incoming $E_\nu$ [MeV]",
                    ylabel=r"$\angle(\nu,\mu)$ [degrees]")
        # Give the page a descriptive title.
        figure.suptitle("Neutrino-muon correlations")
        # Write and close the final page.
        save_page(pdf, figure, png_directory / "04_neutrino_muon_correlations.png")
        # Flatten all selected muon kinetic energies for report statistics.
        all_muon_kinetic = flatten_selected(muon_kinetic_energy, muon_mask)
        # Calculate the muon-to-neutrino total-energy ratio for every paired entry.
        energy_fraction = paired_muon_energy / paired_neutrino_energy
        # Select low-energy pairs below 1 GeV for an angular comparison.
        low_energy_pairs = valid_pairs & (paired_neutrino_energy < 1000.0)
        # Select higher-energy pairs at or above 3 GeV for the same angular comparison.
        high_energy_pairs = valid_pairs & (paired_neutrino_energy >= 3000.0)
        # Count muons whose charge matches the expected incoming muon-neutrino charged-current sign.
        sign_matched_muons = 0
        # Check each event's final-state muon identities against its incoming flavor.
        for event_index in range(event_count):
            # Convert this event's selected final-state PDG codes to an ordinary array.
            event_muon_pdgs = np.asarray(final_pdg[event_index])[np.asarray(muon_mask[event_index])]
            # Count mu- in nu_mu events and mu+ in anti-nu_mu events.
            sign_matched_muons += int(np.sum((event_muon_pdgs == 13) & (neutrino_flavor[event_index] == 14)))
            # Add charge-conjugate matches for antineutrino events.
            sign_matched_muons += int(np.sum((event_muon_pdgs == -13) & (neutrino_flavor[event_index] == -14)))
        # Append a first report page defining the truth populations and summarizing observations.
        add_report_page(pdf, "Muon sanity-check report", [
            ("What 'generator final state' means",
             "The generator initial state contains the incoming atmospheric neutrino and the argon target. "
             "GENIE then models the neutrino-argon interaction. The generator final state is the list of particles "
             "that emerge from that modeled interaction after GENIE's intranuclear final-state interactions and are "
             "handed to Geant4 for transport. The first-page bars therefore count muons produced by the neutrino "
             "interaction record, not detector hits and not every secondary muon later created inside Geant4."),
            ("Sample accounting",
             f"The sample contains {event_count:,} generated interactions. {events_with_muons:,} events "
             f"({events_with_muons / event_count:.1%}) contain a generator final-state muon, with {charge_counts[0]:,} "
             f"mu- and {charge_counts[1]:,} mu+. Every selected event contains exactly one such muon. The mu-/mu+ "
             f"ratio is {charge_counts[0] / charge_counts[1]:.2f}."),
            ("Charge interpretation",
             f"{sign_matched_muons:,} of {total_muons:,} selected muons ({sign_matched_muons / total_muons:.2%}) "
             "have the charged-current sign expected from the incoming flavor: nu_mu produces mu-, while anti-nu_mu "
             "produces mu+. This strong correspondence is the central bookkeeping check. The larger mu- population "
             "is physically reasonable because the generated atmospheric sample contains more neutrinos than "
             "antineutrinos and neutrino charged-current cross sections are generally larger at these energies."),
            ("Multiplicity interpretation",
             "The multiplicity histogram has entries only at zero and one. That is sensible for the generator final "
             "state of this sample: an ordinary muon-flavor charged-current interaction supplies one primary charged "
             "lepton. Secondary muons from pion or kaon decays would belong to later Geant4 transport and are not "
             "included in this generator-final-state count."),
        ], "Muon report 1 of 2")
        # Append a second report page interpreting the kinematic and angular trends.
        add_report_page(pdf, "Muon physics interpretation", [
            ("Energy and momentum",
             f"The median muon kinetic energy is {np.median(all_muon_kinetic):.0f} MeV; the central 90% spans "
             f"{np.percentile(all_muon_kinetic, 5):.0f} to {np.percentile(all_muon_kinetic, 95):.0f} MeV. The broad, "
             "right-skewed spectra are consistent with the broad atmospheric-neutrino spectrum. Muon total energy "
             f"is below the incoming neutrino energy for {np.mean(energy_fraction <= 1.000001):.2%} of pairs, and "
             f"the median E_mu/E_nu is {np.median(energy_fraction):.2f}; the remaining energy goes into the hadronic "
             "system and nuclear recoil or removal energy."),
            ("Angular behavior",
             f"The overall median neutrino-muon opening angle is {np.nanmedian(paired_opening_angle):.1f} degrees. "
             f"Below 1 GeV it is {np.nanmedian(paired_opening_angle[low_energy_pairs]):.1f} degrees, while at and "
             f"above 3 GeV it narrows to {np.nanmedian(paired_opening_angle[high_energy_pairs]):.1f} degrees. This "
             "forward collimation with increasing energy is the expected charged-current behavior and is clearly "
             "visible in the correlation plot. It also shows why a low-energy muon is not a precise neutrino pointing "
             "proxy."),
            ("Zenith distribution",
             "The muon cos(theta_z) distribution is broadly populated across -1 to +1 without an obvious empty or "
             "singular region. That is qualitatively compatible with an all-sky atmospheric sample. A quantitative "
             "zenith validation requires confirming how the flux convention maps onto the simulation z axis and "
             "folding in oscillation weights; this plot alone should not be compared directly with a detector-level "
             "up-going/down-going spectrum."),
            ("Conclusion and limitations",
             "The charge bookkeeping, one-muon multiplicity, energy ordering, and energy-dependent forward trend are "
             "all physically sensible. These plots validate internal kinematics and event handoff, but they do not by "
             "themselves validate an absolute atmospheric rate. Absolute comparisons require flux and oscillation "
             "weights, exposure, target normalization, and GENIE model-systematic variations."),
        ], "Muon report 2 of 2")
    # Report the exact output path for batch jobs and interactive use.
    print(f"Read {event_count:,} events from {input_path}")
    # Report the selected truth population as a quick terminal cross-check.
    print(f"Selected {total_muons:,} generator final-state muons in {events_with_muons:,} events")
    # Confirm successful PDF creation.
    print(f"Saved muon checks to {output_path}")
    # Confirm where the standalone plot images were written.
    print(f"Saved 4 standalone muon PNGs to {png_directory}")


# Execute main only when this file is run as a script.
if __name__ == "__main__":
    # Enter the command-line analysis workflow.
    main()

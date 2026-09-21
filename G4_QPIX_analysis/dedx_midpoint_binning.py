"""Midpoint-based hybrid binning for G4 dE/dx versus residual range.

Each positive-energy Geant4 step is assigned to exactly one bin. Steps whose
midpoints lie below the endpoint transition use fine residual-range bins.
All other steps retain the historical fixed-width bins measured from the
beginning of each individual track. No step is split and no energy is
redistributed between bins.
"""

import numpy as np


def bin_steps_by_midpoint(
    hits,
    group_columns,
    endpoint_bin_width_cm,
    endpoint_max_cm,
    track_bin_width_cm,
):
    """Aggregate G4 steps using fine endpoint and legacy upper-track bins."""
    # A nonempty table is required because track and residual ranges are read
    # directly from the reconstructed Geant4-step rows.
    if hits.empty:
        raise ValueError("No Geant4 hits are available for dE/dx binning")

    # Every width must be positive so floor division produces meaningful bins.
    if endpoint_bin_width_cm <= 0.0:
        raise ValueError("endpoint_bin_width_cm must be positive")
    if endpoint_max_cm <= 0.0:
        raise ValueError("endpoint_max_cm must be positive")
    if track_bin_width_cm <= 0.0:
        raise ValueError("track_bin_width_cm must be positive")

    # Work on a copy so temporary binning columns do not modify the caller's
    # reconstructed hit table or leak into unrelated plotting calculations.
    hits = hits.copy()

    # Classify a complete G4 step from its midpoint residual range. A midpoint
    # strictly below the transition enters the fine endpoint region; a midpoint
    # on the transition belongs to the historical upper-track region.
    endpoint_mask = hits["residual_range_step_cm"] < endpoint_max_cm

    # Convert each endpoint midpoint to its zero-based residual-range bin.
    # For a 0.05 cm width, indices 0 and 1 represent [0, 0.05) and
    # [0.05, 0.10), respectively.
    endpoint_index = np.floor(
        hits.loc[endpoint_mask, "residual_range_step_cm"].to_numpy()
        / endpoint_bin_width_cm
    ).astype(int)

    # Reproduce the original upper-region binning from distance along the
    # individual track, not from residual range. This is the behavior that gave
    # the desired continuous-looking distribution above 1 cm.
    track_index = np.floor(
        hits.loc[~endpoint_mask, "s_mid_cm"].to_numpy() / track_bin_width_cm
    ).astype(int)

    # Store endpoint bins as nonnegative keys. Store historical track bins as
    # -(index + 1), making the first one -1 and preventing key collisions when
    # both regions are aggregated in a single groupby operation.
    hits["hybrid_bin_index"] = 0
    hits.loc[endpoint_mask, "hybrid_bin_index"] = endpoint_index
    hits.loc[~endpoint_mask, "hybrid_bin_index"] = -(track_index + 1)

    # Multiply each step midpoint by its path length now. Summing this numerator
    # and later dividing by total path length gives a path-weighted x coordinate
    # for every plotted track-bin measurement.
    hits["weighted_residual_range_cm"] = (
        hits["residual_range_step_cm"] * hits["ds_cm"]
    )

    # Count source steps explicitly so the CSV reveals how much averaging
    # contributes to each dE/dx point.
    hits["source_step_count"] = 1

    # Aggregate independently for each event, track, particle/truth category,
    # and hybrid bin. Energy and path are summed without splitting any step.
    grouped = (
        hits.groupby(
            [*group_columns, "hybrid_bin_index"], as_index=False, sort=False
        )
        .agg(
            energy_MeV=("E", "sum"),
            path_length_cm=("ds_cm", "sum"),
            weighted_residual_range_cm=("weighted_residual_range_cm", "sum"),
            n_source_steps=("source_step_count", "sum"),
            track_length_cm=("track_length_cm", "first"),
        )
    )

    # Positive path length is required for both the weighted x coordinate and
    # dE/dx denominator. The upstream geometry function already enforces this,
    # but this guard keeps the helper numerically self-contained.
    grouped = grouped[grouped["path_length_cm"] > 0.0].copy()

    # Recover the path-weighted mean residual range of all complete G4 steps
    # assigned to this track-bin. The coordinate varies naturally within the
    # nominal bin according to the complete source-step midpoints.
    grouped["residual_range_cm"] = (
        grouped["weighted_residual_range_cm"] / grouped["path_length_cm"]
    )

    # Calculate the plotted estimator from the unmodified energy and path sums.
    grouped["dEdx_MeV_per_cm"] = (
        grouped["energy_MeV"] / grouped["path_length_cm"]
    )

    # Identify the two algorithms from the sign convention used above.
    grouped_endpoint_mask = grouped["hybrid_bin_index"] >= 0
    grouped["binning_region"] = np.where(
        grouped_endpoint_mask,
        "endpoint_residual_midpoint",
        "legacy_track_start_midpoint",
    )

    # Initialize nominal residual-range boundaries as missing. Exact residual
    # boundaries exist only for endpoint bins; upper-region bins are defined in
    # distance from the track start and therefore have no shared residual edges.
    grouped["range_bin_low_cm"] = np.nan
    grouped["range_bin_high_cm"] = np.nan

    # Recover the shared endpoint-bin boundaries from each nonnegative index.
    endpoint_group_index = grouped.loc[
        grouped_endpoint_mask, "hybrid_bin_index"
    ].to_numpy(dtype=int)
    grouped.loc[grouped_endpoint_mask, "range_bin_low_cm"] = (
        endpoint_group_index * endpoint_bin_width_cm
    )
    grouped.loc[grouped_endpoint_mask, "range_bin_high_cm"] = (
        (endpoint_group_index + 1) * endpoint_bin_width_cm
    )

    # Record the nominal width used by each row. For legacy rows this is the
    # distance-from-track-start width, even though residual boundaries are NaN.
    grouped["range_bin_width_cm"] = np.where(
        grouped_endpoint_mask, endpoint_bin_width_cm, track_bin_width_cm
    )

    # Report the accumulated path relative to nominal width as a diagnostic.
    # Because midpoint assignment keeps whole steps intact, this value can exceed
    # one when long steps cross a nominal boundary; that behavior is intentional.
    grouped["bin_path_fraction"] = (
        grouped["path_length_cm"] / grouped["range_bin_width_cm"]
    )

    # Compare grouped totals with source totals to prove that midpoint grouping
    # neither creates nor removes energy or path length.
    input_energy = float(hits["E"].sum())
    grouped_energy = float(grouped["energy_MeV"].sum())
    input_path = float(hits["ds_cm"].sum())
    grouped_path = float(grouped["path_length_cm"].sum())

    # Scale tolerances to sample size while retaining near machine precision.
    energy_tolerance = max(1e-9, abs(input_energy) * 1e-10)
    path_tolerance = max(1e-9, abs(input_path) * 1e-10)

    # Fail immediately if any source energy was lost or duplicated by grouping.
    if not np.isclose(input_energy, grouped_energy, rtol=1e-10, atol=energy_tolerance):
        raise RuntimeError(
            f"Energy accounting failed: input={input_energy}, grouped={grouped_energy}"
        )

    # Apply the same strict accounting check to reconstructed path length.
    if not np.isclose(input_path, grouped_path, rtol=1e-10, atol=path_tolerance):
        raise RuntimeError(
            f"Path accounting failed: input={input_path}, grouped={grouped_path}"
        )

    # Save every configuration and accounting quantity needed to reproduce or
    # audit the generated plots and detailed track-bin CSV.
    diagnostics = {
        "assignment_method": "whole_step_midpoint",
        "input_step_count": int(len(hits)),
        "output_track_bin_count": int(len(grouped)),
        "endpoint_max_cm": float(endpoint_max_cm),
        "endpoint_bin_width_cm": float(endpoint_bin_width_cm),
        "endpoint_bin_count": int(np.ceil(endpoint_max_cm / endpoint_bin_width_cm)),
        "track_start_bin_width_cm": float(track_bin_width_cm),
        "input_energy_MeV": input_energy,
        "grouped_energy_MeV": grouped_energy,
        "energy_difference_MeV": grouped_energy - input_energy,
        "input_path_cm": input_path,
        "grouped_path_cm": grouped_path,
        "path_difference_cm": grouped_path - input_path,
    }

    # Return both the plotted measurements and their reproducibility record.
    return grouped, diagnostics

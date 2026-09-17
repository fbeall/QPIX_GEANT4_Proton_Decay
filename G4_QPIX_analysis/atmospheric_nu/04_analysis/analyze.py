#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


FLAVORS = [12, -12, 14, -14]
LABELS = {12: r"$\nu_e$", -12: r"$\bar\nu_e$", 14: r"$\nu_\mu$", -14: r"$\bar\nu_\mu$"}


def save(plots: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(plots / name, dpi=170)
    plt.close()


def firsts_cm(array: ak.Array) -> np.ndarray:
    return np.asarray(ak.fill_none(ak.firsts(array), np.nan), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genie", required=True)
    parser.add_argument("--g4", required=True)
    parser.add_argument("--rtd", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}

    with uproot.open(args.genie) as source:
        tree = source["gRooTracker"]
        arrays = tree.arrays(["StdHepPdg", "StdHepStatus", "StdHepP4"], library="ak")
        flavors: list[int] = []
        energies: list[float] = []
        cos_zenith: list[float] = []
        stable_mult: list[int] = []
        for pdg, status, p4 in zip(arrays.StdHepPdg, arrays.StdHepStatus, arrays.StdHepP4):
            pdg_np = np.asarray(pdg)
            status_np = np.asarray(status)
            p4_np = np.asarray(p4)
            initial = (status_np == 0) & np.isin(pdg_np, FLAVORS)
            if not np.any(initial):
                continue
            idx = int(np.flatnonzero(initial)[0])
            flavors.append(int(pdg_np[idx]))
            energies.append(float(p4_np[idx, 3]))
            momentum = float(np.linalg.norm(p4_np[idx, :3]))
            cos_zenith.append(float(p4_np[idx, 2] / momentum) if momentum > 0 else float("nan"))
            stable_mult.append(int(np.sum(status_np == 1)))

        counts = [flavors.count(pdg) for pdg in FLAVORS]
        plt.bar(range(4), counts, color=["#3b6ea8", "#6aa88f", "#c85353", "#d99334"])
        plt.xticks(range(4), [LABELS[pdg] for pdg in FLAVORS])
        plt.ylabel("Interactions")
        save(plots, "genie_flavor.png")

        plt.hist(energies, bins=50, histtype="stepfilled", alpha=0.75)
        plt.xscale("log")
        plt.xlabel("Atmospheric neutrino energy [GeV]")
        plt.ylabel("Events")
        save(plots, "genie_energy.png")

        plt.hist(cos_zenith, bins=20, range=(-1, 1))
        plt.xlabel(r"$\cos\theta_z$")
        plt.ylabel("Events")
        save(plots, "genie_zenith.png")

        max_mult = max(stable_mult) if stable_mult else 1
        plt.hist(stable_mult, bins=np.arange(max_mult + 2) - 0.5)
        plt.xlabel("Stable final-state multiplicity")
        plt.ylabel("Events")
        save(plots, "genie_final_multiplicity.png")

        summary["genie"] = {
            "events": int(tree.num_entries),
            "flavor_counts": dict(zip(map(str, FLAVORS), counts)),
            "energy_GeV": [float(np.nanmin(energies)), float(np.nanmax(energies))] if energies else [],
        }

    with uproot.open(args.g4) as source:
        tree = source["event_tree"]
        arrays = tree.arrays(
            [
                "generator_initial_particle_x",
                "generator_initial_particle_y",
                "generator_initial_particle_z",
                "energy_deposit",
                "particle_pdg_code",
                "hit_energy_deposit",
                "hit_length",
            ],
            library="ak",
            how=dict,
        )
        vx = firsts_cm(arrays["generator_initial_particle_x"])
        vy = firsts_cm(arrays["generator_initial_particle_y"])
        vz = firsts_cm(arrays["generator_initial_particle_z"])
        ed = np.asarray(arrays["energy_deposit"], dtype=float)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for values, limit, axis_name, axis in zip([vx, vy, vz], [230, 600, 360], ["x", "y", "z"], axes):
            axis.hist(values, bins=20, range=(0, limit))
            axis.set(xlabel=f"Vertex {axis_name} [cm]", ylabel="Events")
        save(plots, "geant4_vertex_uniformity.png")

        plt.hist(ed, bins=50)
        plt.xlabel("Total deposited energy [MeV]")
        plt.ylabel("Events")
        save(plots, "geant4_deposited_energy.png")

        flat_edep = ak.to_numpy(ak.flatten(arrays["hit_energy_deposit"], axis=None))
        flat_len = ak.to_numpy(ak.flatten(arrays["hit_length"], axis=None))
        mask = (flat_edep > 0) & (flat_len > 0)
        if np.any(mask):
            dedx = flat_edep[mask] / flat_len[mask]
            plt.hist(dedx, bins=80, range=(0, np.nanpercentile(dedx, 99)))
            plt.xlabel("Geant4 hit dE/dx [MeV/cm]")
            plt.ylabel("Hits")
            save(plots, "geant4_hit_dedx.png")

        pdg_by_event = arrays["particle_pdg_code"]
        summary["geant4"] = {
            "events": int(tree.num_entries),
            "mean_deposit_MeV": float(np.nanmean(ed)) if len(ed) else 0.0,
            "events_with_Kplus": int(ak.sum(ak.any(pdg_by_event == 321, axis=1))),
            "vertices_cm": {
                "x": [float(np.nanmin(vx)), float(np.nanmax(vx))] if len(vx) else [],
                "y": [float(np.nanmin(vy)), float(np.nanmax(vy))] if len(vy) else [],
                "z": [float(np.nanmin(vz)), float(np.nanmax(vz))] if len(vz) else [],
            },
        }

    with uproot.open(args.rtd) as source:
        tree = source["event_tree"]
        resets = tree["pixel_reset"].array(library="ak")
        per_pixel = ak.num(resets, axis=2)
        resets_per_event = np.asarray(ak.sum(per_pixel, axis=1), dtype=float)
        plt.hist(resets_per_event, bins=50)
        plt.xlabel("Q-Pix resets/event")
        plt.ylabel("Events")
        save(plots, "rtd_resets_per_event.png")
        summary["rtd"] = {
            "events": int(tree.num_entries),
            "total_resets": int(np.sum(resets_per_event)),
            "mean_resets": float(np.mean(resets_per_event)) if len(resets_per_event) else 0.0,
        }

    genie_events = int(summary["genie"]["events"])  # type: ignore[index]
    event_counts = [summary[key]["events"] for key in ["genie", "geant4", "rtd"]]  # type: ignore[index]
    checks = {
        "event_counts_match": len(set(event_counts)) == 1,
        "flavor_accounting_ok": sum(counts) == genie_events and (genie_events < 4 or all(counts)),
        "vertices_inside_HD": bool(
            np.all((vx >= 0) & (vx <= 230) & (vy >= 0) & (vy <= 600) & (vz >= 0) & (vz <= 360))
        ),
        "has_deposition": bool(np.any(ed > 0)),
        "has_resets": bool(summary["rtd"]["total_resets"] > 0),  # type: ignore[index]
    }
    summary["checks"] = checks

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not all(checks.values()):
        raise SystemExit("Validation failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download the DUNE-site Honda atmospheric flux table and trim it for GENIE."""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/JIECheng2021/atm_nu_flux_data/main/DUNE/solar-min_without-muon-in-earth/dune-ally-20-12-solmin.d"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(__file__).resolve().parent / "flux"
    out.mkdir(exist_ok=True)
    raw = out / "dune-ally-20-12-solmin.raw.d"
    dst = out / "dune-ally-20-12-solmin-genie.d"

    if args.force or not raw.exists():
        urllib.request.urlretrieve(URL, raw)

    lines = raw.read_text().splitlines()
    result: list[str] = []
    i = 0
    blocks = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        h1, h2 = lines[i], lines[i + 1]
        i += 2
        data: list[tuple[float, str]] = []
        while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith("average flux"):
            parts = lines[i].split()
            if len(parts) >= 5:
                try:
                    data.append((float(parts[0]), lines[i]))
                except ValueError:
                    pass
            i += 1
        keep = [line for energy, line in data if energy >= 0.1 - 1e-12]
        if len(keep) != 101:
            raise SystemExit(f"Block {blocks}: expected 101 energies >=0.1 GeV, got {len(keep)}")
        result.extend([h1, h2, *keep])
        blocks += 1

    if blocks != 240:
        raise SystemExit(f"Expected 240 angular blocks, got {blocks}")

    dst.write_text("\n".join(result) + "\n")
    print(dst)
    print("sha256", hashlib.sha256(dst.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

"""
CLI entry point for running the syPIV pipeline in batch mode (e.g. on HPC).

This mirrors the GUI pipeline but is driven by command-line arguments.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np

from sypivlib.function.dataio import FlowIO, GridIO
from sypivlib.gui.sypiv_worker import SyPIVWorker


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run syPIV batch pipeline.")
    parser.add_argument("--grid", required=True, help="Path to Plot3D grid file (.x)")
    parser.add_argument("--flow", required=True, help="Path to Plot3D flow file (.q)")
    parser.add_argument("--x-min", type=float, required=True, help="IA x-min in physical units")
    parser.add_argument("--x-max", type=float, required=True, help="IA x-max in physical units")
    parser.add_argument("--y-min", type=float, required=True, help="IA y-min in physical units")
    parser.add_argument("--y-max", type=float, required=True, help="IA y-max in physical units")
    parser.add_argument("--snapshots", type=int, default=3, help="Number of snapshot pairs to generate")
    parser.add_argument("--out-dir", required=True, help="Directory to write output arrays (.npy)")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    grid = GridIO(args.grid)
    grid.read_grid()
    flow = FlowIO(args.flow)
    flow.read_flow()

    ia_bounds = [args.x_min, args.x_max, args.y_min, args.y_max]

    particle_params = {
        "n_concentration": 500,
        "min_dia": 144e-9,
        "max_dia": 573e-9,
        "mean_dia": 281e-9,
        "std_dia": 97e-9,
        "density": 810,
        "in_plane": 90.0,
        "distribution": "gaussian",
    }
    laser_params = {
        "position": 0.0009,
        "thickness": 0.0001,
        "pulse_time": 1e-7,
    }
    ccd_params = {
        "xres": 512,
        "yres": 512,
        "dpi": 96,
        "d_ccd": 0.0135,
        "d_ia": 0.0009,
    }
    intensity_params = {
        "sx": 2.0,
        "sy": 2.0,
        "frx": 1.0,
        "fry": 1.0,
        "s": 2,
        "q": 1.0,
    }

    # Minimal wrapper around SyPIVWorker without Qt event loop:
    # instantiate and call run() directly, intercepting emitted snapshots via monkey-patching.
    worker = SyPIVWorker(
        grid=grid,
        flow=flow,
        ia_bounds=ia_bounds,
        particle_params=particle_params,
        laser_params=laser_params,
        ccd_params=ccd_params,
        intensity_params=intensity_params,
        num_snapshots=args.snapshots,
    )

    snapshots: dict[int, dict[str, np.ndarray]] = {}

    def _on_snapshot_complete(snap_num: int, img1: object, img2: object) -> None:
        snapshots[snap_num] = {
            "image1": img1.intensity.values,
            "image2": img2.intensity.values,
        }

    worker.snapshot_complete.connect(_on_snapshot_complete)  # type: ignore[arg-type]

    # Run synchronously
    worker.run()

    # Save all snapshots
    for snap_num, arrs in snapshots.items():
        base = os.path.join(args.out_dir, f"pair{snap_num}")
        np.save(base + "_1.npy", arrs["image1"])
        np.save(base + "_2.npy", arrs["image2"])


if __name__ == "__main__":
    main()


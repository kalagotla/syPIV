"""
Background worker thread for executing the syPIV pipeline.
This prevents the GUI from freezing during computation.
"""

from __future__ import annotations

import traceback
from typing import Optional

import numpy as np
from PyQt6 import QtCore

from sypivlib.function.dataio import FlowIO, GridIO
from sypivlib.sypiv.create_particles import CreateParticles, LaserSheet, Particle
from sypivlib.sypiv.ccd_projection import CCDProjection
from sypivlib.sypiv.intensity import Intensity
from sypivlib.sypiv.image_gen import ImageGen


class SyPIVWorker(QtCore.QThread):
    """
    Worker thread that executes the syPIV pipeline.
    Emits signals for progress updates and completion.
    """

    progress = QtCore.pyqtSignal(str)  # Progress message
    snapshot_complete = QtCore.pyqtSignal(int, object, object)  # snap_num, image1, image2
    finished = QtCore.pyqtSignal(bool, str)  # success, message
    error = QtCore.pyqtSignal(str)  # error message

    def __init__(
        self,
        grid: GridIO,
        flow: FlowIO,
        ia_bounds: list[float],  # [x_min, x_max, y_min, y_max]
        particle_params: dict,
        laser_params: dict,
        ccd_params: dict,
        intensity_params: dict,
        num_snapshots: int = 3,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.grid = grid
        self.flow = flow
        self.ia_bounds = ia_bounds
        self.particle_params = particle_params
        self.laser_params = laser_params
        self.ccd_params = ccd_params
        self.intensity_params = intensity_params
        self.num_snapshots = num_snapshots
        self._cancelled = False

    def cancel(self) -> None:
        """Cancel the computation."""
        self._cancelled = True

    def run(self) -> None:
        """Execute the syPIV pipeline."""
        try:
            # Setup particle
            self.progress.emit("Setting up particles...")
            particle = Particle()
            particle.min_dia = self.particle_params["min_dia"]
            particle.max_dia = self.particle_params["max_dia"]
            particle.mean_dia = self.particle_params["mean_dia"]
            particle.std_dia = self.particle_params["std_dia"]
            particle.density = self.particle_params["density"]
            particle.n_concentration = int(self.particle_params["n_concentration"])
            particle.distribution = self.particle_params.get("distribution", "gaussian")
            particle.compute_distribution()

            # Setup laser sheet
            self.progress.emit("Setting up laser sheet...")
            laser = LaserSheet(self.grid)
            laser.position = self.laser_params["position"]
            laser.thickness = self.laser_params["thickness"]
            laser.pulse_time = self.laser_params["pulse_time"]
            laser.compute_bounds()

            # Generate snapshots
            for snap_num in range(1, self.num_snapshots + 1):
                if self._cancelled:
                    self.finished.emit(False, "Cancelled by user")
                    return

                self.progress.emit(f"Generating snapshot pair {snap_num}/{self.num_snapshots}...")

                # Create particles
                self.progress.emit(f"  Creating particles for snapshot {snap_num}...")
                create_particles = CreateParticles(
                    self.grid, self.flow, particle, laser, self.ia_bounds
                )
                create_particles.ia_bounds = self.ia_bounds
                create_particles.in_plane = self.particle_params.get("in_plane", 90)
                create_particles.compute_locations()

                if self._cancelled:
                    self.finished.emit(False, "Cancelled by user")
                    return

                # Compute second locations
                self.progress.emit(f"  Computing particle trajectories for snapshot {snap_num}...")
                create_particles.compute_locations2_serial()

                if self._cancelled:
                    self.finished.emit(False, "Cancelled by user")
                    return

                # Project to CCD
                self.progress.emit(f"  Projecting particles to CCD for snapshot {snap_num}...")
                projection = CCDProjection(create_particles)
                projection.xres = self.ccd_params["xres"]
                projection.yres = self.ccd_params["yres"]
                projection.dpi = self.ccd_params["dpi"]
                projection.d_ccd = self.ccd_params["d_ccd"]
                projection.d_ia = self.ccd_params["d_ia"]
                projection.compute()

                if self._cancelled:
                    self.finished.emit(False, "Cancelled by user")
                    return

                # Compute intensity for first image
                self.progress.emit(f"  Computing intensity field (image 1) for snapshot {snap_num}...")
                cache1 = (
                    projection.projections[:, 2],  # dia_x
                    projection.projections[:, 2],  # dia_y
                    projection.projections[:, 0],  # xp
                    projection.projections[:, 1],  # yp
                    self.intensity_params["sx"],
                    self.intensity_params["sy"],
                    self.intensity_params["frx"],
                    self.intensity_params["fry"],
                    self.intensity_params["s"],
                    self.intensity_params["q"],
                    create_particles.locations[:, 2],  # z_physical
                )
                intensity1 = Intensity(cache1, projection)
                intensity1.compute_serial()  # Use serial for GUI responsiveness

                if self._cancelled:
                    self.finished.emit(False, "Cancelled by user")
                    return

                # Generate first image
                image_gen1 = ImageGen(intensity1)
                image_gen1.snap(snap_num=snap_num * 2 - 1)

                if self._cancelled:
                    self.finished.emit(False, "Cancelled by user")
                    return

                # Compute intensity for second image
                self.progress.emit(f"  Computing intensity field (image 2) for snapshot {snap_num}...")
                cache2 = (
                    projection.projections2[:, 2],  # dia_x
                    projection.projections2[:, 2],  # dia_y
                    projection.projections2[:, 0],  # xp
                    projection.projections2[:, 1],  # yp
                    self.intensity_params["sx"],
                    self.intensity_params["sy"],
                    self.intensity_params["frx"],
                    self.intensity_params["fry"],
                    self.intensity_params["s"],
                    self.intensity_params["q"],
                    create_particles.locations2[:, 2],  # z_physical
                )
                intensity2 = Intensity(cache2, projection)
                intensity2.compute_serial()

                if self._cancelled:
                    self.finished.emit(False, "Cancelled by user")
                    return

                # Generate second image
                image_gen2 = ImageGen(intensity2)
                image_gen2.snap(snap_num=snap_num * 2)

                # Emit completion signal
                self.snapshot_complete.emit(snap_num, image_gen1, image_gen2)

            self.finished.emit(True, f"Successfully generated {self.num_snapshots} snapshot pairs")

        except Exception as e:
            error_msg = f"Error in syPIV pipeline:\n{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)
            self.finished.emit(False, error_msg)

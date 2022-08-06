# Creates 3d IA from grid and flow data
# Spawns particles on a distribution and returns their locations
import numpy as np
rng = np.random.default_rng(7)


class Particle:
    """
    Class holds details for particles used in PIV experiment
    ---
    User has to provide all the information
    """

    def __init__(self):
        self.distribution = "gaussian"
        self.min_dia = None
        self.max_dia = None
        self.mean_dia = None
        self.std_dia = None
        self.density = None
        self.n_concentration = None
        self.particle_field = None

    def compute_distribution(self):
        """
        Run this method to return a distribution of particle diameters
        :return: numpy.ndarray
        A 1d array of particle diameters
        """
        if self.distribution == "gaussian":
            print("When Gaussian distribution is used,"
                  " the particle statistics are computed using mean and std diameters")
            self.particle_field = rng.normal(self.mean_dia, self.std_dia, int(self.n_concentration))
            self.particle_field = np.clip(self.particle_field, self.min_dia, self.max_dia)
            return

        # TODO: Add Uniform distribution

    pass


class LaserSheet:
    """
    Laser sheet information to pass into CreateParticles
    ---
    User input class
    """

    def __init__(self, grid):
        self.grid = grid
        self.distribution = "gaussian"
        self.position = None
        self.thickness = None
        self.width = None

        print("The laser sheet will reside inside the grid. "
              "Carefully assign position and thickness using parameters below")
        print(f"position and thickness should be between {self.grid.grd_min[:, 2]} and {self.grid.grd_max[:, 2]}")

    def compute_bounds(self):
        self.width = np.array([self.position - self.thickness/2, self.position + self.thickness/2])

    pass


class CreateParticles:
    """
    Module to create 3d Interrogation Area (IA) from grid and flow data
    AND
    Spawns particles based on the given distribution
    """

    def __init__(self, grid, flow, particle, laser_sheet, ia_bounds: list[float, float, float, float]):
        self.grid = grid
        self.flow = flow
        self.particle = particle
        self.laser_sheet = laser_sheet
        # x_min, x_max, y_min, y_max --> ia_bounds
        self.ia_bounds = ia_bounds
        # percent of particles in-plane; rest will be divided equally above and below the ia_plane
        self.in_plane = None
        # locations is an n x 4 array; [x, y, z, diameter]
        self.locations = None
        print(f"ia_bounds should be with in:\n"
              f"In x-direction: {self.grid.grd_min[:, 0]} and {self.grid.grd_max[:, 0]}\n"
              f"In y-direction: {self.grid.grd_min[:, 1]} and {self.grid.grd_max[:, 1]}\n")

    def compute_locations(self):
        # Uniform distribution
        # In-plane points
        _particles_in_plane = int(self.in_plane * self.particle.n_concentration * 0.01)
        _x_loc = rng.uniform(self.ia_bounds[0], self.ia_bounds[1], _particles_in_plane)
        _y_loc = rng.uniform(self.ia_bounds[2], self.ia_bounds[3], _particles_in_plane)
        _z_loc = np.repeat(self.laser_sheet.position, _particles_in_plane)
        self.locations = np.vstack((_x_loc, _y_loc, _z_loc, self.particle.particle_field[:_particles_in_plane])).T

        # Off-plane locations - randomize z
        _particles_off_plane = int(self.particle.n_concentration - _particles_in_plane)
        _x_loc = rng.uniform(self.ia_bounds[0], self.ia_bounds[1], _particles_off_plane)
        _y_loc = rng.uniform(self.ia_bounds[2], self.ia_bounds[3], _particles_off_plane)
        _z_loc = rng.uniform(self.laser_sheet.position - self.laser_sheet.width[0],
                             self.laser_sheet.position + self.laser_sheet.width[1], _particles_off_plane)
        self.locations = np.concatenate((self.locations,
                                        np.vstack((_x_loc, _y_loc, _z_loc,
                                                   self.particle.particle_field[_particles_in_plane:])).T), axis=0)

        return

    pass

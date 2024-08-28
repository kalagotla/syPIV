# Creates 3d IA from grid and flow data
# Spawns particles on a distribution and returns their locations
import numpy as np
from multiprocessing import cpu_count, Pool
from ..function.variables import Variables
from ..function.search import Search
from ..function.interpolation import Interpolation
from ..function.integration import Integration
import tqdm
from mpi4py import MPI
rng = np.random.default_rng()


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
                  " the particle statistics are computed using mean and std diameters\n"
                  "Particle min and max are cutoffs for the distribution")
            self.particle_field = rng.normal(self.mean_dia, self.std_dia, int(self.n_concentration))
            self.particle_field = np.clip(self.particle_field, self.min_dia, self.max_dia)
            rng.shuffle(self.particle_field)
            return

        # TODO: Add Uniform distribution

    pass


# TODO: Add laser pulse for generating second snap
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
        self.pulse_time = None

        print("The laser sheet will reside inside the grid. "
              "Carefully assign position and thickness using parameters below")
        print(f"position and thickness should be between {self.grid.grd_min[:, 2]} and {self.grid.grd_max[:, 2]}")

    def compute_bounds(self):
        self.width = np.array([self.position - self.thickness / 2, self.position + self.thickness / 2])

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
        self.locations2 = []
        self._failed_ids = []
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
        _z_loc = rng.uniform(self.laser_sheet.width[0], self.laser_sheet.width[1], _particles_off_plane)
        self.locations = np.concatenate((self.locations,
                                         np.vstack((_x_loc, _y_loc, _z_loc,
                                                    self.particle.particle_field[_particles_in_plane:])).T), axis=0)

        return

    def _process(self, _location, _task_id):
        try:
            _x, _y, _z, _d = _location
            _idx = Search(self.grid, [_x, _y, _z])
            _idx.compute(method='p-space')

            _interp = Interpolation(self.flow, _idx)
            _interp.compute(method='p-space')

            _var = Variables(_interp)
            _var.compute_velocity()

            _intg = Integration(_interp)
            _new_loc, _ = _intg.compute(method='pRK4', time_step=self.laser_sheet.pulse_time)
            if _new_loc is None:
                # make sure the point moves out of the grid for second image
                _new_loc = np.array((_x, _y, _z)) + self.laser_sheet.pulse_time * _var.velocity.reshape(3)
            return np.hstack((_new_loc, _d))
        except:
            # delete the particle from self.locations
            self._failed_ids.append(_task_id)
            print(f"***Error in task {_task_id}***")
            return None

    def compute_locations2(self):
        """
        Will integrate particles to new locations based on
        Laser pulse time and velocities at their locations
        :return:
        """

        # setup parameters for multiprocessing
        _tasks = np.arange(len(self.locations))
        n = max(1, cpu_count() - 1)
        pool = Pool(n)
        self.locations2 = pool.starmap(self._process, zip(self.locations, _tasks))
        pool.close()
        pool.join()

        # delete failed tasks
        self.locations = np.delete(self.locations, self._failed_ids, axis=0)

        self.locations2 = np.array(self.locations2)
        self.locations2 = np.delete(self.locations2, self._failed_ids, axis=0)
        print(f"Total number of particles as per locations: {len(self.locations)}")
        print(f"Total number of particles as per locations2: {len(self.locations2)}")
        print(f"Failed number of particles: {len(self._failed_ids)}")

        return

    def compute_locations2_serial(self):
        """
        Will integrate particles to new locations based on
        Laser pulse time and velocities at their locations
        :return:
        """

        # for loop for serial computation. Track using tqdm computing locations...
        for _i, _j in enumerate(tqdm.tqdm(self.locations, desc="Computing locations for second image")):
            self.locations2.append(self._process(_j, _i))

        # delete failed tasks
        # self.locations = np.delete(self.locations, self._failed_ids, axis=0)

        self.locations2 = np.array(self.locations2)
        self.locations2 = np.delete(self.locations2, self._failed_ids, axis=0)
        # convert to float64
        self.locations2 = np.array(list(self.locations2), dtype=np.float64)
        print(f"Total number of particles as per locations: {len(self.locations)}")
        print(f"Total number of particles as per locations2: {len(self.locations2)}")
        print(f"Failed number of particles: {len(self._failed_ids)}")

        return

    def compute_locations2_mpi(self):
        """
        Will integrate particles to new locations based on
        Laser pulse time and velocities at their locations
        will work with mpi4py
        :return:
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        self.locations = np.array_split(self.locations, size)[rank]
        # for loop for serial computation. Track using tqdm computing locations...
        for _i, _j in enumerate(tqdm.tqdm(self.locations, desc="Computing locations for second image")):
            self.locations2.append(self._process(_j, _i))

        # delete failed tasks on each rank
        # self.locations = np.delete(self.locations, self._failed_ids, axis=0)
        self.locations2 = np.array(self.locations2)
        self.locations2 = np.delete(self.locations2, self._failed_ids, axis=0)
        # convert to float64
        self.locations2 = np.array(list(self.locations2), dtype=np.float64)

        # gather all the locations
        self.locations = comm.gather(self.locations, root=0)
        self.locations2 = comm.gather(self.locations2, root=0)
        if rank == 0:
            self.locations = np.concatenate(self.locations, axis=0)
            self.locations2 = np.concatenate(self.locations2, axis=0)
            print(f"Total number of particles as per locations: {len(self.locations)}")
            print(f"Total number of particles as per locations2: {len(self.locations2)}")
            print(f"Failed number of particles: {len(self._failed_ids)}")

        self.locations = comm.bcast(self.locations, root=0)
        self.locations2 = comm.bcast(self.locations2, root=0)

        return

    pass

# Generate intensity field at particle particles
from scipy.special import erf
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import dask.array as da
import tqdm
from mpi4py import MPI


# import sys
# resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
# sys.setrecursionlimit(10**6)


class Intensity:
    """
    Parameters
    ----------
    cache : tuple
            (radiusx, radiusy, xp, yp, sx, sy, frx, fry)
            Has all the variables needed for compute function.
            Created to clean up the code
    projection : object from CCDProjection class
            Contains particle location data in pixels
            Contains xres, yres, and dpi data

    Returns
    -------
    intensity: function
            Filled with data from cache; a function of particle locations
    values: numpy.ndarray
            Intensity values for the final image

    By: Dilip Kalagotla ~ kal @ dilip.kalagotla@gmail.com
    Date created: Sat August 6, 2022
    """

    def __init__(self, cache, projection):
        self.cache = cache
        self.projection = projection
        self.intensity = None
        self.values = None

    def setup(self, x, y, ls_thickness, ls_position, dia_x, dia_y, xp, yp, sx, sy, frx, fry, s, q, z_physical, _task):
        """
        cache = (radiusx, radiusy, xp, yp, sx, sy, frx, fry)
        I = intensityField(cache)

        Parameters
        ----------

        Returns
        -------
        intensity : function
            Intensity field as a function of particle particles.

        By: Dilip Kalagotla ~ kal @ dilip.kalagotla@gmail.com
        Date created: Mon May 17 11:00:56 2021

        """
        # start time
        # start = time.perf_counter()

        # compute intensity for each particle location based on Gaussian distribution
        # q is the efficiency factor with which particles scatter light
        # s is the shape factor; 2 --> Gaussian, 10^4 --> uniform

        self.intensity = (q *
                          np.exp(-1 / np.sqrt(2 * np.pi) *
                                 abs(2 * (z_physical - ls_position) ** 2 / ls_thickness ** 2) ** s) *
                          (np.pi / 8 * dia_x * dia_y * sx * sy *
                           (erf((x - xp + 0.5 * frx) / (sx * 2 ** 0.5)) -
                            erf((x - xp - 0.5 * frx) / (sx * 2 ** 0.5))) *
                           (erf((y - yp + 0.5 * fry) / (sy * 2 ** 0.5)) -
                            erf((y - yp - 0.5 * fry) / (sy * 2 ** 0.5)))))
        # end time
        # end = time.perf_counter()

        # print(f'Done computing intensity field for {_task}/{len(self.cache[0])} particles in {end - start} seconds.'
        #       f'process number is {os.getpid()}')

        return self.intensity

    def compute(self, chunksize=5096):
        # Using multiprocessing to compute relative intensity field
        (dia_x, dia_y, xp, yp, sx, sy, frx, fry, s, q, z_physical) = self.cache
        intensity = np.zeros((self.projection.yres, self.projection.xres))
        # create meshgrid for x and y
        x = np.linspace(-self.projection.xres / 2, self.projection.xres / 2, self.projection.xres)
        y = np.linspace(-self.projection.yres / 2, self.projection.yres / 2, self.projection.yres)
        x, y = np.meshgrid(x, y)

        # laser sheet thickness
        ls_thickness = self.projection.particles.laser_sheet.thickness
        ls_position = self.projection.particles.laser_sheet.position

        i = 0
        j = chunksize
        pbar = tqdm.tqdm(total=len(xp), desc="Computing intensity field for particles", position=0, leave=True,
                         colour='green')
        while j <= len(xp):
            n = max(1, cpu_count() - 1)
            pool = Pool(n)
            n_particles = len(dia_x[i:j])
            # x, y, ls_thickness, ls_position,
            # dia_x[i], dia_y[i], xp[i], yp[i], sx, sy, frx, fry, s, q, z_physical[i], i
            itemp = pool.starmap(self.setup, zip([x]*n_particles, [y]*n_particles,
                                                 np.repeat(ls_thickness, n_particles),
                                                 np.repeat(ls_position, n_particles),
                                                 dia_x[i:j], dia_y[i:j], xp[i:j], yp[i:j],
                                                 np.repeat(sx, n_particles), np.repeat(sy, n_particles),
                                                 np.repeat(frx, n_particles), np.repeat(fry, n_particles),
                                                 np.repeat(s, n_particles), np.repeat(q, n_particles), z_physical[i:j],
                                                 np.arange(n_particles)), chunksize=n_particles // n)
            pool.close()
            pool.join()

            intensity += np.sum(itemp, axis=0)
            i = j
            j += chunksize
            # print(f"Done with {i} particles out of {len(xp)}")
            pbar.update(chunksize)

        # compute the remaining particles
        n = max(1, cpu_count() - 1)
        pool = Pool(n)
        n_particles = len(dia_x[i:])
        itemp = pool.starmap(self.setup, zip([x]*n_particles, [y]*n_particles,
                                             np.repeat(ls_thickness, n_particles),
                                             np.repeat(ls_position, n_particles),
                                             dia_x[i:], dia_y[i:], xp[i:], yp[i:],
                                             np.repeat(sx, n_particles), np.repeat(sy, n_particles),
                                             np.repeat(frx, n_particles), np.repeat(fry, n_particles),
                                             np.repeat(s, n_particles), np.repeat(q, n_particles), z_physical[i:],
                                             np.arange(n_particles)))
        pool.close()
        pool.join()
        # print(f"Done with {len(xp)} particles out of {len(xp)}")
        pbar.update(len(xp) - i)
        pbar.close()

        intensity += np.sum(itemp, axis=0)

        # Average intensity field
        intensity = intensity / len(xp)

        # Normalize intensity field to rbg values
        if np.max(intensity) != 0:
            intensity = intensity / np.max(intensity) * 255
        print('Done computing intensity field')

        self.values = intensity

        return self.values

    def compute_serial(self):
        (dia_x, dia_y, xp, yp, sx, sy, frx, fry, s, q, z_physical) = self.cache
        intensity = np.zeros((self.projection.yres, self.projection.xres))
        x = np.linspace(-self.projection.xres / 2, self.projection.xres / 2, self.projection.xres)
        y = np.linspace(-self.projection.yres / 2, self.projection.yres / 2, self.projection.yres)
        x, y = np.meshgrid(x, y)

        # laser sheet thickness
        ls_thickness = self.projection.particles.laser_sheet.thickness
        ls_position = self.projection.particles.laser_sheet.position

        for i in tqdm.tqdm(range(len(xp)), desc="Computing intensity field for particles"):
            intensity += self.setup(x, y, ls_thickness, ls_position,
                                    dia_x[i], dia_y[i], xp[i], yp[i], sx, sy, frx, fry, s, q, z_physical[i], i)

        # Average intensity field
        intensity = intensity / len(xp)

        # Normalize intensity field to rbg values
        if np.max(intensity) != 0:
            intensity = intensity / np.max(intensity) * 255
        print('Done computing intensity field')

        self.values = intensity

        return self.values

    def compute_dask(self):
        """
        Ideal for computing intensity field for higher resolution images
        :return:
        """
        (dia_x, dia_y, xp, yp, sx, sy, frx, fry, s, q, z_physical) = self.cache

        # convert arrays to dask arrays
        dia_x = da.from_array(dia_x, chunks=1000)
        dia_y = da.from_array(dia_y, chunks=1000)
        xp = da.from_array(xp, chunks=1000)
        yp = da.from_array(yp, chunks=1000)
        z_physical = da.from_array(z_physical, chunks=1000)
        intensity = np.zeros((self.projection.yres, self.projection.xres))
        # holder for intensity field -- saves time during addition
        da_intensity = da.zeros((self.projection.yres, self.projection.xres), chunks=(1000, 1000))

        # create meshgrid for x and y
        x = da.linspace(-self.projection.xres / 2, self.projection.xres / 2, self.projection.xres)
        y = da.linspace(-self.projection.yres / 2, self.projection.yres / 2, self.projection.yres)
        x, y = da.meshgrid(x, y)

        # laser sheet thickness
        ls_thickness = self.projection.particles.laser_sheet.thickness
        ls_position = self.projection.particles.laser_sheet.position

        # compute intensity for each particle location based on Gaussian distribution
        # q is the efficiency factor with which particles scatter light
        # s is the shape factor; 2 --> Gaussian, 10^4 --> uniform

        # 200 particles at a time to avoid recursion limit
        for j in tqdm.tqdm(range(0, len(xp), 200), desc=f"Computing intensity field for particles in chunks of 200",
                           position=0, leave=True, colour='green'):
            for i in tqdm.tqdm(range(j, j + 200 if j + 200 < len(xp) else len(xp)),
                               desc=f"Setting up intensity field for particles {j} to"
                                    f" {j + 200 if j + 200 < len(xp) else len(xp)}",
                               position=1, leave=False, colour='blue'):
                da_intensity = (da_intensity +
                                (q *
                                 da.exp(-1 / da.sqrt(2 * np.pi) *
                                        abs(2 * (z_physical[i] - ls_position) ** 2 / ls_thickness ** 2) ** s) *
                                 (np.pi / 8 * dia_x[i] * dia_y[i] * sx * sy *
                                  (erf((x - xp[i] + 0.5 * frx) / (sx * 2 ** 0.5)) -
                                   erf((x - xp[i] - 0.5 * frx) / (sx * 2 ** 0.5))) *
                                  (erf((y - yp[i] + 0.5 * fry) / (sy * 2 ** 0.5)) -
                                   erf((y - yp[i] - 0.5 * fry) / (sy * 2 ** 0.5)))))
                                )
            da_intensity = da_intensity.compute()
            intensity = intensity + da_intensity
            da_intensity = da.zeros((self.projection.yres, self.projection.xres), chunks=(1000, 1000))

        # Average intensity field
        intensity = intensity / len(xp)

        # Normalize intensity field to rbg values
        if np.max(intensity) != 0:
            intensity = intensity / np.max(intensity) * 255
        print('Done computing intensity field')

        self.values = intensity

        return self.values

    def compute_mpi(self):
        """
        Ideal for computing intensity field for higher resolution images
        uses MPI for parallel computing
        :return:
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        # set up the data
        (dia_x, dia_y, xp, yp, sx, sy, frx, fry, s, q, z_physical) = self.cache
        intensity = np.zeros((self.projection.yres, self.projection.xres))
        x = np.linspace(-self.projection.xres / 2, self.projection.xres / 2, self.projection.xres)
        y = np.linspace(-self.projection.yres / 2, self.projection.yres / 2, self.projection.yres)
        x, y = np.meshgrid(x, y)

        # laser sheet thickness
        ls_thickness = self.projection.particles.laser_sheet.thickness
        ls_position = self.projection.particles.laser_sheet.position

        # store length of xp
        _len = len(xp)

        # split the data
        xp = np.array_split(xp, size)[rank]
        yp = np.array_split(yp, size)[rank]
        dia_x = np.array_split(dia_x, size)[rank]
        dia_y = np.array_split(dia_y, size)[rank]
        z_physical = np.array_split(z_physical, size)[rank]

        for i in tqdm.tqdm(range(len(xp)), desc="Computing intensity field for particles on process " + str(rank)):
            intensity += self.setup(x, y, ls_thickness, ls_position,
                                    dia_x[i], dia_y[i], xp[i], yp[i],
                                    sx, sy, frx, fry, s, q, z_physical[i], i)

        # wait for all processes to finish
        comm.Barrier()
        # sum up all the intensity fields from all the processes
        if rank == 0:
            for i in range(1, size):
                intensity += comm.recv(source=i)

            # Average intensity field -- use _len because xp is split
            intensity = intensity / _len

            # Normalize intensity field to rbg values
            if np.max(intensity) != 0:
                intensity = intensity / np.max(intensity) * 255
            print('Done computing intensity field')

            self.values = intensity
        else:
            comm.send(intensity, dest=0)

        self.values = comm.bcast(self.values, root=0)

        return self.values


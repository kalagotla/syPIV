# Generate intensity field at particle particles
from scipy.special import erf
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import time
import os
import dask.array as da
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

    def setup(self, dia_x, dia_y, xp, yp, sx, sy, frx, fry, s, q, z_physical, _task):
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
        start = time.perf_counter()
        # create meshgrid for x and y
        x = np.linspace(-self.projection.xres/2, self.projection.xres/2, self.projection.xres)
        y = np.linspace(-self.projection.yres/2, self.projection.yres/2, self.projection.yres)
        x, y = np.meshgrid(x, y)

        # laser sheet thickness
        ls_thickness = self.projection.particles.laser_sheet.thickness
        ls_position = self.projection.particles.laser_sheet.position

        # compute intensity for each particle location based on Gaussian distribution
        # q is the efficiency factor with which particles scatter light
        # s is the shape factor; 2 --> Gaussian, 10^4 --> uniform

        self.intensity = (q *
                          np.exp(-1/np.sqrt(2*np.pi) *
                                 abs(2*(z_physical-ls_position)**2/ls_thickness**2)**s) *
                          (np.pi / 8 * dia_x * dia_y * sx * sy *
                              (erf((x - xp + 0.5 * frx) / (sx * 2 ** 0.5)) -
                               erf((x - xp - 0.5 * frx) / (sx * 2 ** 0.5))) *
                              (erf((y - yp + 0.5 * fry) / (sy * 2 ** 0.5)) -
                               erf((y - yp - 0.5 * fry) / (sy * 2 ** 0.5)))))
        # end time
        end = time.perf_counter()

        print(f'Done computing intensity field for {_task}/{len(self.cache[0])} particles in {end - start} seconds.'
              f'process number is {os.getpid()}')

        return self.intensity

    def compute(self, chunksize=5096):
        # Using multiprocessing to compute relative intensity field
        (dia_x, dia_y, xp, yp, sx, sy, frx, fry, s, q, z_physical) = self.cache
        intensity = np.zeros((self.projection.yres, self.projection.xres))

        i = 0
        j = chunksize
        while j <= len(xp):
            n = max(1, cpu_count() - 1)
            pool = Pool(n)
            n_particles = len(dia_x[i:j])
            itemp = pool.starmap(self.setup, zip(dia_x[i:j], dia_y[i:j], xp[i:j], yp[i:j],
                                             np.repeat(sx, n_particles), np.repeat(sy, n_particles),
                                             np.repeat(frx, n_particles), np.repeat(fry, n_particles),
                                             np.repeat(s, n_particles), np.repeat(q, n_particles), z_physical[i:j],
                                             np.arange(n_particles)), chunksize=n_particles//n)
            pool.close()
            pool.join()

            intensity += np.sum(itemp, axis=0)
            i = j
            j += chunksize
            print(f"Done with {i} particles out of {len(xp)}")

        n = max(1, cpu_count() - 1)
        pool = Pool(n)
        n_particles = len(dia_x[i:])
        itemp = pool.starmap(self.setup, zip(dia_x[i:], dia_y[i:], xp[i:], yp[i:],
                                             np.repeat(sx, n_particles), np.repeat(sy, n_particles),
                                             np.repeat(frx, n_particles), np.repeat(fry, n_particles),
                                             np.repeat(s, n_particles), np.repeat(q, n_particles), z_physical[i:],
                                             np.arange(n_particles)))
        pool.close()
        pool.join()
        print(f"Done with {len(xp)} particles out of {len(xp)}")

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

        for i in range(len(xp)):
            intensity += self.setup(dia_x[i], dia_y[i], xp[i], yp[i], sx, sy, frx, fry, s, q, z_physical[i], i)

        # Average intensity field
        intensity = intensity / len(xp)

        # Normalize intensity field to rbg values
        if np.max(intensity) != 0:
            intensity = intensity / np.max(intensity) * 255
        print('Done computing intensity field')

        self.values = intensity

        return self.values

    def compute_dask(self):
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
        for j in range(0, len(xp), 200):
            start = time.perf_counter()
            for i in range(j, j + 200 if j + 200 < len(xp) else len(xp)):
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
            end = time.perf_counter()
            print(f"Done with {i} particles out of {len(xp)}. Time taken: {end - start} seconds.")

        # Average intensity field
        intensity = intensity / len(xp)

        # Normalize intensity field to rbg values
        if np.max(intensity) != 0:
            intensity = intensity / np.max(intensity) * 255
        print('Done computing intensity field')

        self.values = intensity

        return self.values

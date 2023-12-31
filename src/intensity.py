# Generate intensity field at particle particles
from scipy.special import erf
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count


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

        print(f'Done computing intensity field for {_task}/{len(self.cache[0])} particles')

        return self.intensity

    def compute(self):
        # Using multiprocessing to compute relative intensity field
        (dia_x, dia_y, xp, yp, sx, sy, frx, fry, s, q, z_physical) = self.cache
        intensity = np.zeros((self.projection.yres, self.projection.xres))

        n = max(1, cpu_count() - 1)
        pool = Pool(n)
        n_particles = len(dia_x)
        chunksize = int(n_particles / n)
        itemp = pool.starmap(self.setup, zip(dia_x, dia_y, xp, yp,
                                             np.repeat(sx, n_particles), np.repeat(sy, n_particles),
                                             np.repeat(frx, n_particles), np.repeat(fry, n_particles),
                                             np.repeat(s, n_particles), np.repeat(q, n_particles), z_physical,
                                             np.arange(n_particles)),
                             chunksize=chunksize)
        pool.close()
        pool.join()

        intensity += np.sum(itemp, axis=0)

        # Average intensity field
        intensity = intensity / n_particles

        # Normalize intensity field to rbg values
        if np.max(intensity) != 0:
            intensity = intensity / np.max(intensity) * 255
        print('Done computing intensity field')

        self.values = intensity

        return self.values


# Generate intensity field at particle particles
from scipy.special import erf
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
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

    def setup(self):
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

        (dia_x, dia_y, xp, yp, sx, sy, frx, fry) = self.cache
        x = np.linspace(-self.projection.xres/2, self.projection.xres/2, self.projection.xres)
        y = np.linspace(-self.projection.yres/2, self.projection.yres/2, self.projection.yres)
        x, y = np.meshgrid(x, y)

        # TODO: Need to multiply with max. intensity

        self.intensity = lambda xp, yp, dia_x, dia_y: (np.pi / 8 * dia_x * dia_y * sx * sy *
                                                       (erf((x - xp + 0.5 * frx) / (sx * 2 ** 0.5)) -
                                                        erf((x - xp - 0.5 * frx) / (sx * 2 ** 0.5))) *
                                                       (erf((y - yp + 0.5 * fry) / (sy * 2 ** 0.5)) -
                                                        erf((y - yp - 0.5 * fry) / (sy * 2 ** 0.5))))

        return self.intensity

    @staticmethod
    def multi_process(self, function, xp, yp, dia_x, dia_y, chunksize):
        # Using multiprocessing to compute relative intensity field
        intensity = np.zeros((self.projection.yres, self.projection.xres))
        i = 0
        j = chunksize
        while j <= len(xp):
            n = max(1, cpu_count() - 1)
            pool = ThreadPool(n)
            Itemp = pool.starmap(function, zip(xp[i:j], yp[i:j], dia_x[i:j], dia_y[i:j]))
            pool.close()
            pool.join()

            intensity += np.sum(Itemp, axis=0)
            i = j
            j += chunksize
            print(f"Done with {i} particles out of {len(xp)}")

        n = max(1, cpu_count() - 1)
        pool = ThreadPool(n)
        itemp = pool.starmap(function, zip(xp[i:], yp[i:], dia_x[i:], dia_y[i:]))
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

        return intensity

    def compute(self, chunksize=512):
        print('Computing intensity field...')
        (dia_x, dia_y, xp, yp, sx, sy, frx, fry) = self.cache
        self.values = self.multi_process(self, self.intensity, xp, yp, dia_x, dia_y, chunksize)

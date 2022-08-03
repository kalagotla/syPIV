import numpy as np


class CCDProjection:
    """
    Use to project particles obtained from create_particles onto
    CCD of a camera system
    """

    def __init__(self, particles):
        self.particles = particles
        self.d_ccd = None
        self.d_ia = None
        self.projections = np.empty((self.particles.locations.shape[0], 3))

    def compute(self):
        """

        :return:
        """
        loc = self.particles.locations
        self.projections[:, 0] = loc[:, 0] * self.d_ia / (loc[:, 2] - self.d_ccd)
        self.projections[:, 1] = loc[:, 1] * self.d_ia / (loc[:, 2] - self.d_ccd)
        self.projections[:, 2] = loc[:, 3]
        return

    pass

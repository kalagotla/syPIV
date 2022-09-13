import numpy as np


class CCDProjection:
    """
    Use to project particles obtained from create_particles onto
    CCD of a camera system
    """

    def __init__(self, particles, xres=512, yres=512, dpi=96):
        self.particles = particles
        self.d_ccd = None
        self.d_ia = None
        self.xres = xres
        self.yres = yres
        self.dpi = dpi
        self.projections = np.empty((self.particles.locations.shape[0], 3))
        self.projections2 = np.empty((self.particles.locations2.shape[0], 3))

    def compute(self):
        """
        Computes the particles projection onto a CCD
        Set the sensor size using xres, yres, and dpi
        Appropriate distances for projection should be inputted as well (d_ccd & d_ia)
        :return:
        projections: np.ndarray of shape (nx3)
        projections2: np.ndarray of shape (nx3)
        (x,y,d) --> locations of particles in pixels and their respective diameters in meters
        """
        loc = self.particles.locations
        origin = [np.mean(self.particles.ia_bounds[:2]), np.mean(self.particles.ia_bounds[2:]), loc[0, 2]]
        # Adjust locations to origin before projection
        loc_adj = loc[:, :3] - origin
        self.projections[:, 0] = loc_adj[:, 0] * self.d_ccd / (loc_adj[:, 2] - self.d_ia)
        self.projections[:, 1] = loc_adj[:, 1] * self.d_ccd / (loc_adj[:, 2] - self.d_ia)
        self.projections[:, 2] = loc[:, 3]

        # Convert to pixels for further processing
        self.projections[:, :2] = self.projections[:, :2] * self.dpi / 25.4e-3

        # Compute projections for the second snap
        loc = self.particles.locations2
        origin = [np.mean(self.particles.ia_bounds[:2]), np.mean(self.particles.ia_bounds[2:]), loc[0, 2]]
        # Adjust locations to origin before projection
        loc_adj = loc[:, :3] - origin
        self.projections2[:, 0] = loc_adj[:, 0] * self.d_ccd / (loc_adj[:, 2] - self.d_ia)
        self.projections2[:, 1] = loc_adj[:, 1] * self.d_ccd / (loc_adj[:, 2] - self.d_ia)
        self.projections2[:, 2] = loc[:, 3]

        # Convert to pixels for further processing
        self.projections2[:, :2] = self.projections2[:, :2] * self.dpi / 25.4e-3

        return

    pass

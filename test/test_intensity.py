import unittest
import matplotlib.pyplot as plt
import numpy as np


class TestIntensity(unittest.TestCase):
    def test_intensity(self):
        from src.sypivlib.function.dataio import GridIO, FlowIO
        from src.sypivlib.sypiv.create_particles import Particle, LaserSheet, CreateParticles
        from src.sypivlib.sypiv.ccd_projection import CCDProjection
        from src.sypivlib.sypiv.intensity import Intensity

        # Read-in the grid and flow file
        grid = GridIO('../data/plate_data/plate.sp.x')
        grid.read_grid()
        grid.compute_metrics()
        flow = FlowIO('../data/plate_data/sol-0000010.q')
        flow.read_flow()

        # Set particle data
        p = Particle()
        p.min_dia = 144e-9  # m
        p.max_dia = 573e-9  # m
        p.mean_dia = 281e-9  # m
        p.std_dia = 97e-9  # m
        p.density = 810  # kg/m3
        p.n_concentration = 25
        p.compute_distribution()

        # Read-in the laser sheet
        laser = LaserSheet(grid)
        laser.position = 0.05  # in m
        laser.thickness = 4e-3  # in m (Data obtained from LaVision)
        laser.pulse_time = 4e-6
        laser.compute_bounds()

        # Create particle locations array
        ia_bounds = [None, None, None, None]
        loc = CreateParticles(grid, flow, p, laser, ia_bounds)
        loc.ia_bounds = [0.3, 0.5, 0.3, 0.5]  # in m
        loc.in_plane = 70
        loc.compute_locations()
        loc.compute_locations2_serial()

        # Sample code to plot particle locations and relative diameters
        _in_plane = int(p.n_concentration * loc.in_plane * 0.01)
        # plot in-plane particle locations
        plt.scatter(loc.locations[:_in_plane, 0], loc.locations[:_in_plane, 1],
                    s=10 * loc.locations[:_in_plane, 3] / p.min_dia, c='g')
        # plot out-of-plane locations
        plt.scatter(loc.locations[_in_plane:, 0], loc.locations[_in_plane:, 1],
                    s=10 * loc.locations[_in_plane:, 3] / p.min_dia, c='r')

        # Create particle projections (Simulating data from EUROPIV)
        proj = CCDProjection(loc)
        proj.d_ccd = 900  # in m
        proj.d_ia = 1000  # in m
        proj.dpi = 72
        proj.compute()

        # Sample code to plot particle locations and relative diameters
        _in_plane = int(p.n_concentration * loc.in_plane * 0.01)
        # plot in-plane particle locations
        plt.figure()
        plt.scatter(proj.projections[:_in_plane, 0], proj.projections[:_in_plane, 1], c='g',
                    s=10 * proj.projections[:_in_plane, 2] / p.min_dia)
        # plot out-of-plane locations
        plt.scatter(proj.projections[_in_plane:, 0], proj.projections[_in_plane:, 1], c='r',
                    s=10 * proj.projections[_in_plane:, 2] / p.min_dia)
        plt.title('Projected data - Not scaled')
        cache = (proj.projections[:, 2], proj.projections[:, 2],
                 proj.projections[:, 0], proj.projections[:, 1],
                 0.5, 0.5, 1.0, 1.0,
                 2, 1, loc.locations[:, 2])
        intensity = Intensity(cache, proj)
        intensity.compute_serial()

        # Creating temp arrays to test. This will be done internally in the code of image_gen
        xp, yp = proj.projections[:, 0], proj.projections[:, 1]
        x = np.linspace(-proj.xres / 2, proj.xres / 2, proj.xres)
        y = np.linspace(-proj.yres / 2, proj.yres / 2, proj.yres)
        xsize = proj.xres / proj.dpi
        ysize = proj.yres / proj.dpi

        # Plot the contour
        fig = plt.figure(figsize=[xsize, ysize], dpi=proj.dpi)
        ax = plt.axes([0.0, 0.0, 1.0, 1.0], xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
        ax.contourf(x, y, intensity.values, cmap='hot')

        # Plot another to look at the scale
        plt.figure()
        plt.contourf(x, y, intensity.values)

        # plot image to finalize the image
        fig = plt.figure(figsize=[xsize, ysize], dpi=proj.dpi)
        ax = plt.axes([0.0, 0.0, 1.0, 1.0])
        ax.imshow(intensity.values, cmap='gray', origin='lower')
        plt.show()


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np


class TestImageGen(unittest.TestCase):
    def test_image_gen(self):
        from src.sypivlib.function.dataio import GridIO, FlowIO
        from src.sypivlib.sypiv.create_particles import Particle, LaserSheet, CreateParticles
        from src.sypivlib.sypiv.ccd_projection import CCDProjection
        from src.sypivlib.sypiv.intensity import Intensity
        from src.sypivlib.sypiv.image_gen import ImageGen

        # Read-in the grid and flow file
        grid = GridIO('../data/shocks/interpolated_data/mgrd_to_p3d.x')
        grid.read_grid()
        grid.compute_metrics()
        flow = FlowIO('../data/shocks/interpolated_data/mgrd_to_p3d_particle.q')
        flow.read_flow()

        # Set particle data
        p = Particle()
        p.min_dia = 1940e-9  # m
        p.max_dia = 1940e-9  # m
        p.mean_dia = 1940e-9  # m
        p.std_dia = 0  # m
        p.density = 810  # kg/m3
        p.n_concentration = 2
        p.compute_distribution()

        # Read-in the laser sheet
        laser = LaserSheet(grid)
        # z-location
        laser.position = 0.00025  # in m
        laser.thickness = 0.0001  # in m (Data obtained from LaVision)
        laser.pulse_time = 1e-9
        laser.compute_bounds()

        # path to save files
        path = '../data/shocks/interpolated_data/particle_snaps/'

        for i in range(1):
            # Create particle locations array
            ia_bounds = [None, None, None, None]
            loc = CreateParticles(grid, flow, p, laser, ia_bounds)
            # x_min, x_max, y_min, y_max --> ia_bounds
            loc.ia_bounds = [0.0016, 0.0025, 0.0002, 0.0004]  # in m
            loc.in_plane = 100
            loc.compute_locations()
            # To adjust for the test case; set locations manually
            # Two locations out-of-plane case
            # loc.locations = np.array([[0.0016+0.000225, 0.0003, 0.00025, 1940e-9],
            #                           [0.0025-0.000225, 0.0003, 0.00025, 1840e-9]])
            # one location center of the plane
            loc.locations = np.array([[0.0025-0.000225*2, 0.0003, 0.00025, p.mean_dia]])
            loc.compute_locations2()

            # Create particle projections (Simulating data from EUROPIV)
            proj = CCDProjection(loc)
            proj.dpi = 72
            proj.xres = 16
            proj.yres = 16
            # Set distance based on similar triangles relationship
            proj.d_ccd = proj.xres * 25.4e-3 / proj.dpi  # in m
            proj.d_ia = 0.0009  # in m; ia_bounds (max - min)
            proj.compute()

            # (radiusx, radiusy, xp, yp, sx, sy, frx, fry, s, q, z_physical)
            cache = (proj.projections[:, 2], proj.projections[:, 2],
                     proj.projections[:, 0], proj.projections[:, 1],
                     2.0, 2.0, 1.0, 1.0,
                     2, 1, loc.locations[:, 2])  # 2 is gaussian profile, 1 is reflectivity factor, z_physical
            intensity = Intensity(cache, proj)
            intensity.compute()

            snap = ImageGen(intensity)
            snap.snap(snap_num=1)
            # snap.save_snap(fname=path + str(i) + '_1.tif')
            snap.check_data(snap_num=1)
            print('Done with image 1 for pair number ' + str(i) + '\n')

            cache2 = (proj.projections2[:, 2], proj.projections2[:, 2],
                     proj.projections2[:, 0], proj.projections2[:, 1],
                     2.0, 2.0, 1.0, 1.0,
                     2, 1, loc.locations2[:, 2])
            intensity2 = Intensity(cache2, proj)
            intensity2.compute()
            #
            snap2 = ImageGen(intensity2)
            snap2.snap(snap_num=2)
            # snap2.save_snap(fname=path + str(i) + '_2.tif')
            snap2.check_data(snap_num=2)

            print('Done with image 2 for pair number ' + str(i) + '\n')


if __name__ == '__main__':
    unittest.main()

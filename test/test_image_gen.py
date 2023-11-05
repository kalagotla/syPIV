import unittest


class TestImageGen(unittest.TestCase):
    def test_image_gen(self):
        from src.dataio import GridIO, FlowIO
        from src.create_particles import Particle, LaserSheet, CreateParticles
        from src.ccd_projection import CCDProjection
        from src.intensity import Intensity
        from src.image_gen import ImageGen

        # Read-in the grid and flow file
        grid = GridIO('../data/shocks/interpolated_data/mgrd_to_p3d.x')
        grid.read_grid()
        grid.compute_metrics()
        flow = FlowIO('../data/shocks/interpolated_data/mgrd_to_p3d_particle.q')
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
            loc.in_plane = 70
            loc.compute_locations()
            loc.compute_locations2()

            # Create particle projections (Simulating data from EUROPIV)
            proj = CCDProjection(loc)
            proj.dpi = 72
            proj.xres = 1024
            proj.yres = 1024
            # Set distance based on similar triangles relationship
            proj.d_ccd = proj.xres * 25.4e-3 / proj.dpi  # in m
            proj.d_ia = 0.0009  # in m; ia_bounds (max - min)
            proj.compute()

            cache = (proj.projections[:, 2], proj.projections[:, 2],
                     proj.projections[:, 0], proj.projections[:, 1],
                     2.0, 2.0, 1.0, 1.0)
            intensity = Intensity(cache, proj)
            intensity.setup()
            intensity.compute()

            snap = ImageGen(intensity)
            snap.snap(snap_num=1)
            # snap.save_snap(fname=path + str(i) + '_1.tif')
            snap.check_data(snap_num=1)
            print('Done with image 1 for pair number ' + str(i) + '\n')

            cache2 = (proj.projections2[:, 2], proj.projections2[:, 2],
                     proj.projections2[:, 0], proj.projections2[:, 1],
                     2.0, 2.0, 1.0, 1.0)
            intensity2 = Intensity(cache2, proj)
            intensity2.setup()
            intensity2.compute()
            #
            snap2 = ImageGen(intensity2)
            snap2.snap(snap_num=2)
            # snap2.save_snap(fname=path + str(i) + '_2.tif')
            snap2.check_data(snap_num=2)

            print('Done with image 2 for pair number ' + str(i) + '\n')


if __name__ == '__main__':
    unittest.main()

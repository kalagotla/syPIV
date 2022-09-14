import unittest


class TestImageGen(unittest.TestCase):
    def test_image_gen(self):
        from src.dataio import GridIO, FlowIO
        from src.create_particles import Particle, LaserSheet, CreateParticles
        from src.ccd_projection import CCDProjection
        from src.intensity import Intensity
        from src.image_gen import ImageGen

        # Read-in the grid and flow file
        grid = GridIO('../data/shocks/shock_test.sb.sp.x')
        grid.read_grid()
        grid.compute_metrics()
        flow = FlowIO('../data/shocks/shock_test.sb.sp.q')
        flow.read_flow()

        # Set particle data
        p = Particle()
        p.min_dia = 144e-9  # m
        p.max_dia = 573e-9  # m
        p.mean_dia = 281e-9  # m
        p.std_dia = 97e-9  # m
        p.density = 810  # kg/m3
        p.n_concentration = 5000
        p.compute_distribution()

        # Read-in the laser sheet
        laser = LaserSheet(grid)
        laser.position = 0.0009  # in m
        laser.thickness = 0.0001  # in m (Data obtained from LaVision)
        laser.pulse_time = 1e-9
        laser.compute_bounds()

        # Create particle locations array
        ia_bounds = [None, None, None, None]
        loc = CreateParticles(grid, flow, p, laser, ia_bounds)
        loc.ia_bounds = [0, 0.001, 0, 0.001]  # in m
        loc.in_plane = 70
        loc.compute_locations()
        loc.compute_locations2()

        # Create particle projections (Simulating data from EUROPIV)
        proj = CCDProjection(loc)
        proj.d_ccd = 14  # in m
        proj.d_ia = 1  # in m
        proj.dpi = 960
        proj.xres = 512
        proj.yres = 512
        proj.compute()

        cache = (proj.projections[:, 2], proj.projections[:, 2],
                 proj.projections[:, 0], proj.projections[:, 1],
                 2.0, 2.0, 1.0, 1.0)
        intensity = Intensity(cache, proj)
        intensity.setup()
        intensity.compute()

        snap = ImageGen(intensity)
        snap.snap(snap_num=1)
        snap.save_snap(fname='../../snap_1.tif')
        # snap.check_data(snap_num=1)

        cache2 = (proj.projections2[:, 2], proj.projections2[:, 2],
                 proj.projections2[:, 0], proj.projections2[:, 1],
                 2.0, 2.0, 1.0, 1.0)
        intensity2 = Intensity(cache2, proj)
        intensity2.setup()
        intensity2.compute()
        #
        snap2 = ImageGen(intensity2)
        snap2.snap(snap_num=2)
        snap2.save_snap(fname='../../snap_2.tif')
        # snap2.check_data(snap_num=2)


if __name__ == '__main__':
    unittest.main()

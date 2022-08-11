import unittest


class TestImageGen(unittest.TestCase):
    def test_image_gen(self):
        from src.dataio import GridIO, FlowIO
        from src.create_particles import Particle, LaserSheet, CreateParticles
        from src.ccd_projection import CCDProjection
        from src.intensity import Intensity
        from src.image_gen import ImageGen

        # Read-in the grid and flow file
        grid = GridIO('../data/plate_data/plate.sp.x')
        grid.read_grid()
        flow = FlowIO('../data/plate_data/sol-0000010.q')
        flow.read_flow()

        # Set particle data
        p = Particle()
        p.min_dia = 144e-9  # m
        p.max_dia = 573e-9  # m
        p.mean_dia = 281e-9  # m
        p.std_dia = 97e-9  # m
        p.density = 810  # kg/m3
        p.n_concentration = 2500
        p.compute_distribution()

        # Read-in the laser sheet
        laser = LaserSheet(grid)
        laser.position = 0.05  # in m
        laser.thickness = 4e-3  # in m (Data obtained from LaVision)
        laser.compute_bounds()

        # Create particle locations array
        ia_bounds = [None, None, None, None]
        loc = CreateParticles(grid, flow, p, laser, ia_bounds)
        loc.ia_bounds = [0.3, 0.5, 0.3, 0.5]  # in m
        loc.in_plane = 70
        loc.compute_locations()

        # Create particle projections (Simulating data from EUROPIV)
        proj = CCDProjection(loc)
        proj.d_ccd = 70  # in m
        proj.d_ia = 1000  # in m
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
        # snap.first_snap()
        # snap.save_snap(fname='../../snap_1.tif')
        snap.check_data()


if __name__ == '__main__':
    unittest.main()

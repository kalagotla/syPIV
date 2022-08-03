import unittest
import matplotlib.pyplot as plt


class TestCCDProjection(unittest.TestCase):
    def test_ccd_projection(self):
        from src.dataio import GridIO, FlowIO
        from src.create_particles import Particle, LaserSheet, CreateParticles
        from src.ccd_projection import CCDProjection

        # Read-in the grid and flow file
        grid = GridIO('../data/shocks/shock_test.sb.sp.x')
        grid.read_grid()
        flow = FlowIO('../data/shocks/shock_test.sb.sp.q')
        flow.read_flow()

        # Set particle data
        p = Particle()
        p.min_dia = 144e-9
        p.max_dia = 573e-9
        p.mean_dia = 281e-9
        p.std_dia = 97e-9
        p.density = 810
        p.n_concentration = 5000
        p.compute_distribution()
        # print(p.particle_field)

        # Read-in the laser sheet
        laser = LaserSheet(grid)
        laser.position = 0.05
        laser.thickness = 5e-2
        laser.compute_bounds()
        # print(laser.width)

        # Create particle locations array
        ia_bounds = [None, None, None, None]
        loc = CreateParticles(grid, flow, p, laser, ia_bounds)
        loc.ia_bounds = [0, 0.003, 0, 0.001]
        loc.in_plane = 90
        loc.compute_locations()

        # Create particle projections
        proj = CCDProjection(loc)
        proj.d_ccd = 5e-1  # in meters
        proj.d_ia = 25e-1  # in meters
        proj.compute()

        # Sample code to plot particle locations and relative diameters
        _in_plane = int(p.n_concentration * loc.in_plane * 0.01)
        # plot in-plane particle locations
        plt.scatter(proj.projections[:_in_plane, 0], proj.projections[:_in_plane, 1], c='g', s=5)
        # plot out-of-plane locations
        plt.scatter(proj.projections[_in_plane:, 0], proj.projections[_in_plane:, 1], c='r', s=5)

        plt.show()


if __name__ == '__main__':
    unittest.main()

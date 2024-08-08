import unittest
import matplotlib.pyplot as plt


class TestCCDProjection(unittest.TestCase):
    def test_ccd_projection(self):
        from src.sypivlib.function.dataio import GridIO, FlowIO
        from src.sypivlib.sypiv.create_particles import Particle, LaserSheet, CreateParticles
        from src.sypivlib.sypiv.ccd_projection import CCDProjection

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
        p.n_concentration = 50
        p.compute_distribution()
        # print(p.particle_field)

        # Read-in the laser sheet
        laser = LaserSheet(grid)
        laser.position = 0.05  # in m
        laser.thickness = 4e-3  # in m (Data obtained from LaVision)
        laser.pulse_time = 1e-5
        laser.compute_bounds()
        # print(laser.width)

        # Create particle locations array
        ia_bounds = [None, None, None, None]
        loc = CreateParticles(grid, flow, p, laser, ia_bounds)
        loc.ia_bounds = [0.3, 0.5, 0.3, 0.5]
        loc.in_plane = 70
        loc.compute_locations()
        loc.compute_locations2_serial()

        # Create particle projections (Simulating data from EUROPIV)
        proj = CCDProjection(loc)
        proj.d_ccd = 1000  # in mm
        proj.d_ia = 100  # in mm
        proj.compute()

        # Sample code to plot particle locations and relative diameters
        _in_plane = int(p.n_concentration * loc.in_plane * 0.01)
        # plot in-plane particle locations
        plt.scatter(proj.projections[:_in_plane, 0], proj.projections[:_in_plane, 1], c='g', s=5)
        # plot out-of-plane locations
        plt.scatter(proj.projections[_in_plane:, 0], proj.projections[_in_plane:, 1], c='r', s=5)

        plt.figure()
        # plot in-plane particle locations
        plt.scatter(proj.projections2[:_in_plane, 0], proj.projections2[:_in_plane, 1], c='g', s=5)
        # plot out-of-plane locations
        plt.scatter(proj.projections2[_in_plane:, 0], proj.projections2[_in_plane:, 1], c='r', s=5)

        plt.show()


if __name__ == '__main__':
    unittest.main()

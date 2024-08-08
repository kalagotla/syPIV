import unittest
import matplotlib.pyplot as plt


class TestCreateParticles(unittest.TestCase):
    def test_create_particles(self):
        from src.sypivlib.function.dataio import GridIO, FlowIO
        from src.sypivlib.sypiv.create_particles import Particle, LaserSheet, CreateParticles

        # Read-in the grid and flow file
        grid = GridIO('../data/shocks/shock_test.sb.sp.x')
        grid.read_grid()
        grid.compute_metrics()
        flow = FlowIO('../data/shocks/shock_test.sb.sp.q')
        flow.read_flow()

        # Set particle data
        p = Particle()
        p.min_dia = 144e-9
        p.max_dia = 573e-9
        p.mean_dia = 281e-9
        p.std_dia = 97e-9
        p.density = 810
        p.n_concentration = 500
        p.compute_distribution()
        # print(p.particle_field)

        # Read-in the laser sheet
        laser = LaserSheet(grid)
        laser.position = 0.0009
        laser.thickness = 0.0001  # Adjusted based on grid thickness
        laser.pulse_time = 1e-7
        laser.compute_bounds()
        # print(laser.width)

        # Create particle locations array
        ia_bounds = [None, None, None, None]
        loc = CreateParticles(grid, flow, p, laser, ia_bounds)
        loc.ia_bounds = [0, 0.003, 0, 0.001]
        loc.in_plane = 90
        loc.compute_locations()
        loc.compute_locations2_serial()

        # Sample code to plot particle locations and relative diameters
        _in_plane = int(p.n_concentration * loc.in_plane * 0.01)
        # plot in-plane particle locations
        plt.scatter(loc.locations[:_in_plane, 0], loc.locations[:_in_plane, 1],
                    s=10*loc.locations[:_in_plane, 3]/p.min_dia, c='g')
        # plot out-of-plane locations
        plt.scatter(loc.locations[_in_plane:, 0], loc.locations[_in_plane:, 1],
                    s=10*loc.locations[_in_plane:, 3]/p.min_dia, c='r')
        plt.xlim([-0.0001, 0.004])
        plt.ylim([0, 0.0019])
        plt.title('Particles created on the given IA for the first snap')

        # plt.show()

        # plot in-plane particle locations
        plt.figure()
        plt.scatter(loc.locations2[:_in_plane, 0], loc.locations2[:_in_plane, 1],
                    s=10 * loc.locations2[:_in_plane, 3] / p.min_dia, c='g')
        # plot out-of-plane locations
        plt.scatter(loc.locations2[_in_plane:, 0], loc.locations2[_in_plane:, 1],
                    s=10 * loc.locations2[_in_plane:, 3] / p.min_dia, c='r')
        plt.xlim([-0.0001, 0.004])
        plt.ylim([0, 0.0019])
        plt.title('Particles created on the given IA for the second snap')

        plt.show()


if __name__ == '__main__':
    unittest.main()

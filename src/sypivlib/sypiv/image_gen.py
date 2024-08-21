# Generate first and second images imitating PIV

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI


class ImageGen:
    """
    Class to generate images
    """

    def __init__(self, intensity):
        self.intensity = intensity
        self.fig = None
        self.ax = None
        self.snap_num = None

    def snap(self, snap_num=0):
        """
        Function to generate first snap from the intensity data
        :return:
        """
        self.snap_num = snap_num
        # Specified in inches --> python default
        xsize = self.intensity.projection.xres / self.intensity.projection.dpi
        ysize = self.intensity.projection.yres / self.intensity.projection.dpi
        self.fig = plt.figure(figsize=[xsize, ysize], dpi=self.intensity.projection.dpi)
        ax = plt.axes([0.0, 0.0, 1.0, 1.0])
        ax.imshow(self.intensity.values, cmap='gray', origin='lower')
        ax.axis('tight')
        # remove black border
        ax.axis('off')
        # plt.show()

        return

    def snap_mpi(self, snap_num=0):
        """
        Function to generate first snap from the intensity data
        :return:
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.snap_num = snap_num
        if rank == 0:
            # Specified in inches --> python default
            xsize = self.intensity.projection.xres / self.intensity.projection.dpi
            ysize = self.intensity.projection.yres / self.intensity.projection.dpi
            self.fig = plt.figure(figsize=[xsize, ysize], dpi=self.intensity.projection.dpi)
            ax = plt.axes([0.0, 0.0, 1.0, 1.0])
            ax.imshow(self.intensity.values, cmap='gray', origin='lower')
            ax.axis('tight')
            # remove black border
            ax.axis('off')

        return

    def save_snap(self, fname=None):
        if fname is None:
            fname = "snap_" + str(self.snap_num) + ".tif"
        self.fig.savefig(fname=fname)
        # close the figure
        plt.close(self.fig)

        return

    def save_snap_mpi(self, fname=None):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            if fname is None:
                fname = "snap_" + str(self.snap_num) + ".tif"
            self.fig.savefig(fname=fname)
            # close the figure
            plt.close(self.fig)

        return

    def check_data(self, snap_num=0):
        """
        Function to generate images during the image gen process
        1. Create Particles in physical space
        2. Projections of particles onto CCD
        3. Contour plot of intensity
        4. Image of the particle snaps - This can be saved!
        :return:
        """
        print("Data at various steps during image generation will be plotted...")

        # Plot data after creating particles
        # Use the percent provided to count number of particles
        plt.figure()
        particles = self.intensity.projection.particles
        _in_plane = int(particles.particle.n_concentration *
                        particles.in_plane * 0.01)
        # plot in-plane particle locations
        plt.scatter(particles.locations[:_in_plane, 0], particles.locations[:_in_plane, 1],
                    s=10 * particles.locations[:_in_plane, 3] / particles.particle.min_dia, c='g', label='in-plane')
        # plot out-of-plane locations
        plt.scatter(particles.locations[_in_plane:, 0], particles.locations[_in_plane:, 1],
                    s=10 * particles.locations[_in_plane:, 3] / particles.particle.min_dia, c='r', label='out-of-plane')
        plt.title('Particles created on the given IA')
        plt.legend()

        # Plot data after particle projection
        plt.figure()
        proj = self.intensity.projection
        plt.scatter(proj.projections[:_in_plane, 0], proj.projections[:_in_plane, 1], c='g',
                    s=10 * proj.projections[:_in_plane, 2] / particles.particle.min_dia, label='in-plane')
        # plot out-of-plane locations
        plt.scatter(proj.projections[_in_plane:, 0], proj.projections[_in_plane:, 1], c='r',
                    s=10 * proj.projections[_in_plane:, 2] / particles.particle.min_dia, label='out-of-plane')
        plt.title('Projected data; shown in pixels; particle sizes are representative;\n data is mirrored')
        plt.legend()

        # Plot data in contours
        xsize = proj.xres / self.intensity.projection.dpi
        ysize = proj.yres / self.intensity.projection.dpi
        x = np.linspace(-proj.xres / 2, proj.xres / 2, proj.xres)
        y = np.linspace(-proj.yres / 2, proj.yres / 2, proj.yres)
        fig = plt.figure(figsize=[xsize, ysize], dpi=proj.dpi)
        ax = plt.axes([0.0, 0.0, 1.0, 1.0], xlim=(min(x), max(x)), ylim=(min(y), max(y)))
        ax.contourf(x, y, self.intensity.values, cmap='hot')
        ax.set_title('Contour plot of intensities')

        # Show the image
        self.snap(snap_num=snap_num)
        plt.show()

        return



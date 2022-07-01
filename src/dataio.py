# Read 2D planar data for image generation

class DataIO:
    """
    Module to read 2D planar ordered data generated from project-arrakis

    ...

    Attributes
    ----------
    Input :
        filename : str
            name of the data file generated from project-arrakis
    Output :
        df : dataframe (dask)
            dataframe with grid and flow data
    """

    def __init__(self, folderpath):
        self.folderpath = folderpath
        self.df = None
        self.psize = None

    def read_data(self):
        import dask.dataframe as dd
        import re

        # x --> grid co-ordinates
        # u --> fluid velocity
        # v --> particle velocity

        columns = ['x0', 'x1', 'x2',
                   'v0', 'v1', 'v2',
                   'u0', 'u1', 'u2']

        self.psize = list(map(int, re.findall(r'\d+', self.folderpath)))[-1]
        self.df = dd.read_csv(self.folderpath, delim_whitespace=True)
        if self.df.shape[1] <= len(columns):
            # If particle velocities are not available; line below adjusts number of columns
            self.df.columns = columns[len(self.df.columns)]
        else:
            self.df = self.df.iloc[:, :len(columns)]
            self.df.columns = columns

        return

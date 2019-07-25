import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

from ..base import Base


class GaussianProcess(Base):
    """
    Class to interpolate by fitting a gaussian process to given
    data.
    """

    def __init__(self, resolution="standard", coordinate_type="Euclidean", **kwargs):
        super().__init__(resolution, coordinate_type)
        self.reg = GaussianProcessRegressor(normalize_y=True, kernel=Matern())

    def _fit(self, X, y):
        """Function for fitting.
        This function is not supposed to be called directly.
        """
        self.reg.fit(X, y)
        return self

    def _predict_grid(self, x1lim, x2lim):
        """Function for grid interpolation.
        This function is not supposed to be called directly.
        """
        # getting the boundaries for interpolation
        x1min, x1max = x1lim
        x2min, x2max = x2lim
        
        # building the grid
        x1 = np.linspace(x1min, x1max, self.resolution)
        x2 = np.linspace(x2min, x2max, self.resolution)
        X1, X2 = np.meshgrid(x1, x2)
        return self.reg.predict(np.asarray([X1.ravel(), X2.ravel()]).T)

    def _predict(self, X):
        """Function for interpolation on specific points.
        This function is not supposed to be called directly.
        """
        return self.reg.predict(X)

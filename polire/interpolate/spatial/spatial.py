import numpy as np

from ..base import Base
from ...utils.distance import euclidean, haversine

class SpatialAverage(Base):
    """
Spatial averaging is a simple interpolation method that predicts the target value as the mean of other known target values; known points that lie within a set distance from the unknown point are used in averaging.
    Note that radius you specify must be in kilometres if you are passing latitude and longitude as inputs
    """

    def __init__(
        self, radius=100, resolution="standard", coordinate_type="Euclidean", **kwargs
    ):
        super().__init__(resolution, coordinate_type)
        self.radius = radius
        if self.coordinate_type == 'Geographic':
            self.distance = haversine
        elif self.coordinate_type == 'Euclidean':
            self.distance = euclidean
        else:
            raise NotImplementedError("Only Geographic and Euclidean Coordinates are available")
    def _fit(self, X, y):
        """Function for fitting.
        This function is not supposed to be called directly.
        """
        self.X = X
        self.y = y
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
        return self._predict(np.asarray([X1.ravel(), X2.ravel()]).T)

    def _predict(self, X):
        """Function for interpolation on specific points.
        This function is not supposed to be called directly.
        """
        return self._average(X)

    def _average(self, X):
        y_pred = []
        for ix in range(X.shape[0]):
            dist = self.distance(X[ix], self.X)
            mask = self.radius >= dist
            # print ('mask', mask)
            points_within_rad = mask.sum()
            # print ('points_within_rad', points_within_rad)
            y_pred.append(sum(self.y[mask]) / points_within_rad)
        return np.asarray(y_pred)

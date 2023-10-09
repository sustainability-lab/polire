"""
This is a module for inverse distance weighting (IDW) Spatial Interpolation
"""
import numpy as np
from ..utils.distance import haversine, euclidean
from ..base import Base


class IDW(Base):
    """A class that is declared for performing IDW Interpolation.
    For more information on how this method works, kindly refer to
    https://en.wikipedia.org/wiki/Inverse_distance_weighting

    Parameters
    ----------
    exponent : positive float, optional
        The rate of fall of values from source data points.
        Higher the exponent, lower is the value when we move
        across space. Default value is 2.

    Attributes
    ----------
    Interpolated Values : {array-like, 2D matrix}, shape(resolution, resolution)
    This contains all the interpolated values when the interpolation is performed
    over a grid, instead of interpolation over a set of points.

    X : {array-like, 2D matrix}, shape(n_samples, 2)
    Set of all the coordinates available for interpolation.

    y : array-like, shape(n_samples,)
    Set of all the available values at the specified X coordinates.

    result : array_like, shape(n_to_predict, )
    Set of all the interpolated values when interpolating over a given
    set of data points.

    """

    def __init__(
        self, exponent=2, resolution="standard", coordinate_type="Euclidean"
    ):
        super().__init__(resolution, coordinate_type)
        self.exponent = exponent
        self.interpolated_values = None
        self.X = None
        self.y = None
        self.result = None
        if self.coordinate_type == "Geographic":
            self.distance = haversine
        elif self.coordinate_type == "Euclidean":
            self.distance = euclidean
        else:
            raise NotImplementedError(
                "Only Geographic and Euclidean Coordinates are available"
            )

    def _fit(self, X, y):
        """This function is for the IDW Class.
        This is not expected to be called directly
        """
        self.X = X
        self.y = y
        return self

    def _predict_grid(self, x1lim, x2lim):
        """Gridded interpolation for natural neighbors interpolation. This function should not
        be called directly.
        """
        lims = (*x1lim, *x2lim)
        x1min, x1max, x2min, x2max = lims
        x1 = np.linspace(x1min, x1max, self.resolution)
        x2 = np.linspace(x2min, x2max, self.resolution)
        X1, X2 = np.meshgrid(x1, x2)
        return self._predict(np.array([X1.ravel(), X2.ravel()]).T)

    def _predict(self, X):
        """The function call to predict using the interpolated data
        in IDW interpolation. This should not be called directly.
        """
        
        dist = self.distance(self.X, X)
        for i in range(len(dist)):
            for j in range(len(dist[0])):
                if (dist[i][j]==0):
                    dist[i][j] = 1.0e-10

        weights = 1 / np.power(dist, self.exponent)
        result = (weights * self.y[:, None]).sum(axis=0) / weights.sum(axis=0)

        # if point is from train data, ground truth must not change
        # for i in range(X.shape[0]):
        #     mask = np.equal(X[i], self.X).all(axis=1)
        #     if mask.any():
        #         result[i] = (self.y * mask).sum()

        return result

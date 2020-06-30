r"""
This is a module for IDW Spatial Interpolation
"""
import numpy as np
from ...utils.distance import haversine, euclidean
from ..base import Base
from copy import deepcopy


def is_row_in_array(row, arr):
    return list(row) in arr.tolist()


def get_index(row, arr):
    t1 = np.where(arr[:, 0] == row[0])
    t2 = np.where(arr[:, 1] == row[1])
    index = np.intersect1d(t1, t2)[0]
    # If length of index exceeds one!! - Uniqueness Error
    return index


class Idw(Base):
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

    def __init__(self, exponent=2, resolution="standard", coordinate_type="Euclidean"):
        super().__init__(resolution, coordinate_type)
        self.exponent = exponent
        self.interpolated_values = None
        self.X = None
        self.y = None
        self.result = None
        if self.coordinate_type == 'Geographic':
            self.distance = haversine
        elif self.coordinate_type == 'Euclidean':
            self.distance = euclidean
        else:
            raise NotImplementedError("Only Geographic and Euclidean Coordinates are available")

    def _fit(self, X, y):
        """This function is for the IDW Class.
        This is not expected to be called directly
        """
        self.X = X
        self.y = y
        return self

    def _predict_grid(self, x1lim, x2lim):
        """ Gridded interpolation for natural neighbors interpolation. This function should not
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
        result = np.zeros(X.shape[0])
        for i in range(len(X)):
            point = X[i]

            # Preserve point estimates. This is mandatory in IDW
            flag = is_row_in_array(point, self.X)

            if flag:
                index = get_index(point, self.X)
                result[i] = self.y[index]
            else:
                weights = 1 / (self.distance(point, self.X) ** self.exponent)
                result[i] = np.multiply(self.y.reshape(self.y.shape[0],), weights).sum() / (weights.sum())
        self.result = result
        return self.result.reshape(result.shape[0], -1)

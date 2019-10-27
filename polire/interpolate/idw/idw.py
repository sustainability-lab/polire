r"""
This is a module for IDW Spatial Interpolation
"""
import numpy as np
from ...utils import gridding
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

    def __init__(self, exponent=2, resolution="standard", coordinate_type="Eucledian"):
        super().__init__(resolution, coordinate_type)
        self.exponent = exponent
        self.interpolated_values = None
        self.X = None
        self.y = None
        self.result = None

    def _fit(self, X, y):
        """This function is for the IDW Class.
        This is not expected to be called directly
        """
        self.X = X
        self.y = y
        return self

    def _predict_grid(self, x1lim, x2lim):
        lims = (*x1lim, *x2lim)
        # X = deepcopy(np.c_[X,y])

        X = np.c_[self.X, self.y]
        x1min, x1max, x2min, x2max = lims
        x1 = np.linspace(x1min, x1max, self.resolution)
        x2 = np.linspace(x2min, x2max, self.resolution)
        xx, yy = np.meshgrid(x1, x2)

        new = []
        # This list stores all the tuples that contain points, and the closest point to the point in the 2D Grid that we are concerned with.

        for points in X:
            # Check the closest point in the grid, corresponding to the source data point.
            # We assume that the source is located at those exact coordinates on the grid.
            # IDW is done using these source data point locations, while computing distance.

            min_dist = np.inf
            val = 0
            for j in range(len(yy)):
                temp = yy[j, 0]
                for i in range(len(xx[0])):
                    # Just checks for the closest location
                    dist = np.linalg.norm(np.array([xx[0][i], temp]) - points[:2])
                    if dist < min_dist:
                        min_dist = dist
                        val = (i, j)
            new.append((points, val))
            # New now contains all the points that we're concerned with.

        new_grid = np.zeros((len(xx), len(yy)))
        for i in range(len(new)):
            # Storing the source data point values into the corresponding grid locations.
            # As from above, we store all the values needed in the new_list.
            x = new[i][1][0]
            y = new[i][1][1]
            if new[i][0][2] == 0.0:
                new_grid[x][y] = 1e-20
                ## This is to ensure the constraint that the input data is actually zero somewhere -- rare case...
                ## We can approxiamte it very well
            else:
                new_grid[x][y] = new[i][0][2]
        x_nz, y_nz = np.nonzero(new_grid)

        # This list_nz contains all the non -zero points or the source data points that we will have
        list_nz = []
        for i in range(len(x_nz)):
            list_nz.append((x_nz[i], y_nz[i]))

        final = deepcopy(new_grid)
        ## The final grid that we create will have all the needed interpolated values
        for i in range(len(xx[0])):
            for j in range(len(yy)):
                normalise = 0
                if (i, j) in list_nz:
                    ## These are the source data points. We can't interpolate here.
                    # The point estimates have to be preserved
                    continue
                else:
                    for elem in range(len(x_nz)):
                        # Looping through every non-zero location
                        source = np.array([x_nz[elem], y_nz[elem]])
                        target = np.array([xx[0][i], yy[j][0]])
                        ## Here we use the dist to compute the interpolated value, based on the exponent that the user provides.
                        dist = (
                            np.abs(xx[0][source[0]] - target[0]) ** self.exponent
                            + np.abs(yy[source[1]][0] - target[1]) ** self.exponent
                        ) ** (1 / self.exponent)
                        final[i][j] += new_grid[x_nz[elem], y_nz[elem]] / dist
                        normalise += 1 / (dist)
                # We divide by the normalise factor, as per the definiton of the inverse distance weighted interpolation
                final[i][j] /= normalise
        self.interpolated_values = final
        self.x_grid = xx
        self.y_grid = yy

        # Return the self object -- This is useful if we want to do something like = object.fit().predict() etc.,
        return self.interpolated_values

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
                weights = np.array(
                    [
                        1 / (np.linalg.norm(point - self.X[j]) ** self.exponent)
                        for j in range(len(self.X))
                    ]
                )
                result[i] = np.multiply(self.y.reshape(self.y.shape[0],), weights).sum() / (weights.sum())
        self.result = result
        return self.result

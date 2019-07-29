import numpy as np
from scipy.interpolate import bisplrep, bisplev


from ..base import Base
from ...utils import find_closest


class BSpline(Base):
    """
    Class to use a bivariate B-spline to interpolate values.
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.interpolate.bisplrep.html#scipy.interpolate.bisplrep
    
    Parameters
    ----------
    kx, ky: int, int, optional
        The degrees of the spline (1 <= kx, ky <= 5). 
        Third order (kx=ky=3) is recommended.

    s : float, optional
        A non-negative smoothing factor. If weights correspond
        to the inverse of the standard-deviation of the errors
        in z, then a good s-value should be found in the
        range `(m-sqrt(2*m),m+sqrt(2*m))` where `m=len(x)`.
    """

    def __init__(
        self, kx=3, ky=3, s=None, resolution="standard", coordinate_type="Euclidean"
    ):
        super().__init__(resolution, coordinate_type)
        self.kx = kx
        self.ky = ky
        self.s = s

    def _fit(self, X, y):
        """The function call to fit the spline model on the given data.
        This function is not supposed to be called directly.        
        """
        # fitting the curve
        # bisplrep returns details of the fitted curve
        # read bisplrep docs for more info about it's return values.
        self.tck = bisplrep(X[:, 0], X[:, 1], y, kx=self.kx, ky=self.ky, s=self.s)
        return self

    def _predict_grid(self, x1lim, x2lim):
        """The function to predict grid interpolation using the BSpline.
        This function is not supposed to be called directly.
        """
        # getting the boundaries for interpolation
        x1min, x1max = x1lim
        x2min, x2max = x2lim

        # interpolating over the grid
        # TODO Relook here, we might expect the result to be transpose
        return bisplev(
            np.linspace(x1min, x1max, self.resolution),
            np.linspace(x2min, x2max, self.resolution),
            self.tck,
        )

    def _predict(self, X):
        """The function to predict using the BSpline interpolation.
        This function is not supposed to be called directly.
        """
        results = []
        for ix in range(X.shape[0]):
            interpolated_y = bisplev(
                X[ix, 0], X[ix, 1], self.tck
            ).item()  # one value returned
            results.append(interpolated_y)

        return np.array(results)

        # # form a grid
        # x1 = np.linspace(self.x1min_d, self.x1max_d, self.resolution),
        # x2 = np.linspace(self.x2min_d, self.x2max_d, self.resolution),
        # X1, X2 = np.meshgrid(x1, x2)

        # # be default run grid interpolation on the whole train data
        # interpolated_grid = bisplev(
        #     x1, x2,
        #     self.tck,
        # )

        # # find the closest points on the interpolated grid
        # ix = find_closest(grid=(X1, X2), X)
        # return interpolated_grid[ix] # TODO this can be wrong, must depend on

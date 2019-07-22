from ..constants import RESOLUTION


class Base:
    """A class that is declared for performing Interpolation.
    This class should not be called directly, use one of it's
    children.
    """

    def __init__(self, resolution, coordinate_types):
        self.resolution = RESOLUTION[resolution]
        self.coordinate_type = coordinate_types
        self._fit_called = False

    def fit(self, X, y):
        """The function call to fit the model on the given data.

        Parameters
        ----------

        X: {array-like, 2D matrix}, shape(n_samples, 2)
            The set of all coordinates, where we have ground truth
            values
        y: array-like, shape(n_samples,)
            The set of all the ground truth values using which
            we perform interpolation

        Returns
        -------

        self : object
            Returns self

        """
        # saving that fit was called
        self._fit_called = True

        # saving boundaries
        self.x1min_d = min(X[:, 0])
        self.x1max_d = max(X[:, 0])
        self.x2min_d = min(X[:, 1])
        self.x2max_d = max(X[:, 1])
        return self._fit(X, y)  # calling child specific fit method

    def predict(self, X):
        """The function call to return interpolated data on specific
        points.

        Parameters
        ----------

        X: {array-like, 2D matrix}, shape(n_samples, 2)
            The set of all coordinates, where we have ground truth
            values

        Returns
        -------

        y_pred : array-like, shape(n_samples,)
            The set of interpolated values for the points used to
            call the function.
        """
        # checking if model is fitted or not
        assert self._fit_called, "First call fit method to fit the model"

        # calling child specific _predict method
        return self._predict(X)

    def predict_grid(self, x1lim=None, x2lim=None):
        """Function to interpolate data on a grid of given size.
        .
        Parameters
        ----------
        x1lim: tuple(float, float),
            Upper and lower bound on 1st dimension for the interpolation.

        x2lim: tuple(float, float),
            Upper and lower bound on 2nd dimension for the interpolation.
        
        Returns
        -------
        y: array-like, shape(n_samples,)
            Interpolated values on the grid requested.
        """
        # checking if model is fitted or not
        assert self._fit_called, "First call fit method to fit the model"

        # by default we interpolate over the whole grid
        if x1lim is None:
            x1lim = (self.x1min_d, self.x1max_d)
        if x2lim is None:
            x2lim = (self.x2min_d, self.x2max_d)
        (x1min, x1max) = x1lim
        (x2min, x2max) = x2lim

        # extrapolation isn't supported yet
        assert self.x1min_d >= x1min, "Extrapolation not supported"
        assert self.x1max_d <= x1max, "Extrapolation not supported"
        assert self.x2min_d >= x2min, "Extrapolation not supported"
        assert self.x2max_d <= x2max, "Extrapolation not supported"

        # calling child specific _predict_grid method
        return self._predict_grid(x1lim, x2lim)

    def _fit(self, X, y):
        raise NotImplementedError
    def _predict_grid(self, x1lim, x2lim):
        raise NotImplementedError
    def _predict(self, X):
        raise NotImplementedError
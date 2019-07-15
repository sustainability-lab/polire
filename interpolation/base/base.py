from ..constants import RESOLUTION

class Base:
    """A class that is declared for performing Interpolation.
    This class should not be called directly, use one of it's
    children.
    """
    def __init__(
        self,
        resolution,
        coordinate_types
    ):
        self.resolution = RESOLUTION[resolution]
        self.coordinate_type = coordinate_types

    def fit(self, X, y):
        """ The function call to fit the model on the given data. 
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
        self.x1min_d = min(X[:, 0])
        self.x1max_d = max(X[:, 0])
        self.x2min_d = min(X[:, 1])
        self.x2max_d = max(X[:, 1])
        return self._fit(X, y) # calling child specific fit method

    def predict(self, x1min, x1max, x2min, x2max):
        """ The function call to predict using the interpolated data.
        Parameters
        ----------
        x1min, x1max: float,
            Upper and lower bound on x1 dimention for interpolation.

        x2min, x2max: float,
            Upper and lower bound on x2 dimention for interpolation.
        
        Returns
        -------
        y: array-like, shape(n_samples,)
            The set of all the ground truth values using which
            we perform interpolation.
        """
        assert (self.x1min_d >= x1min), "Extrapolation not supported"
        assert (self.x1max_d <= x1max), "Extrapolation not supported"
        assert (self.x2min_d >= x2min), "Extrapolation not supported"
        assert (self.x2max_d <= x2max), "Extrapolation not supported"
        lims = (x1min, x1max, x2min, x2max)
        return self._predict(lims)
        
    def _fit(self, X, y):
        raise NotImplementedError

    def _predict(self, X, y):
        raise NotImplementedError
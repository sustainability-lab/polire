"""
This is a module for GP Interpolation
"""
import numpy as np
from ..base import Base
from GPy.models import GPRegression
from GPy.kern import RBF


class GP(Base):
    """A class that is declared for performing GP interpolation.
    GP interpolation (usually) works on the principle of finding the 
    best unbiased predictor. 

    Parameters
    ----------
    type : str, optional
    This parameter defines the type of Kriging under consideration. This 
    implementation uses PyKrige package  (https://github.com/bsmurphy/PyKrige).
    The user needs to choose between "Ordinary" and "Universal".

    """

    def __init__(
        self,
        kernel=RBF(2, ARD=True),
    ):

        super().__init__()
        self.kernel = kernel

    def _fit(self, X, y, n_restarts=5, verbose=False, random_state=None):
        """ Fit method for GP Interpolation
        This function shouldn't be called directly.
        """
        np.random.seed(random_state)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.model = GPRegression(X, y, self.kernel)
        self.model.optimize_restarts(n_restarts, verbose=verbose)
        return self

    def _predict_grid(self, x1lim, x2lim):
        """The function that is called to return the interpolated data in Kriging Interpolation
        in a grid. This method shouldn't be called directly"""
        lims = (*x1lim, *x2lim)
        x1min, x1max, x2min, x2max = lims
        x1 = np.linspace(x1min, x1max, self.resolution)
        x2 = np.linspace(x2min, x2max, self.resolution)

        X1, X2 = np.meshgrid(x1, x2)
        X = np.array([(i, j) for i, j in zip(X1.ravel(), X2.ravel())])

        predictions = self.model.predict(X)[0].reshape(len(x1), len(x2))

        return predictions.ravel()

    def _predict(self, X, return_variance=False):
        """This function should be called to return the interpolated data in kriging
        in a pointwise manner. This method shouldn't be called directly."""

        predictions, variance = self.model.predict(X)
        if return_variance:
            return predictions.ravel(), variance
        else:
            return predictions.ravel()

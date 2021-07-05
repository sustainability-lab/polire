import numpy as np

from ..base import Base


class Random(Base):
    """
    Class to randomly interpolate by picking values between maximum and
    minimum measurements.

    Note: Even if a point on the requested grid is present in 
    the training set, we return a random value for it.
    """

    def __init__(self, resolution="standard", coordinate_type="Euclidean"):
        super().__init__(resolution, coordinate_type)

    def _fit(self, X, y):
        """Function for fitting random interpolation.
        This function is not supposed to be called directly.
        """
        self.ymax = max(y)
        self.ymin = min(y)
        return self

    def _predict_grid(self, x1lim, x2lim):
        """Function for random grid interpolation.
        This function is not supposed to be called directly.
        """
        return np.random.uniform(
            low=self.ymin, high=self.ymax, size=(self.resolution, self.resolution)
        )

    def _predict(self, X):
        """Function for random interpolation.
        This function is not supposed to be called directly.
        """
        return np.random.uniform(low=self.ymin, high=self.ymax, size=(X.shape[0]))

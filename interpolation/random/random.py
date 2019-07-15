import numpy as np

from ..base import Base


class Random(Base):
    """
    Randomly interpolate by picking values between maximum and
    minimum measurements.

    Note: Even if the point to predict is present in training
    set, we return a random value.
    """
  
    def __init__(
        self,
        resolution = 'standard',
        coordinate_type = 'Euclidean'
    ):
        super().__init__(resolution, coordinate_type)

    def _fit(self, X, y):
        """Function for fitting random interpolation."""
        self.ymax = max(y)
        self.ymin = min(y)
        return self
        
    def _predict(self, lims):
        """Function for random interpolation."""
        x1min, x1max, x2min, x2max = lims
        return np.random.uniform(
            low=self.ymin,
            high=self.ymax,
            size=(self.resolution, self.resolution)
        )

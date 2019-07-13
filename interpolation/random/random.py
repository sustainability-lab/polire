import numpy as np

from ..base import Base


class Random(Base):
    """
    Randomly interpolate by picking values between maximum and
    minimum measurements.
    
    Parameters
    ----------
    resolution: str, optional
        Decides the smoothness of the interpolation. Note that
        interpolation is done over a grid. Higher the resolution
        means more grid cells and more time for interpolation.
        Default value is 'standard'
    coordinate_type: str, optional
        Decides the distance metric to be used, while performing
        interpolation. Euclidean by default.  
    """
  
    def __init__(self):
        super().__init__() # no use

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
        
        self.ymax = max(y)
        self.ymin = min(y)
        return self
        
    def predict(self, X):
        """ The function call to predict using the interpolated data.
        Parameters
        ----------
        X: {array-like, 2D matrix}, shape(n_samples, 2)
            The set of all coordinates, where we have ground truth
            values
        
        Returns
        -------
        y: array-like, shape(n_samples,)
            The set of all the ground truth values using which
            we perform interpolation.
            
        Note: Even if the point to predict is present in training
        set, we return a random value.
        """
        samples, _ = X.shape
        return np.random.uniform(
            low=self.ymin,
            high=self.ymax,
            size=samples
        )

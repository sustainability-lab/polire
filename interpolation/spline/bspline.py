import numpy as np
from scipy.interpolate import bisplrep, bisplev


from ..base import Base


class BSpline(Base):
    """
    Using a bivariate B-spline interpolate values.
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.interpolate.bisplrep.html#scipy.interpolate.bisplrep
    
    Parameters
    ----------
    kx, ky: int, optional
        The degrees of the spline (1 <= kx, ky <= 5). 
        Third order (kx=ky=3) is recommended.

    s : float, optional
        A non-negative smoothing factor. If weights correspond
        to the inverse of the standard-deviation of the errors
        in z, then a good s-value should be found in the
        range `(m-sqrt(2*m),m+sqrt(2*m))` where `m=len(x)`.
    """
  
    def __init__(
        self,
        kx = 3,
        ky = 3,
        s = None,
        resolution = 'standard',
        coordinate_type = 'Euclidean'
    ):
        super().__init__(resolution, coordinate_type)
        self.kx = kx
        self.ky = ky
        self.s = s

    def _fit(self, X, y):
        """ The function call to fit the spline model on the given data. 
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
        # fitting the curve 
        self.tck = bisplrep(
            X[:, 0], X[:, 1], y,
            kx=self.kx, ky=self.ky,
            s=self.s
        )
        return self
        
    def _predict(self, lims):
        """ The function call to predict using the BSpline interpolation.
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
        x1min, x1max, x2min, x2max = lims
        return bisplev(
            np.linspace(x1min, x1max, self.resolution),
            np.linspace(x2min, x2max, self.resolution),
            self.tck
        )
